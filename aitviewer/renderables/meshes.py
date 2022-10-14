"""
Copyright (C) 2022  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import moderngl
import numpy as np
import os
import trimesh
import tqdm
import re
import pickle

from aitviewer.scene.node import Node
from aitviewer.shaders import get_flat_lit_with_edges_face_color_program
from aitviewer.shaders import get_smooth_lit_with_edges_face_color_program
from aitviewer.shaders import get_smooth_lit_with_edges_program
from aitviewer.shaders import get_flat_lit_with_edges_program
from aitviewer.shaders import get_smooth_lit_texturized_program
from aitviewer.utils import set_lights_in_program
from aitviewer.utils import set_material_properties
from aitviewer.utils.decorators import hooked
from aitviewer.utils.so3 import euler2rot_numpy, rot2euler_numpy
from aitviewer.utils.utils import compute_vertex_and_face_normals
from functools import lru_cache
from moderngl_window.opengl.vao import VAO
from PIL import Image
from trimesh.triangles import points_to_barycentric


class Meshes(Node):
    """A sequence of triangle meshes. This assumes that the mesh topology is fixed over the sequence."""

    def __init__(self,
                 vertices,
                 faces,
                 vertex_normals=None,
                 face_normals=None,
                 vertex_colors=None,
                 face_colors=None,
                 uv_coords=None,
                 path_to_texture=None,
                 cast_shadow=True,
                 pickable=True,
                 flat_shading=False,
                 draw_edges=False,
                 draw_outline=False,
                 icon="\u008d",
                 **kwargs):
        """
        Initializer.
        :param vertices: A np array of shape (N, V, 3) or (V, 3).
        :param faces: A np array of shape (F, 3).
        :param vertex_normals: A np array of shape (N, V, 3). If not provided, the vertex normals will be computed,
          which incurs some overhead.
        :param face_normals: A np array of shape (N, F, 3). If not provided, the face normals will be computed, which
          incurs some overhead.
        :param vertex_colors: A np array of shape (N, V, 4) overriding the uniform color.
        :param face_colors: A np array of shape (N, F, 4) overriding the uniform or vertex colors.
        :param uv_coords: A np array of shape (V, 2) if the mesh is to be textured.
        :param path_to_texture: Path to an image file that serves as the texture.
        """
        if len(vertices.shape) == 2 and vertices.shape[-1] == 3:
            vertices = vertices[np.newaxis]
        assert len(vertices.shape) == 3
        assert len(faces.shape) == 2
        super(Meshes, self).__init__(n_frames=vertices.shape[0], icon=icon, **kwargs)

        self._vertices = vertices
        self._faces = faces.astype(np.int32)

        def _maybe_unsqueeze(x):
            return x[np.newaxis] if x is not None and x.ndim == 2 else x

        self._vertex_normals = _maybe_unsqueeze(vertex_normals)
        self._face_normals = _maybe_unsqueeze(face_normals)
        self.vertex_colors = _maybe_unsqueeze(vertex_colors)
        self.face_colors = _maybe_unsqueeze(face_colors)

        # Texture handling.
        self.has_texture = uv_coords is not None
        self.uv_coords = uv_coords

        if self.has_texture:
            self.use_pickle_texture = path_to_texture.endswith((".pickle", "pkl"))
            if self.use_pickle_texture:
                self.texture_image = pickle.load(open(path_to_texture, "rb"))
            else:
                self.texture_image = Image.open(path_to_texture).transpose(method=Image.FLIP_TOP_BOTTOM).convert("RGB")
        else:
            self.texture_image = None

        # Enable rendering passes
        self.cast_shadow = cast_shadow
        self.fragmap = pickable
        self.depth_prepass = True
        self.outline = True

        # Misc.
        self._flat_shading = flat_shading
        self.draw_edges = draw_edges
        self.draw_outline = draw_outline
        self.show_texture = self.has_texture
        self.norm_coloring = False
        self.normals_r = None
        self.need_upload = True
        self._use_uniform_color = self._vertex_colors is None and self._face_colors is None

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        # Update vertices and redraw
        self._vertices = vertices
        self.n_frames = len(vertices)

        # If vertex normals were supplied, they are no longer valid.
        self._vertex_normals = None

        # Must clear all LRU caches where the vertices are used.
        self.compute_vertex_and_face_normals.cache_clear()

        self.redraw()

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, f):
        self._faces = f.astype(np.int32)

    @property
    def current_vertices(self):
        idx = self.current_frame_id if self.vertices.shape[0] > 1 else 0
        return self.vertices[idx]

    @current_vertices.setter
    def current_vertices(self, vertices):
        idx = self.current_frame_id if self.vertices.shape[0] > 1 else 0
        self._vertices[idx] = vertices
        self.compute_vertex_and_face_normals.cache_clear()
        self.redraw()

    @property
    def n_faces(self):
        return self.faces.shape[0]

    @property
    def n_vertices(self):
        return self.vertices.shape[1]

    @property
    def vertex_faces(self):
        # To compute the normals we need to know a mapping from vertex ID to all faces that this vertex is part of.
        # Because we are lazy we abuse trimesh to compute this for us. Not all vertices have the maximum degree, so
        # this array is padded with -1 if necessary.
        return trimesh.Trimesh(self.vertices[0], self.faces, process=False).vertex_faces

    @property
    def vertex_normals(self):
        """Get or compute all vertex normals (this might take a while for long sequences)."""
        if self._vertex_normals is None:
            vertex_normals, _ = compute_vertex_and_face_normals(self.vertices, self.faces, self.vertex_faces,
                                                                normalize=True)
            if vertex_normals.ndim < 3:
                vertex_normals = vertex_normals.unsqueeze(0)
            self._vertex_normals = vertex_normals
        return self._vertex_normals

    @property
    def face_normals(self):
        """Get or compute all face normals (this might take a while for long sequences)."""
        if self._face_normals is None:
            _, face_normals = compute_vertex_and_face_normals(self.vertices, self.faces, self.vertex_faces,
                                                              normalize=True)
            if face_normals.ndim < 3:
                face_normals = face_normals.unsqueeze(0)
            self._face_normals = face_normals
        return self._face_normals

    def vertex_normals_at(self, frame_id):
        """Get or compute the vertex normals at the given frame."""
        if self._vertex_normals is None:
            vn, _ = self.compute_vertex_and_face_normals(frame_id, normalize=True)
        else:
            assert len(self._vertex_normals.shape) == 3, f"Got shape {self._vertex_normals.shape}"
            vn = self._vertex_normals[frame_id]
        return vn

    def face_normals_at(self, frame_id):
        """Get or compute the face normals at the given frame."""
        if self._face_normals is None:
            _, fn = self.compute_vertex_and_face_normals(frame_id, normalize=True)
        else:
            assert len(self._face_normals.shape) == 3, f"Got shape {self._face_normals.shape}"
            fn = self._face_normals[frame_id]
        return fn

    @property
    def vertex_colors(self):
        if self._vertex_colors is None:
            self._vertex_colors = np.full((self.n_frames, self.n_vertices, 4), self.material.color)
        return self._vertex_colors

    @vertex_colors.setter
    def vertex_colors(self, vertex_colors):
        # If vertex_colors are None, we resort to the material color.
        if vertex_colors is None:
            self._vertex_colors = None
            self._use_uniform_color = True
        elif isinstance(vertex_colors, tuple) and len(vertex_colors) == 4:
            self.vertex_colors = None
            self._use_uniform_color = True
            self.material.color = vertex_colors
        else:
            if len(vertex_colors.shape) == 2:
                assert vertex_colors.shape[0] == self.n_vertices
                vertex_colors = np.repeat(vertex_colors[np.newaxis], self.n_frames, axis=0)
            assert len(vertex_colors.shape) == 3
            self._vertex_colors = vertex_colors
            self._use_uniform_color = False
            self.redraw()

    @property
    def current_vertex_colors(self):
        if self._use_uniform_color:
            return np.full((self.n_vertices, 4), self.material.color)
        else:
            idx = self.current_frame_id if self.vertex_colors.shape[0] > 1 else 0
            return self.vertex_colors[idx]

    @property
    def face_colors(self):
        return self._face_colors

    @face_colors.setter
    def face_colors(self, face_colors):
        if face_colors is not None:
            if len(face_colors.shape) == 2:
                face_colors = face_colors[np.newaxis]
            self._face_colors = face_colors
            self._use_uniform_color = False
        else:
            self._face_colors = None
        self.redraw()

    @property
    def current_face_colors(self):
        if self._use_uniform_color:
            return np.full((self.n_faces, 4), self.material.color)
        else:
            idx = self.current_frame_id if self.face_colors.shape[0] > 1 else 0
            return self.face_colors[idx]

    @Node.color.setter
    def color(self, color):
        self.material.color = color

        if self.face_colors is None:
            self.vertex_colors = color

    @property
    def flat_shading(self):
        return self._flat_shading

    @flat_shading.setter
    def flat_shading(self, flat_shading):
        if self._flat_shading != flat_shading:
            self._flat_shading = flat_shading
            self.redraw()

    def closest_vertex_in_triangle(self, tri_id, point):
        face_vertex_id = np.linalg.norm((self.current_vertices[self.faces[tri_id]] - point), axis=-1).argmin()
        return self.faces[tri_id][face_vertex_id]

    def get_bc_coords_from_points(self, tri_id, points):
        return points_to_barycentric(self.current_vertices[self.faces[[tri_id]]], points)[0]

    @lru_cache(2048)
    def compute_vertex_and_face_normals(self, frame_id, normalize=False):
        """
        Compute face and vertex normals for the given frame. We use an LRU cache since this is a potentially
        expensive operation. This function exists because computing the normals on all frames can increase the
        startup time of the viewer considerably.

        :param frame_id: On which frame to compute the normals.
        :param normalize: Whether or not to normalize the normals. Not doing it is faster and the shaders typically
          enforce unit length of normals anyway.
        :return: The vertex and face normals as a np arrays of shape (V, 3) and (F, 3) respectively.
        """
        vs = self.vertices[frame_id:frame_id + 1] if self.vertices.shape[0] > 1 else self.vertices
        vn, fn = compute_vertex_and_face_normals(vs, self.faces, self.vertex_faces, normalize)
        return vn.squeeze(0), fn.squeeze(0)

    @property
    def bounds(self):
        return self.get_bounds(self.vertices)

    @property
    def current_bounds(self):
        return self.get_bounds(self.current_vertices)

    def is_transparent(self):
        return self.color[3] < 1.0

    def on_frame_update(self):
        """Called whenever a new frame must be displayed."""
        super().on_frame_update()
        self.redraw()

    def _upload_buffers(self):
        """Upload the current frame data to the GPU for rendering."""
        if not self.is_renderable or not self._need_upload:
            return

        self._need_upload = False

        # Write positions.
        self.vbo_vertices.write(self.current_vertices.astype('f4').tobytes())

        # Write normals.
        if not self.flat_shading:
            vertex_normals = self.vertex_normals_at(self.current_frame_id)
            self.vbo_normals.write(vertex_normals.astype('f4').tobytes())

        if self.face_colors is None:
            # Write vertex colors.
            self.vbo_colors.write(self.current_vertex_colors.astype('f4').tobytes())
        else:
            # Write face colors
            self.ssbo_face_colors.write(self.current_face_colors.astype('f4').tobytes())

        # Write uvs.
        if self.has_texture:
            self.vbo_uvs.write(self.uv_coords.astype('f4').tobytes())

    def redraw(self, **kwargs):
        self._need_upload = True

    # noinspection PyAttributeOutsideInit
    @Node.once
    def make_renderable(self, ctx: moderngl.Context):
        """Prepares this object for rendering. This function must be called before `render` is used."""
        self.smooth_prog = get_smooth_lit_with_edges_program()
        self.flat_prog = get_flat_lit_with_edges_program()
        self.smooth_face_prog = get_smooth_lit_with_edges_face_color_program()
        self.flat_face_prog = get_flat_lit_with_edges_face_color_program()

        vertices = self.current_vertices
        vertex_normals = self.vertex_normals_at(self.current_frame_id)
        vertex_colors = self.current_vertex_colors

        self.vbo_vertices = ctx.buffer(vertices.astype('f4').tobytes())
        self.vbo_normals = ctx.buffer(vertex_normals.astype('f4').tobytes())
        self.vbo_colors = ctx.buffer(vertex_colors.astype('f4').tobytes())
        self.vbo_indices = ctx.buffer(self.faces.tobytes())

        self.vao = VAO()
        self.vao.buffer(self.vbo_vertices, '3f4', 'in_position')
        self.vao.buffer(self.vbo_normals, '3f4', 'in_normal')
        self.vao.buffer(self.vbo_colors, '4f4', 'in_color')
        self.vao.index_buffer(self.vbo_indices)

        self.ssbo_face_colors = ctx.buffer(reserve=self.faces.shape[0] * 16)
        if self.face_colors is not None:
            self.ssbo_face_colors.write(self.current_face_colors.astype('f4').tobytes())

        if self.has_texture:
            img = self.texture_image
            if self.use_pickle_texture:
                self.texture = ctx.texture(img.shape[:2], img.shape[2], img.tobytes())
            else:
                self.texture = ctx.texture(img.size, 3, img.tobytes())
            self.texture_prog = get_smooth_lit_texturized_program()
            self.vbo_uvs = ctx.buffer(self.uv_coords.astype('f4').tobytes())
            self.vao.buffer(self.vbo_uvs, '2f4', 'in_uv')

    @hooked
    def release(self):
        if self.is_renderable:
            self.vao.release()
            if self.has_texture:
                self.texture.release()

    def render(self, camera, **kwargs):
        self._upload_buffers()

        if self.has_texture and self.show_texture:
            prog = self.texture_prog
            prog['diffuse_texture'] = 0
            self.texture.use(0)
        else:
            if self.face_colors is None:
                if self.flat_shading:
                    prog = self.flat_prog
                else:
                    prog = self.smooth_prog
            else:
                if self.flat_shading:
                    prog = self.flat_face_prog
                else:
                    prog = self.smooth_face_prog
                self.ssbo_face_colors.bind_to_storage_buffer(0)
            prog['norm_coloring'].value = self.norm_coloring

        prog['use_uniform_color'] = self._use_uniform_color
        prog['uniform_color'] = self.material.color
        prog['draw_edges'].value = 1.0 if self.draw_edges else 0.0
        prog['win_size'].value = kwargs['window_size']

        self.set_camera_matrices(prog, camera, **kwargs)
        set_lights_in_program(prog, kwargs['lights'], kwargs['shadows_enabled'])
        set_material_properties(prog, self.material)
        self.receive_shadow(prog, **kwargs)
        self.vao.render(prog, moderngl.TRIANGLES)

    def render_positions(self, prog):
        if self.is_renderable:
            self._upload_buffers()
            self.vao.render(prog, moderngl.TRIANGLES)

    def _show_normals(self):
        """Create and add normals at runtime"""
        vn = self.vertex_normals

        bounds = self.bounds
        diag = np.linalg.norm(bounds[:, 0] - bounds[:, 1])

        length = 0.005 * max(diag, 1) / self.scale
        vn = vn / np.linalg.norm(vn, axis=-1, keepdims=True) * length

        # Must import here because if we do it at the top we create a circular dependency.
        from aitviewer.renderables.arrows import Arrows

        positions = self.vertices
        self.normals_r = Arrows(positions, positions + vn,
                                r_base=length / 10, r_head=2 * length / 10, p=0.25, name='Normals')
        self.normals_r.current_frame_id = self.current_frame_id
        self.add(self.normals_r)

    def gui(self, imgui):
        super(Meshes, self).gui(imgui)

        _, self.show_texture = imgui.checkbox('Render Texture##render_texture{}'.format(self.unique_name),
                                              self.show_texture)
        _, self.norm_coloring = imgui.checkbox('Norm Coloring##norm_coloring{}'.format(self.unique_name),
                                               self.norm_coloring)
        _, self.flat_shading = imgui.checkbox('Flat shading [F]##flat_shading{}'.format(self.unique_name),
                                              self.flat_shading)
        _, self.draw_edges = imgui.checkbox('Draw edges [E]##draw_edges{}'.format(self.unique_name),
                                            self.draw_edges)
        _, self.draw_outline = imgui.checkbox('Draw outline##draw_outline{}'.format(self.unique_name),
                                              self.draw_outline)

        if self.normals_r is None:
            if imgui.button('Show Normals ##show_normals{}'.format(self.unique_name)):
                self._show_normals()

    def gui_context_menu(self, imgui):
        _, self.flat_shading = imgui.menu_item("Flat shading", "F", selected=self.flat_shading, enabled=True)
        _, self.draw_edges = imgui.menu_item("Draw edges", "E", selected=self.draw_edges, enabled=True)
        _, self.draw_outline = imgui.menu_item("Draw outline", selected=self.draw_outline)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        super().gui_context_menu(imgui)


    def gui_io(self, imgui):
        if imgui.button('Export OBJ##export_{}'.format(self.unique_name)):
            mesh = trimesh.Trimesh(vertices=self.current_vertices, faces=self.faces, process=False)
            mesh.export('../export/' + self.name + '.obj')

    def key_event(self, key, wnd_keys):
        if key == wnd_keys.F:
            self.flat_shading = not self.flat_shading
        elif key == wnd_keys.E:
            self.draw_edges = not self.draw_edges


class VariableTopologyMeshes(Node):
    """
    A sequence of meshes that do not share the same topology (i.e. different nr of vertices, faces, texture, etc.).
    This simply treats every time instance as a separate mesh, so it's not optimized for performance.
    """

    def __init__(self,
                 vertices,
                 faces,
                 vertex_normals=None,
                 face_normals=None,
                 vertex_colors=None,
                 face_colors=None,
                 uv_coords=None,
                 texture_paths=None,
                 preload=True,
                 **kwargs):
        """
        Initializer.
        :param vertices: A list of length N with np arrays of size (V_n, 3).
        :param faces: A list of length N with np arrays of shape (F_n, 3).
        :param vertex_normals: An optional list of length N with np arrays of shape (V_n, 3).
        :param face_normals: An optional list of length N with np arrays of shape (F_n, 3).
        :param vertex_colors: An optional list of length N with np arrays of shape (V_n, 4) overriding the
          uniform color.
        :param face_colors: An optional list of length N with np arrays of shape (F_n, 4) overriding the uniform
          or vertex colors.
        :param uv_coords: An optional list of length N with np arrays of shape (V_n, 2) if the mesh is to be textured.
        :param texture_paths: An optional list of length N containing paths to the texture as an image file.
        :param preload: Whether or not to pre-load all the meshes. This increases loading time and memory consumption,
          but allows interactive animations.
        """
        assert len(vertices) == len(faces)
        super(VariableTopologyMeshes, self).__init__(n_frames=len(vertices))
        self.preload = preload

        self.vertices = vertices
        self.faces = faces
        self.vertex_normals = vertex_normals
        self.face_normals = face_normals
        self.vertex_colors = vertex_colors
        self.face_colors = face_colors
        self.uv_coords = uv_coords
        self.texture_paths = texture_paths
        self.mesh_kwargs = kwargs

        self._current_mesh = dict()  # maps from frame ID to mesh
        self._all_meshes = []
        if self.preload:
            for f in range(self.n_frames):
                m = self._construct_mesh_at_frame(f)
                self._all_meshes.append(m)
        # Set to true after changing the color and used when preloading
        # is not enabled to override the current mesh color with the new color
        self._override_color = False

        self.show_texture = True
        self.norm_coloring = False
        self.flat_shading = False
        self.draw_edges = False
        self.ctx = None

    def _construct_mesh_at_frame(self, frame_id):
        m = Meshes(self.vertices[frame_id],
                   self.faces[frame_id],
                   self.vertex_normals[frame_id] if self.vertex_normals is not None else None,
                   self.face_normals[frame_id] if self.face_normals is not None else None,
                   self.vertex_colors[frame_id] if self.vertex_colors is not None else None,
                   self.face_colors[frame_id] if self.face_colors is not None else None,
                   self.uv_coords[frame_id] if self.uv_coords is not None else None,
                   self.texture_paths[frame_id] if self.texture_paths is not None else None,
                   **self.mesh_kwargs)
        return m

    @classmethod
    def from_trimeshes(cls, trimeshes, **kwargs):
        """Initialize from a list of trimeshes."""
        vertices = []
        faces = []
        vertex_normals = []
        for m in trimeshes:
            vertices.append(m.vertices)
            faces.append(m.faces)
            vertex_normals.append(m.vertex_normals)
        meshes = cls(vertices, faces, vertex_normals, **kwargs)
        return meshes

    @classmethod
    def from_plys(cls, plys, **kwargs):
        """Initialize from a list paths to .ply files."""
        vertices, faces, vertex_normals, vertex_colors = [], [], [], []
        sc = kwargs.get('vertex_scale', 1.0)
        for i in tqdm.tqdm(range(len(plys))):
            m = trimesh.load(plys[i])
            vertices.append(m.vertices * sc)
            faces.append(m.faces)
            vertex_normals.append(m.vertex_normals)
            vertex_colors.append(m.visual.vertex_colors / 255.0)
        return cls(vertices, faces, vertex_normals, vertex_colors=vertex_colors)

    @classmethod
    def from_directory(cls, path, preload=False, vertex_scale=1.0, high_quality=False, **kwargs):
        """
        Initialize from a directory containing mesh and texture data.

        Mesh files must be in pickle (.pkl) or obj (.obj) format, their name
        must start with 'mesh' and end with a frame number.

        Texture files must be in pickle (.pkl), png (.png) or jpeg (.jpg or .jpeg) format
        and their name must match the respective mesh filename with 'atlas' instead of 'mesh'
        at the start of the filename.

        Example:
        path/mesh_001.obj
        path/mesh_002.obj
        path/atlas_001.png
        path/atlas_002.png

        A mesh pickle file must be a dictionary of numpy array with the following shapes:
        {
            'vertices': (V, 3)
            'normals':  (V, 3)
            'uvs':      (V, 2)
            'faces':    (F, 3)
        }

        A texture pickle file must be a raw RGB image stored as a numpy array of shape (width, height, 3)
        """

        files = os.listdir(path)

        # Supported mesh formats in order of preference (fastest to slowest)
        mesh_supported_formats = [".pkl", ".obj"]

        mesh_format = None
        for format in mesh_supported_formats:
            if any(map(lambda x: x.startswith("mesh") and x.endswith(format), files)):
                mesh_format = format
                break

        if mesh_format is None:
            raise ValueError(
                f'Unable to find mesh with supported extensions ({", ".join(mesh_supported_formats)}) at {path}')

        # Supported texture formats in order of preference (fastest to slowest)
        texture_supported_formats = [".pkl", ".jpg", ".jpeg", ".png"]

        # If high_quality is set to true prioritize PNGs
        if high_quality:
            texture_supported_formats = [".png", ".jpg", ".jpeg", ".pkl"]

        texture_format = None
        for format in texture_supported_formats:
            if any(map(lambda x: x.startswith("atlas") and x.endswith(format), files)):
                texture_format = format
                break

        if texture_format is None:
            raise ValueError(
                f'Unable to find atlas with supported extensions ({", ".join(texture_supported_formats)}) at {path}')

        # Load all objects sorted by the keyframe number specified in the file name
        regex = re.compile(r"(\d*)$")

        def sort_key(x):
            name = os.path.splitext(x)[0]
            return int(regex.search(name).group(0))

        obj_names = filter(lambda x: x.startswith("mesh") and x.endswith(mesh_format), files)
        obj_names = sorted(obj_names, key=sort_key)

        vertices, faces, vertex_normals, uvs = [], [], [], []
        texture_paths = []

        for obj_name in tqdm.tqdm(obj_names):
            if mesh_format == ".pkl":
                mesh = pickle.load(open(os.path.join(path, obj_name), "rb"))
                vertices.append(mesh['vertices'] * vertex_scale)
                faces.append(mesh['faces'])
                vertex_normals.append(mesh['normals'])
                uvs.append(mesh['uvs'])
            else:
                mesh = trimesh.load(os.path.join(path, obj_name), process=False)
                vertices.append(mesh.vertices * vertex_scale)
                faces.append(mesh.faces)
                vertex_normals.append(mesh.vertex_normals)
                uvs.append(np.array(mesh.visual.uv).squeeze())

            texture_paths.append(
                os.path.join(path, obj_name.replace("mesh", "atlas").replace(mesh_format, texture_format)))

        return cls(vertices, faces, vertex_normals, uv_coords=uvs,
                   texture_paths=texture_paths, preload=preload, **kwargs)

    @property
    def current_mesh(self):
        if self.preload:
            return self._all_meshes[self.current_frame_id]
        else:
            m = self._current_mesh.get(self.current_frame_id, None)
            if m is None:
                # Need to construct a new one and clean up the old one.
                for k in self._current_mesh:
                    self._current_mesh[k].release()
                m = self._construct_mesh_at_frame(self.current_frame_id)

                # Set mesh position and scale
                m.update_transform(self.model_matrix)

                # Set mesh material
                m.material = self.material

                # Set draw settings.
                m.flat_shading = self.flat_shading
                m.draw_edges = self.draw_edges

                # Override the mesh color only if it has been changed by the user
                if self._override_color:
                    m.color = self.color

                m.make_renderable(self.ctx)
                self._current_mesh = {self.current_frame_id: m}
            return m

    @property
    def bounds(self):
        # TODO: this currently only returns the current mesh bounds for performance/simplicity.
        return self.current_mesh.bounds

    @property
    def current_bounds(self):
        return self.current_mesh.current_bounds

    def closest_vertex_in_triangle(self, tri_id, point):
        return self.current_mesh.closest_vertex_in_triangle(tri_id, point)

    def get_bc_coords_from_points(self, tri_id, points):
        return self.current_mesh.get_bc_coords_from_points(tri_id, points)

    def is_transparent(self):
        return self.color[3] < 1.0

    # noinspection PyAttributeOutsideInit
    @Node.once
    def make_renderable(self, ctx):
        if self.preload:
            for m in self._all_meshes:
                m.make_renderable(ctx)
        else:
            # A bit hacky so that we can make dynamically loaded meshes renderable when we load them.
            self.ctx = ctx
            self.current_mesh.make_renderable(ctx)

    def render(self, camera, **kwargs):
        self.current_mesh.show_texture = self.show_texture
        self.current_mesh.norm_coloring = self.norm_coloring
        self.current_mesh.flat_shading = self.flat_shading
        self.current_mesh.draw_edges = self.draw_edges
        self.current_mesh.render(camera, **kwargs)

    def render_depth_prepass(self, camera, **kwargs):
        self.current_mesh.render_depth_prepass(camera, **kwargs)

    def render_shadowmap(self, light_mvp, program):
        self.current_mesh.render_shadowmap(light_mvp, program)

    def render_fragmap(self, ctx, camera, prog, uid=None):
        # Since the current mesh is not a child node we cannot capture its selection.
        # Therefore we draw to the fragmap using our own id instead of the mesh id.
        self.current_mesh.render_fragmap(ctx, camera, prog, self.uid)

    def render_outline(self, ctx, camera, prog):
        self.current_mesh.render_outline(ctx, camera, prog)

    def gui_context_menu(self, imgui):
        _, self.flat_shading = imgui.menu_item("Flat shading", "F", selected=self.flat_shading, enabled=True)
        _, self.draw_edges = imgui.menu_item("Draw edges", "E", selected=self.draw_edges, enabled=True)
        _, self.draw_outline = imgui.menu_item("Draw outline", selected=self.draw_outline)

        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        super().gui_context_menu(imgui)

    def gui_affine(self, imgui):
        """ Render GUI for affine transformations"""
        # Position controls
        up, pos = imgui.drag_float3('Position##pos{}'.format(self.unique_name), *self.position, 0.1, format='%.2f')
        if up:
            self.position = pos

        # Rotation controls
        euler_angles = rot2euler_numpy(self.rotation[np.newaxis], degrees=True)[0]
        ur, euler_angles = imgui.drag_float3('Rotation##pos{}'.format(self.unique_name), *euler_angles, 0.1,
                                             format='%.2f')
        if ur:
            self.rotation = euler2rot_numpy(np.array(euler_angles)[np.newaxis], degrees=True)[0]

        # Scale controls
        us, scale = imgui.drag_float('Scale##scale{}'.format(self.unique_name), self.scale, 0.01, min_value=0.001,
                                     max_value=10.0, format='%.3f')
        if us:
            self.scale = scale

        if up or ur or us:
            if self.preload:
                for m in self._all_meshes:
                    m.update_transform(self.model_matrix)
            else:
                self.current_mesh.update_transform((self.model_matrix))

    def gui_material(self, imgui, show_advanced=True):
        # Color Control
        uc, color = imgui.color_edit4("Color##color{}'".format(self.unique_name), *self.material.color, show_alpha=True)
        if uc:
            self.color = color
            # If meshes are already loaded go through all and update the vertex colors
            if self.preload:
                for m in self._all_meshes:
                    m.color = color
            # Otherwise only update the current mesh and enable color override for other meshes
            else:
                self.current_mesh.color = color
                self._override_color = True

        _, self.show_texture = imgui.checkbox('Render Texture', self.show_texture)
        _, self.norm_coloring = imgui.checkbox('Norm Coloring', self.norm_coloring)
        _, self.flat_shading = imgui.checkbox('Flat shading [F]', self.flat_shading)
        _, self.draw_edges = imgui.checkbox('Draw edges [E]', self.draw_edges)
        _, self.draw_outline = imgui.checkbox('Draw outline', self.draw_outline)

        if show_advanced:
            if imgui.tree_node("Advanced material##advanced_material{}'".format(self.unique_name)):
                # Diffuse
                ud, diffuse = imgui.slider_float('Diffuse##diffuse{}'.format(self.unique_name),
                                                 self.current_mesh.material.diffuse,
                                                 0.0, 1.0, '%.2f')

                if ud:
                    self.material.diffuse = diffuse
                    # If meshes are already loaded go through all and update the diffuse value
                    if self.preload:
                        for m in self._all_meshes:
                            m.material.diffuse = diffuse
                    # Otherwise only update the current mesh, other meshes will be updated when loaded
                    else:
                        self.current_mesh.material.diffuse = diffuse

                # Ambient
                ua, ambient = imgui.slider_float('Ambient##ambient{}'.format(self.unique_name),
                                                 self.current_mesh.material.ambient,
                                                 0.0, 1.0, '%.2f')
                if ua:
                    self.material.ambient = ambient
                    # If meshes are already loaded go through all and update the ambient value
                    if self.preload:
                        for m in self._all_meshes:
                            m.material.ambient = ambient
                    # Otherwise only update the current mesh, other meshes will be updated when loaded
                    else:
                        self.current_mesh.material.ambient = ambient

                imgui.tree_pop()

    def key_event(self, key, wnd_keys):
        if key == wnd_keys.F:
            self.flat_shading = not self.flat_shading
        elif key == wnd_keys.E:
            self.draw_edges = not self.draw_edges

    @hooked
    def release(self):
        if self.preload:
            for m in self._all_meshes:
                m.release()
        else:
            m = self._current_mesh.get(self.current_frame_id, None)
            if m is not None:
                m.release()

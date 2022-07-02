"""
Copyright (C) 2022  ETH Zurich, Manuel Kaufmann, Velko Vechev

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
import os

import moderngl
import numpy as np
import trimesh
import tqdm

from functools import lru_cache
from PIL import Image

from aitviewer.scene.node import Node
from aitviewer.shaders import get_smooth_lit_with_edges_program
from aitviewer.shaders import get_flat_lit_with_edges_program
from aitviewer.shaders import get_smooth_lit_texturized_program
from aitviewer.utils.utils import compute_vertex_and_face_normals
from aitviewer.utils import set_lights_in_program
from aitviewer.utils import set_material_properties
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
                 texture_alpha=1.0,
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
        :param texture_alpha: Set transparency for texture.
        """
        if len(vertices.shape) == 2 and vertices.shape[-1] == 3:
            vertices = vertices[np.newaxis]
        assert len(vertices.shape) == 3
        assert len(faces.shape) == 2
        super(Meshes, self).__init__(n_frames=vertices.shape[0], **kwargs)

        self._vertices = vertices
        self.faces = faces.astype(np.int32)

        def _maybe_unsqueeze(x):
            return x[np.newaxis] if x is not None and x.ndim == 2 else x

        self._vertex_normals = _maybe_unsqueeze(vertex_normals)
        self._face_normals = _maybe_unsqueeze(face_normals)
        self.vertex_colors = _maybe_unsqueeze(vertex_colors)
        self.face_colors = _maybe_unsqueeze(face_colors)

        # Texture handling.
        self.has_texture = uv_coords is not None
        self.uv_coords = uv_coords
        self.texture_image = Image.open(path_to_texture).transpose(method=Image.FLIP_TOP_BOTTOM).convert(
            "RGB") if self.has_texture else None
        self.texture_alpha = texture_alpha

        # Misc.
        self.flat_shading = False
        self.show_texture = self.has_texture
        self.norm_coloring = False
        self.normals_r = None


    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        self._vertices = vertices
        self.n_frames = len(vertices)
        # If vertex normals were supplied, they are no longer valid.
        self._vertex_normals = None
        # Must clear all LRU caches where the vertices are used.
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
        return self._vertex_colors

    @vertex_colors.setter
    def vertex_colors(self, vertex_colors):
        # Vertex colors cannot be empty
        if vertex_colors is None:
            self._vertex_colors = np.full((self.n_frames, self.n_vertices, 4), self.color)
        elif isinstance(vertex_colors, tuple) and len(vertex_colors) == 4:
            self._vertex_colors = np.full((self.n_frames, self.n_vertices, 4), vertex_colors)
        else:
            self._vertex_colors = vertex_colors

    @property
    def face_colors(self):
        return self._face_colors

    @face_colors.setter
    def face_colors(self, face_colors):
        self._face_colors = face_colors
        if face_colors is not None:
            self._vertex_colors = np.tile(self.face_colors[:, :, np.newaxis], [1, 1, 3, 1])
        self.redraw()

    @Node.color.setter
    def color(self, color):
        alpha_changed = np.abs((np.array(color) - np.array(self.color)))[-1] > 0
        self.material.color = color
        if self.is_renderable:
            # If alpha changed, don't update all colors
            if alpha_changed:
                self._vertex_colors[..., -1] = color[-1]
            # Otherwise, if no face colors, then override the existing vertex colors
            elif self.face_colors is None:
                self.vertex_colors = color

        self.redraw()

    @property
    def current_vertices(self):
        return self.vertices[self.current_frame_id]

    @property
    def current_vertex_colors(self):
        return self.vertex_colors[self.current_frame_id]

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
        vs = self.vertices[frame_id:frame_id + 1]
        vn, fn = compute_vertex_and_face_normals(vs, self.faces, self.vertex_faces, normalize)
        return vn.squeeze(0), fn.squeeze(0)

    @property
    def bounds(self):
        return self.get_bounds(self.vertices)

    def on_frame_update(self):
        """Called whenever a new frame must be displayed."""
        super().on_frame_update()
        self.redraw()

    def redraw(self):
        """Upload the current frame data to the GPU for rendering."""
        if not self.is_renderable:
            return

        # Each write call takes about 1-2 ms
        vertices = self.current_vertices
        vertex_colors = self.current_vertex_colors

        if not self.flat_shading:
            vertex_normals = self.vertex_normals_at(self.current_frame_id)
            if self.face_colors is not None:
                vn_for_drawing = vertex_normals[self.faces]
            else:
                vn_for_drawing = vertex_normals
            self.vbo_normals.write(vn_for_drawing.astype('f4').tobytes())

        if self.face_colors is not None:
            vertices = vertices[self.faces]

        self.vbo_vertices.write(vertices.astype('f4').tobytes())
        self.vbo_colors.write(vertex_colors.astype('f4').tobytes())

        if self.has_texture:
            self.texture_prog['texture_alpha'].value = self.texture_alpha
            self.vbo_uvs.write(self.uv_coords.astype('f4').tobytes())

    def _clear_buffer(self):
        self.vbo_vertices.clear()
        self.vbo_colors.clear()
        if self.has_texture:
            self.vbo_uvs.clear()

    # noinspection PyAttributeOutsideInit
    @Node.once
    def make_renderable(self, ctx):
        """Prepares this object for rendering. This function must be called before `render` is used."""
        self.smooth_prog = get_smooth_lit_with_edges_program()
        self.flat_prog = get_flat_lit_with_edges_program()

        vertices = self.current_vertices
        vertex_normals = self.vertex_normals_at(self.current_frame_id)
        vertex_colors = self.current_vertex_colors

        # Face colors draw 3 distinct vertices per face.
        if self.face_colors is not None:
            vertices = vertices[self.faces]
            vn_for_drawing = vertex_normals[self.faces]
        else:
            vn_for_drawing = vertex_normals

        self.vbo_vertices = ctx.buffer(vertices.astype('f4').tobytes())
        self.vbo_normals = ctx.buffer(vn_for_drawing.astype('f4').tobytes())
        self.vbo_indices = ctx.buffer(self.faces.tobytes()) if self.face_colors is None else None
        self.vbo_colors = ctx.buffer(vertex_colors.astype('f4').tobytes())

        self.smooth_vao = ctx.vertex_array(self.smooth_prog,
                                           [(self.vbo_vertices, '3f4 /v', 'in_position'),
                                            (self.vbo_normals, '3f4 /v', 'in_normal'),
                                            (self.vbo_colors, '4f4 /v', 'in_color')],
                                           self.vbo_indices)

        self.flat_vao = ctx.vertex_array(self.flat_prog,
                                         [(self.vbo_vertices, '3f4 /v', 'in_position'),
                                          (self.vbo_colors, '4f4 /v', 'in_color')],
                                         self.vbo_indices)

        self.cast_shadow(self.vbo_vertices, self.vbo_indices)

        self.make_pickable(self.vbo_vertices, self.vbo_indices)

        if self.has_texture:
            img = self.texture_image
            self.texture = ctx.texture(img.size, 3, img.tobytes())
            self.texture_prog = get_smooth_lit_texturized_program()
            self.texture_prog['texture_alpha'].value = self.texture_alpha
            self.vbo_uvs = ctx.buffer(self.uv_coords.astype('f4').tobytes())
            self.texture_vao = ctx.vertex_array(self.texture_prog,
                                                [(self.vbo_vertices, '3f4 /v', 'in_position'),
                                                 (self.vbo_normals, '3f4 /v', 'in_normal'),
                                                 (self.vbo_uvs, '2f4 /v', 'in_uv')],
                                                self.vbo_indices)

    def release(self):
        self.smooth_vao.release()
        self.flat_vao.release()
        if self.has_texture:
            self.texture_vao.release()

    def render(self, camera, **kwargs):
        # Check if flat shading changed, in which case we need to update the VBOs.
        flat = kwargs.get('flat_rendering', False)
        if flat != self.flat_shading:
            self.flat_shading = flat
            self.redraw()
        vao = self._prepare_vao(camera, **kwargs)
        vao.render(moderngl.TRIANGLES)

    def _prepare_vao(self, camera, **kwargs):
        """Prepare the shader pipeline and the VAO."""
        if self.has_texture and self.show_texture:
            prog, vao = self.texture_prog, self.texture_vao
            self.texture.use(1)
            prog['diffuse_texture'].value = 1
        else:
            prog, vao = (self.flat_prog, self.flat_vao) if self.flat_shading else (self.smooth_prog, self.smooth_vao)
            prog['norm_coloring'].value = self.norm_coloring

        prog['draw_edges'].value = 1.0 if kwargs['draw_edges'] and self.material._show_edges else 0.0
        prog['win_size'].value = kwargs['window_size']

        self.set_camera_matrices(prog, camera, **kwargs)
        set_lights_in_program(prog, kwargs['lights'])
        set_material_properties(prog, self.material)
        self.receive_shadow(prog, **kwargs)
        return vao

    def _show_normals(self):
        """Create and add normals at runtime"""
        vn = self.vertex_normals
        length = 0.01  # TODO make this either a parameter in the GUI or dependent on the bounding box.
        vn = vn / np.linalg.norm(vn, axis=-1, keepdims=True) * length

        # Must import here because if we do it at the top we create a circular dependency.
        from aitviewer.renderables.arrows import Arrows
        self.normals_r = Arrows(self.vertices, self.vertices + vn,
                                r_base=length / 10, r_head=2 * length / 10, p=0.25, name='Normals')
        self.normals_r.position = self.position
        self.normals_r.current_frame_id = self.current_frame_id
        self.add(self.normals_r, gui_elements=['material'])

    def gui(self, imgui):
        super(Meshes, self).gui(imgui)
        if self.normals_r is None:
            if imgui.button("Show Normals ##s_normals{}".format(self.unique_name)):
                self._show_normals()

            _, self.show_texture = imgui.checkbox('Render Texture', self.show_texture)
            _, self.norm_coloring = imgui.checkbox('Norm Coloring', self.norm_coloring)

            # TODO: Add  export workflow for all nodes
            if imgui.button("Export OBJ"):
                mesh = trimesh.Trimesh(vertices=self.current_vertices, faces=self.faces, process=False)
                mesh.export('../export/' + self.name + '.obj')


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
        :param vertex_colors: An optional list of length N with np arrays of shape (V_n, 4) overriding the uniform color.
        :param face_colors: An optional list of length N with np arrays of shape (F_n, 4) overriding the uniform or vertex colors.
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

        self.show_texture = True
        self.norm_coloring = False
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
        self.material = m.material
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
                m.make_renderable(self.ctx)
                self._current_mesh = {self.current_frame_id: m}
            return m

    @property
    def bounds(self):
        return self.current_mesh.get_bounds(self.current_mesh.vertices)

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
        self.current_mesh.render(camera, **kwargs)

    def render_shadowmap(self, light_mvp, program):
        self.current_mesh.render_shadowmap(light_mvp, program)

    def gui_position(self, imgui):
        # Position controls
        u, pos = imgui.drag_float3('Position##pos{}'.format(self.unique_name), *self.current_mesh.position, 0.1, format='%.2f')
        if u:
            for m in self._all_meshes:
                m.position = pos

    def gui_scale(self, imgui):
        # Scale controls
        u, scale = imgui.drag_float('Scale##scale{}'.format(self.unique_name), self.current_mesh.scale, 0.01, min_value=0.001,
                                    max_value=10.0, format='%.3f')
        if u:
            self.current_mesh.scale = scale

    def gui_material(self, imgui, show_advanced=True):
        # Color Control
        uc, color = imgui.color_edit4("Color##color{}'".format(self.unique_name), *self.current_mesh.material.color, show_alpha=True)
        if uc:
            for m in self._all_meshes:
                m.color = color

        _, self.show_texture = imgui.checkbox('Render Texture', self.show_texture)
        _, self.norm_coloring = imgui.checkbox('Norm Coloring', self.norm_coloring)

        if show_advanced:
            if imgui.tree_node("Advanced material##advanced_material{}'".format(self.unique_name)):
                # Diffuse
                ud, diffuse = imgui.slider_float('Diffuse##diffuse{}'.format(self.unique_name), self.current_mesh.material.diffuse,
                                                 0.0, 1.0, '%.2f')
                if ud:
                    for m in self._all_meshes:
                        m.material.diffuse = diffuse

                # Ambient
                ua, ambient = imgui.slider_float('Ambient##ambient{}'.format(self.unique_name), self.current_mesh.material.ambient,
                                                 0.0, 1.0, '%.2f')
                if ua:
                    for m in self._all_meshes:
                        m.material.ambient = ambient

                imgui.tree_pop()

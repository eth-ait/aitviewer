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
import moderngl
import numpy as np
import trimesh

from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.material import Material
from aitviewer.scene.node import Node
from aitviewer.shaders import get_cylinder_program
from aitviewer.utils import set_lights_in_program
from aitviewer.utils import set_material_properties
from aitviewer.utils import compute_vertex_and_face_normals
from aitviewer.utils.so3 import aa2rot_numpy as aa2rot
from moderngl_window.opengl.vao import VAO


_CYLINDER_SECTORS = 8


def _create_disk(n_disks=1, radius=1.0, sectors=None, plane='xz'):
    """
    Create `n_disks` many disks centered at the origin with the given radius1.
    :param n_disks: How many disks to create.
    :param radius: Radius of the disks.
    :param sectors: How many lines to use to approximate the disk. Increasing this will lead to more vertex data.
    :param plane: In which plane to create the disk.
    :return: Vertices as a np array of shape (N, V, 3), and face data as a np array of shape (F, 3).
    """
    assert plane in ['xz', 'xy', 'yz']
    sectors = sectors or _CYLINDER_SECTORS
    angle = 2 * np.pi / sectors

    c1 = 'xyz'.index(plane[0])
    c2 = 'xyz'.index(plane[1])
    c3 = 'xyz'.index('xyz'.replace(plane[0], '').replace(plane[1], ''))

    # Vertex Data.
    vertices = np.zeros((n_disks, sectors + 1, 3))
    x = radius * np.cos(np.arange(sectors) * angle)
    y = radius * np.sin(np.arange(sectors) * angle)
    vertices[:, 1:, c1] = x
    vertices[:, 1:, c2] = y

    # Faces.
    faces = np.zeros((sectors, 3), dtype=np.int32)
    idxs = np.array(range(1, sectors+1), dtype=np.int32)
    faces[:, 2] = idxs
    faces[:-1, 1] = idxs[1:]
    faces[-1, 1] = 1

    return {'vertices': vertices, 'faces': faces}


def _create_cylinder_from_to(v1, v2, radius1=1.0, radius2=1.0, sectors=None):
    """
    Create cylinders from points v1 to v2.
    :param v1: A np array of shape (N, 3).
    :param v2: A np array of shape (N, 3).
    :param radius1: The radius at the bottom of the cylinder.
    :param radius2: The radius at the top of the cylinder.
    :param sectors: how many lines to use to approximate the bottom disk.
    :return: Vertices and normals as a np array of shape (N, V, 3) and face data in shape (F, 3), i.e. only one
      face array is created for all cylinders.
    """
    sectors = sectors or _CYLINDER_SECTORS
    n_cylinders = v1.shape[0]

    # Create bottom lid.
    bottom = _create_disk(n_disks=n_cylinders, radius=radius1, sectors=sectors)

    # We must also change the winding of the bottom triangles because we have backface culling enabled and
    # otherwise we wouldn't see the bottom lid even if the normals are correct.
    fs_bottom = bottom['faces']
    fs_bottom[:, 1], fs_bottom[:, 2] = fs_bottom[:, 2], fs_bottom[:, 1].copy()

    # Create top lid.
    top = _create_disk(n_disks=n_cylinders, radius=radius2, sectors=sectors)
    p2 = np.zeros((n_cylinders, 3))
    p2[:, 1] = np.linalg.norm(v2 - v1, axis=-1)
    top['vertices'] = top['vertices'] + p2[:, np.newaxis]

    # Shift indices of top faces by how many vertices the bottom lid has.
    n_vertices = bottom['vertices'].shape[1]
    fs_top = top['faces'] + n_vertices

    # Create the faces that make up the coat between bottom and top lid.
    idxs_bot = np.array(range(1, sectors+1), dtype=np.int32)
    idxs_top = idxs_bot + n_vertices
    fs_coat1 = np.zeros((sectors, 3), dtype=np.int32)
    fs_coat1[:, 0] = idxs_top
    fs_coat1[:-1, 1] = idxs_top[1:]
    fs_coat1[-1, 1] = idxs_top[0]
    fs_coat1[:, 2] = idxs_bot

    fs_coat2 = np.zeros((sectors, 3), dtype=np.int32)
    fs_coat2[:, 0] = fs_coat1[:, 1]
    fs_coat2[:-1, 1] = idxs_bot[1:]
    fs_coat2[-1, 1] = idxs_bot[0]
    fs_coat2[:, 2] = idxs_bot

    # Concatenate everything to create a single mesh.
    vs = np.concatenate([bottom['vertices'], top['vertices']], axis=1)
    fs = np.concatenate([fs_bottom, fs_top, fs_coat1, fs_coat2], axis=0)

    # Compute smooth normals.
    vertex_faces = trimesh.Trimesh(vs[0], fs, process=False).vertex_faces
    ns, _ = compute_vertex_and_face_normals(vs[0:1], fs, vertex_faces, normalize=True)
    ns = np.repeat(ns, n_cylinders, axis=0)

    # Rotate cylinders to align the the given data.
    vs, ns = _rotate_cylinder_to(v2-v1, vs, ns)

    # Translate cylinders to the given positions
    vs += v1[:, np.newaxis]

    return {'vertices': vs, 'normals': ns, 'faces': fs}


def _create_cone_from_to(v1, v2, radius=1.0, sectors=None):
    """
    Create a cone from points v1 to v2.
    :param v1: A np array of shape (N, 3).
    :param v2: A np array of shape (N, 3).
    :param radius: The radius for the disk of the cone.
    :param sectors: how many lines to use to approximate the bottom disk.
    :return: Vertices and normals as a np array of shape (N, V, 3) and face data in shape (F, 3), i.e. only one
      face array is created for all cones.
    """
    sectors = sectors or _CYLINDER_SECTORS
    n_cylinders = v1.shape[0]

    # Create bottom lid.
    bottom = _create_disk(n_cylinders, radius, sectors)

    # We must also change the winding of the bottom triangles because we have backface culling enabled and
    # otherwise we wouldn't see the bottom lid even if the normals are correct.
    fs_bottom = bottom['faces']
    fs_bottom[:, 1], fs_bottom[:, 2] = fs_bottom[:, 2], fs_bottom[:, 1].copy()

    # Add the top position as a new vertex.
    p2 = np.zeros((n_cylinders, 3))
    p2[:, 1] = np.linalg.norm(v2 - v1, axis=-1)
    vs = np.concatenate([bottom['vertices'], p2[:, np.newaxis]], axis=1)
    n_vertices = vs.shape[1]
    idx_top = n_vertices-1

    # Create the faces from the bottom lid to the top point.
    idxs_bot = np.array(range(1, sectors + 1), dtype=np.int32)
    fs_coat = np.zeros((sectors, 3), dtype=np.int32)
    fs_coat[:, 0] = idx_top
    fs_coat[:-1, 1] = idxs_bot[1:]
    fs_coat[-1, 1] = idxs_bot[0]
    fs_coat[:, 2] = idxs_bot

    # Concatenate everything to create a single mesh.
    fs = np.concatenate([fs_bottom, fs_coat], axis=0)

    # Compute smooth normals.
    vertex_faces = trimesh.Trimesh(vs[0], fs, process=False).vertex_faces
    ns, _ = compute_vertex_and_face_normals(vs[0:1], fs, vertex_faces, normalize=True)
    ns = np.repeat(ns, n_cylinders, axis=0)

    # Rotate cones to align the the given data.
    vs, ns = _rotate_cylinder_to(v2 - v1, vs, ns)

    # Translate cones to the given positions
    vs += v1[:, np.newaxis]

    return {'vertices': vs, 'normals': ns, 'faces': fs}


def _rotate_cylinder_to(target, vs, ns):
    """
    Rotate vertices and normals such that the main axis of the cylinder aligns with `target`.
    :param target: A np array of shape (N, 3) specifying the target direction for each cylinder given in `vs`.
    :param vs: A np array of shape (N, V, 3), i.e. the vertex data for each cylinder.
    :param ns: A np array of shape (N, V, 3), i.e. the normal data for each cylinder.
    :return: The rotated vertices and normals.
    """
    n_cylinders = vs.shape[0]
    s = np.array([[0.0, 1.0, 0.0]]).repeat(n_cylinders, axis=0)
    v = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-16)
    a = np.cross(s, v, axis=1)
    dot = np.sum(s * v, axis=1)
    theta = np.arccos(dot)
    is_col = np.linalg.norm(a, axis=1) < 10e-6
    a[is_col] = np.array([1.0, 0.0, 0.0])
    theta[is_col] = 0.0
    theta[np.logical_and(dot < 0.0, is_col)] = np.pi
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    rot = aa2rot(a * theta[..., np.newaxis])
    vs = np.matmul(rot[:, np.newaxis], vs[..., np.newaxis]).squeeze(-1)
    ns = np.matmul(rot[:, np.newaxis], ns[..., np.newaxis]).squeeze(-1)
    return vs, ns


class Lines(Node):
    """Render lines as cylinders or cones. Can render approx. 600k lines at 40 fps."""

    def __init__(self,
                 lines,
                 r_base=0.01,
                 r_tip=None,
                 color=(0.0, 0.0, 1.0, 1.0),
                 mode='line_strip',
                 **kwargs):
        """
        Initializer.
        :param lines: Set of 3D coordinates as a np array of shape (F, L, 3).
        :param r_base: Thickness of the line.
        :param r_tip: If set, the thickness of the line will taper from r_base to r_tip. If set to 0.0 it will create
          a proper cone.
        :param color: Color of the line (4-tuple).
        :param mode: 'lines' or 'line_strip' -> ModernGL drawing mode - LINE_STRIP oder LINES
        """
        assert len(color) == 4
        assert len(lines.shape) >= 2
        assert mode == "lines" or mode == "line_strip"
        self.mode = mode
        self._lines = None
        self.lines = lines

        super(Lines, self).__init__(n_frames=self.lines.shape[0], **kwargs)

        self.r_base = r_base
        self.r_tip = r_tip

        vs, fs, ns = self.get_mesh()
        material = kwargs.get('material', Material(color=color, ambient=0.2))
        self.mesh = Meshes(vs, fs, ns, color=color, material=material)
        self.mesh.position = self.position
        self.add(self.mesh, has_gui=True, show_in_hierarchy=False)

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, value):
        self._lines = value if len(value.shape) == 3 else value[np.newaxis]

    def gui(self, imgui):
        self.mesh.gui(imgui)

    def redraw(self, **kwargs):
        vs, fs, ns = self.get_mesh()
        self.mesh.vertices = vs
        self.mesh.faces = fs
        super().redraw()

    @property
    def color(self):
        return self.mesh.color

    @color.setter
    def color(self, color):
        self.mesh.color = color

    def get_mesh(self):
        # Extract pairs of lines such that line i goes from v0s[i] to v1s[i].
        if self.mode == 'lines':
            assert self.lines.shape[1] % 2 == 0
            v0s = self.lines[:, ::2]
            v1s = self.lines[:, 1::2]
        else:
            v0s = self.lines[:, :-1]
            v1s = self.lines[:, 1:]

        # Data is in the form of (F, N_LINES, V, 3), convert it to (F*N_LINES, 3)
        n_lines = v0s.shape[1]
        v0s = np.reshape(v0s, (-1, 3))
        v1s = np.reshape(v1s, (-1, 3))
        self.r_tip = self.r_base if self.r_tip is None else self.r_tip
        # If r_tip is below a certain threshold, we create a proper cone, i.e. with just a single vertex at the top.
        if self.r_tip < 10e-6:
            data = _create_cone_from_to(v0s, v1s, radius=self.r_base)
        else:
            data = _create_cylinder_from_to(v0s, v1s, radius1=self.r_base, radius2=self.r_tip)

        # Convert to (F, N_LINES*V, 3)
        n_vertices = data['vertices'].shape[1]
        vs = np.reshape(data['vertices'], [self.n_frames, -1, 3])
        ns = np.reshape(data['normals'], [self.n_frames, -1, 3])
        fs = []
        for i in range(n_lines):
            fs.append(data['faces'] + i * n_vertices)
        fs = np.concatenate(fs)

        return vs, fs, ns


class LinesWithGeometryShader(Node):
    """
    Draw a line between n points using a cylinder geometry shader.
    Two radii can be set resulting in a cylinder, tapered cylinder, or an arrow (r_tip = 0.0)
    """

    def __init__(self,
                 lines,
                 r_base=0.01,
                 r_tip=None,
                 color=(0.0, 0.0, 1.0, 1.0),
                 mode='line_strip',
                 **kwargs):
        """
        Initializer.
        :param lines: Set of 3D coordinates as a np array of shape (N, L, 3).
        :param r_base: Thickness of the line.
        :param r_tip: If set, the thickness of the line will taper from r_base to r_tip
        :param color: Color of the line (4-tuple).
        :param mode: 'lines' or 'line_strip' -> ModernGL drawing mode - LINE_STRIP oder LINES
        """
        assert len(color) == 4
        assert len(lines.shape) >= 2
        assert mode == "lines" or mode == "line_strip"
        lines = lines if len(lines.shape) == 3 else lines[np.newaxis]

        super(LinesWithGeometryShader, self).__init__(n_frames=len(lines), color=color, **kwargs)

        self.r_base = r_base
        self.r_tip = r_tip if r_tip is not None else r_base
        self.lines = lines
        self.colors = np.full((self.n_lines, 4), self.color)
        self.mode = moderngl.LINE_STRIP if mode == 'line_strip' else moderngl.LINES
        self.vao = VAO("cylinder", mode=self.mode)

    @classmethod
    def from_start_end_points(cls, v0, v1, **kwargs):
        """Create lines that connect from v0[i, j] to v1[i, j] for frame i and line j."""
        vs = np.zeros((v0.shape[0], v0.shape[1] * 2, 3))
        vs[:, ::2] = v0
        vs[:, 1::2] = v1
        return cls(vs, mode='lines', **kwargs)

    def on_frame_update(self):
        if self.is_renderable:
            self.vbo_vertices.write(self.lines_current.astype('f4').tobytes())

    def update_data(self, lines):
        self.lines = lines
        self.on_frame_update()

    @property
    def n_lines(self):
        return self.lines.shape[1]

    @property
    def lines_current(self):
        return self.lines[self.current_frame_id]

    @Node.color.setter
    def color(self, color):
        self.material.color = color
        self.colors = np.full((self.n_lines, 4), self.color)
        if self.is_renderable:
            self.vbo_colors.write(self.colors.astype('f4').tobytes())

    # noinspection PyAttributeOutsideInit
    @Node.once
    def make_renderable(self, ctx):
        self.prog = get_cylinder_program()
        self.prog['r1'] = self.r_base
        self.prog['r2'] = self.r_tip

        self.vbo_vertices = ctx.buffer(self.lines_current.astype('f4').tobytes())
        self.vbo_colors = ctx.buffer(self.colors.astype('f4').tobytes())
        self.vao.buffer(self.vbo_vertices, '3f', ['in_position'])
        self.vao.buffer(self.vbo_colors, '4f', ['in_color'])

    def render(self, camera, **kwargs):
        self.set_camera_matrices(self.prog, camera, **kwargs)
        set_lights_in_program(self.prog, kwargs['lights'])
        set_material_properties(self.prog, self.material)
        self.vao.render(self.prog)

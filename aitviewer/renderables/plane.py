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

from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.node import Node
from aitviewer.shaders import get_smooth_lit_with_edges_program
from aitviewer.utils import set_lights_in_program
from aitviewer.utils import set_material_properties


class Plane(Node):
    """
    Draw a plane.
    """
    def __init__(self,
                 center,
                 v1,
                 v2,
                 size=10.0,
                 color=(0.5, 0.5, 0.5, 1.0),
                 **kwargs):
        """
        Initializer.
        :param center: Center of the plane.
        :param v1: 3D vector lying in the plane.
        :param v2: 3D vector lying in the plane and not co-linear to `v2`.
        :param size: Size of the plane.
        :param color: Color of the plane.
        """
        super(Plane, self).__init__(color=color, **kwargs)
        if np.dot(v1, v2) > 0.00001:
            raise ValueError('v1 and v2 are not orthogonal.')

        self.center = center
        self.v1 = v1 / np.linalg.norm(v1)
        self.v2 = v2 / np.linalg.norm(v2)
        self.size = size

        self.vertices, self.normals = self._get_renderable_data()
        self.colors = np.full((self.vertices.shape[0], 4), color)

    def _get_renderable_data(self):
        p0 = self.center + self.v1 * self.size - self.v2 * self.size
        p1 = self.center + self.v1 * self.size + self.v2 * self.size
        p2 = self.center - self.v1 * self.size + self.v2 * self.size
        p3 = self.center - self.v1 * self.size - self.v2 * self.size
        normal = np.cross(self.v2, self.v1)
        normal = np.tile(normal[np.newaxis], (4, 1))
        return np.row_stack([p1, p0, p2, p3]), normal

    @Node.color.setter
    def color(self, color):
        self.material.color = color
        self.colors = np.full((self.vertices.shape[0], 4), color)
        if self.is_renderable:
            self.vbo_colors.write(self.colors.astype('f4').tobytes())

    # noinspection PyAttributeOutsideInit
    @Node.once
    def make_renderable(self, ctx):
        self.prog = get_smooth_lit_with_edges_program()
        self.vbo_vertices = ctx.buffer(self.vertices.astype('f4').tobytes())
        self.vbo_normals = ctx.buffer(self.normals.astype('f4').tobytes())
        self.vbo_colors = ctx.buffer(self.colors.astype('f4').tobytes())

        self.vao = ctx.vertex_array(self.prog,
                                    [(self.vbo_vertices, '3f4 /v', 'in_position'),
                                     (self.vbo_normals, '3f4 /v', 'in_normal'),
                                     (self.vbo_colors, '4f4 /v', 'in_color')])

    def render(self, camera, **kwargs):
        self.set_camera_matrices(self.prog, camera, **kwargs)
        self.receive_shadow(self.prog, **kwargs)

        self.prog['draw_edges'].value = 1.0 if kwargs['draw_edges'] else 0.0
        self.prog['norm_coloring'].value = False
        self.prog['win_size'].value = kwargs['window_size']

        set_lights_in_program(self.prog, kwargs['lights'])
        set_material_properties(self.prog, self.material)
        self.vao.render(moderngl.TRIANGLE_STRIP)


class Chessboard(Node):
    """A plane that is textured like a chessboard."""
    def __init__(self,
                 side_length,
                 n_tiles,
                 color1=(0.0, 0.0, 0.0, 1.0),
                 color2=(1.0, 1.0, 1.0, 1.0),
                 plane='xz',
                 height=0.0,
                 tiling=True,
                 **kwargs):
        """
        Initializer.
        :param side_length: Length of one side of the plane.
        :param n_tiles: Number of tiles for the chessboard pattern.
        :param color1: First color of the chessboard pattern.
        :param color2: Second color of the chessboard pattern.
        :param plane: In which plane the chessboard lies. Allowed are 'xz', 'xy', 'yz'.
        :param height: The height of the plane.
        :param kwargs: Remaining kwargs.
        """
        assert plane in ['xz', 'xy', 'yz']
        super(Chessboard, self).__init__(**kwargs)
        self.side_length = side_length
        self.n_tiles = n_tiles
        self.c1 = np.array(color1)
        self.c2 = np.array(color2)
        self.plane = plane
        self.height = height
        self.tiling = tiling

        vs, fs, fc, c1_idxs, c2_idxs = self._construct_board()
        self.fcs_tiled = fc
        self.c1_idxs = c1_idxs
        self.c2_idxs = c2_idxs

        self.mesh = Meshes(vs, fs, face_colors=fc)
        self.mesh.position = self.position
        self.add(self.mesh, has_gui=False, show_in_hierarchy=False)

    # noinspection PyAttributeOutsideInit
    def _construct_board(self):
        """Construct the chessboard mesh."""
        vertices = []
        faces = []
        face_colors = []
        c1_idxs = []  # Store indices into face_colors containing color 1.
        c2_idxs = []  # Store indices into face_colors containing color 2.
        tl = self.side_length / self.n_tiles
        dim1 = 'xyz'.index(self.plane[0])
        dim2 = 'xyz'.index(self.plane[1])
        up = 'xyz'.index('xyz'.replace(self.plane[0], '').replace(self.plane[1], ''))

        for r in range(self.n_tiles):
            for c in range(self.n_tiles):
                v0 = np.zeros([3])
                v0[dim1] = r * tl
                v0[dim2] = c * tl

                v1 = np.zeros([3])
                v1[dim1] = (r + 1) * tl
                v1[dim2] = c * tl

                v2 = np.zeros([3])
                v2[dim1] = (r + 1) * tl
                v2[dim2] = (c + 1) * tl

                v3 = np.zeros([3])
                v3[dim1] = r * tl
                v3[dim2] = (c + 1) * tl

                vertices.extend([v0, v1, v2, v3])

                # Need counter-clock-wise ordering
                faces.append([len(vertices) - 4, len(vertices) - 1, len(vertices) - 3])
                faces.append([len(vertices) - 3, len(vertices) - 1, len(vertices) - 2])

                if r % 2 == 0 and c % 2 == 0:
                    c = self.c1
                    fc_idxs = c1_idxs
                elif r % 2 == 0 and c % 2 != 0:
                    c = self.c2
                    fc_idxs = c2_idxs
                elif r % 2 != 0 and c % 2 != 0:
                    c = self.c1
                    fc_idxs = c1_idxs
                else:
                    c = self.c2
                    fc_idxs = c2_idxs
                face_colors.append(c)
                face_colors.append(c)
                fc_idxs.extend([len(face_colors) - 2, len(face_colors) - 1])

        vs = np.stack(vertices)
        vs = vs - np.mean(vertices, axis=0, keepdims=True)
        vs[:, up] = self.height
        fs = np.stack(faces)
        cs = np.stack(face_colors)

        return vs, fs, cs, c1_idxs, c2_idxs

    def _update_colors(self):
        self.fcs_tiled[self.c1_idxs] = self.c1
        self.fcs_tiled[self.c2_idxs] = self.c2 if self.tiling else self.c1
        self.mesh.face_colors = self.fcs_tiled

    def gui(self, imgui):
        self.gui_position(imgui)

        u, c1 = imgui.color_edit4("Color 1##color{}'".format(self.unique_name), *self.c1, show_alpha=True)
        if u:
            self.c1 = c1
            self._update_colors()

        u, c2 = imgui.color_edit4("Color 2##color{}'".format(self.unique_name), *self.c2, show_alpha=True)
        if u:
            self.c2 = c2
            self._update_colors()

        u, self.tiling = imgui.checkbox('Toggle Tiling', self.tiling)
        if u:
            self._update_colors()


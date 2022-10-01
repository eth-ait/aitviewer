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
import numpy as np

from aitviewer.scene.node import Node
from aitviewer.renderables.spheres import Spheres
from aitviewer.renderables.arrows import Arrows


class RigidBodies(Node):
    """
    A sequence of N 3D positions and orientations in space.
    """
    def __init__(self,
                 rb_pos,
                 rb_ori,
                 radius=0.02,
                 length=0.2,
                 radius_cylinder=None,
                 color=(0.0, 1.0, 0.5, 1.0),
                 icon="\u0086",
                 **kwargs):
        """
        Initializer.
        :param rb_pos: A np array of shape (F, N, 3) containing N rigid-body centers over F time steps.
        :param rb_ori: A np array of shape (F, N, 3, 3) containing N rigid-body orientations over F time steps.
        :param radius: Radius of the sphere at the origin of the rigid body.
        :param length: Length of arrows representing the orientation of the rigid body.
        :param radius_cylinder: Radius of the cylinder representing the orientation, default is length / 50
        :param color: Color of the rigid body centers (4-tuple).
        """
        self.rb_pos = rb_pos[np.newaxis] if rb_pos.ndim == 2 else rb_pos
        self.rb_ori = rb_ori[np.newaxis] if rb_ori.ndim == 3 else rb_ori
        super(RigidBodies, self).__init__(n_frames=self.rb_pos.shape[0], color=color, icon=icon, **kwargs)

        self.radius = radius
        self.length = length

        self.spheres = Spheres(rb_pos, radius=radius, color=color, is_selectable=False)
        self._add_node(self.spheres, show_in_hierarchy=False)

        self.coords = []
        r_base = radius_cylinder or length / 50
        r_head = r_base * 2
        c = [0.0, 0.0, 0.0, 1.0]
        for i in range(3):
            line = self.rb_ori[..., :, i]
            line = line / np.linalg.norm(line, axis=-1, keepdims=True) * length
            color = c.copy()
            color[i] = 1.0
            axs = Arrows(self.rb_pos, self.rb_pos + line, r_base=r_base, r_head=r_head, color=tuple(color),
                         is_selectable=False)
            self._add_node(axs, show_in_hierarchy=False)
            self.coords.append(axs)

    @Node.color.setter
    def color(self, color):
        self.material.color = color
        self.spheres.color = color

    @property
    def current_rb_pos(self):
        idx = self.current_frame_id if self.rb_pos.shape[0] > 1 else 0
        return self.rb_pos[idx]

    @current_rb_pos.setter
    def current_rb_pos(self, pos):
        idx = self.current_frame_id if self.rb_pos.shape[0] > 1 else 0
        self.rb_pos[idx] = pos

    @property
    def current_rb_ori(self):
        idx = self.current_frame_id if self.rb_ori.shape[0] > 1 else 0
        return self.rb_ori[idx]

    @current_rb_ori.setter
    def current_rb_ori(self, ori):
        idx = self.current_frame_id if self.rb_ori.shape[0] > 1 else 0
        self.rb_ori[idx] = ori

    def redraw(self, **kwargs):
        if kwargs.get('current_frame_only', False):
            self.spheres.current_sphere_positions = self.current_rb_pos

            for i in range(3):
                line = self.rb_ori[..., :, i][self.current_frame_id]
                line = line / np.linalg.norm(line, axis=-1, keepdims=True) * self.length
                axs = self.coords[i]
                axs.current_origins = self.current_rb_pos
                axs.current_tips = self.current_rb_pos + line
        else:
            self.spheres.sphere_positions = self.rb_pos

            for i in range(3):
                line = self.rb_ori[..., :, i]
                line = line / np.linalg.norm(line, axis=-1, keepdims=True) * self.length
                axs = self.coords[i]
                axs.origins = self.rb_pos
                axs.tips = self.rb_pos + line

        super().redraw(**kwargs)

    def get_index_from_node_and_triangle(self, node, tri_id):
        idx = self.spheres.get_index_from_node_and_triangle(node, tri_id)
        if idx is not None:
            return idx

        for a in self.coords:
            idx = a.get_index_from_node_and_triangle(node, tri_id)
            if idx is not None:
                return idx

    def color_one(self, index, color):
        col = np.full((1, self.spheres.n_vertices * self.spheres.n_spheres, 4), self.color)
        col[:, self.spheres.n_vertices * index: self.spheres.n_vertices * (index + 1)] = np.array(color)
        self.spheres.vertex_colors = col

    def gui(self, imgui):
        super(RigidBodies, self).gui(imgui)
        # Scale controls
        u, scale = imgui.drag_float('Sphere Radius##radius{}'.format(self.unique_name), self.spheres.radius,
                                    0.01, min_value=0.001, max_value=10.0, format='%.3f')
        if u:
            self.spheres.radius = scale
            self.redraw()

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
import os

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
        super(RigidBodies, self).__init__(n_frames=self.rb_pos.shape[0], color=color, **kwargs)

        self.radius = radius
        self.length = length

        self.spheres = Spheres(rb_pos, radius=radius, color=color, position=self.position)
        self._add_node(self.spheres, has_gui=False, show_in_hierarchy=False)

        self.coords = []
        r_base = radius_cylinder or length / 50
        r_head = r_base * 2
        c = [0.0, 0.0, 0.0, 1.0]
        for i in range(3):
            line = self.rb_ori[..., :, i]
            line = line / np.linalg.norm(line, axis=-1, keepdims=True) * length
            color = c.copy()
            color[i] = 1.0
            axs = Arrows(self.rb_pos, self.rb_pos + line, r_base=r_base, r_head=r_head, color=tuple(color))
            axs.position = self.position
            self._add_node(axs, has_gui=False, show_in_hierarchy=False)
            self.coords.append(axs)

    @Node.color.setter
    def color(self, color):
        self.material.color = color
        self.spheres.color = color

    def redraw(self, **kwargs):
        self.spheres.sphere_positions = self.rb_pos
        self.spheres.redraw(**kwargs)

        for i in range(3):
            line = self.rb_ori[..., :, i]
            line = line / np.linalg.norm(line, axis=-1, keepdims=True) * self.length
            axs = self.coords[i]
            axs.position = self.position
            axs.origins = self.rb_pos
            axs.tips = self.rb_pos + line
            axs.redraw(**kwargs)

    def gui(self, imgui):
        super(RigidBodies, self).gui(imgui)
        # Scale controls
        u, scale = imgui.drag_float('Sphere Radius##radius{}'.format(self.unique_name), self.spheres.radius,
                                    0.01, min_value=0.001, max_value=10.0, format='%.3f')
        if u:
            self.spheres.radius = scale
            self.redraw()

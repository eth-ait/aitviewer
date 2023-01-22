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
import numpy as np

from aitviewer.renderables.arrows import Arrows
from aitviewer.renderables.spheres import Spheres
from aitviewer.scene.node import Node
from aitviewer.utils.utils import (
    compute_union_of_bounds,
    compute_union_of_current_bounds,
)


class RigidBodies(Node):
    """
    A sequence of N 3D positions and orientations in space.
    """

    def __init__(
        self,
        rb_pos,
        rb_ori,
        radius=0.02,
        length=0.2,
        radius_cylinder=None,
        color=(0.0, 1.0, 0.5, 1.0),
        icon="\u0086",
        **kwargs,
    ):
        """
        Initializer.
        :param rb_pos: A np array of shape (F, N, 3) containing N rigid-body centers over F time steps.
        :param rb_ori: A np array of shape (F, N, 3, 3) containing N rigid-body orientations over F time steps.
        :param radius: Radius of the sphere at the origin of the rigid body.
        :param length: Length of arrows representing the orientation of the rigid body.
        :param radius_cylinder: Radius of the cylinder representing the orientation, default is length / 50
        :param color: Color of the rigid body centers (4-tuple).
        """
        self._rb_pos = rb_pos[np.newaxis] if rb_pos.ndim == 2 else rb_pos
        self._rb_ori = rb_ori[np.newaxis] if rb_ori.ndim == 3 else rb_ori
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
            axs = Arrows(
                self.rb_pos,
                self.rb_pos + line,
                r_base=r_base,
                r_head=r_head,
                color=tuple(color),
                is_selectable=False,
            )
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

    @property
    def rb_pos(self):
        return self._rb_pos

    @rb_pos.setter
    def rb_pos(self, rb_pos):
        self._rb_pos = rb_pos if len(rb_pos.shape) == 3 else rb_pos[np.newaxis]
        self.n_frames = self._rb_pos.shape[0]

    @property
    def rb_ori(self):
        return self._rb_ori

    @rb_ori.setter
    def rb_ori(self, rb_ori):
        self._rb_ori = rb_ori if len(rb_ori.shape) == 4 else rb_ori[np.newaxis]
        self.n_frames = self._rb_ori.shape[0]

    @property
    def bounds(self):
        return compute_union_of_bounds(self.coords)

    @property
    def current_bounds(self):
        return compute_union_of_current_bounds(self.coords)

    def redraw(self, **kwargs):
        if kwargs.get("current_frame_only", False):
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

    def color_one(self, index, color):
        self.spheres.color_one(index, color)

    def gui(self, imgui):
        _, self.spheres.radius = imgui.drag_float(
            "Sphere Radius".format(self.unique_name),
            self.spheres.radius,
            0.01,
            min_value=0.001,
            max_value=10.0,
            format="%.3f",
        )
        super(RigidBodies, self).gui(imgui)

    def update_frames(self, rb_pos, rb_ori, frames):
        self.rb_pos[frames] = rb_pos
        self.rb_ori[frames] = rb_ori
        self.n_frames = self.rb_pos.shape[0]
        self.redraw()

    def add_frames(self, rb_pos, rb_ori):
        if len(rb_pos.shape) == 2:
            rb_pos = rb_pos[np.newaxis]
        self.rb_pos = np.append(self.rb_pos, rb_pos, axis=0)

        if len(rb_ori.shape) == 3:
            rb_ori = rb_ori[np.newaxis]
        self.rb_ori = np.append(self.rb_ori, rb_ori, axis=0)

        self.n_frames = self.rb_pos.shape[0]
        self.redraw()

    def remove_frames(self, frames):
        self.rb_pos = np.delete(self.rb_pos, frames, axis=0)
        self.rb_ori = np.delete(self.rb_ori, frames, axis=0)

        self.n_frames = self.rb_pos.shape[0]
        self.redraw()

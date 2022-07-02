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

from aitviewer.scene.material import Material
from aitviewer.scene.node import Node
from aitviewer.renderables.spheres import Spheres
from aitviewer.renderables.lines import Lines


class Skeletons(Node):
    """
    Render a skeleton as a set of spheres that are connected with cone-shaped lines.
    """
    def __init__(self,
                 joint_positions,
                 joint_connections,
                 joint_rotations=None,
                 radius=0.01,
                 color=(1.0, 177 / 255, 1 / 255, 1.0),
                 **kwargs):
        """
        Initializer.
        :param joint_positions: A np array of shape (F, J, 3) containing J joint positions over F many time steps.
        :param joint_connections: The definition of the skeleton as a numpy array of shape (N_LINES, 2) where each row
          defines one connection between joints. The max entry in this array must be < J.
        :param joint_rotations: Optional joint orientations as a np array of shape (F, J, 3, 3).
        :param radius: Radius of the sphere located at each joint's position.
        :param color: 4-tuple color, yellow by default.
        :param kwargs: Remaining render arguments.
        """
        if not isinstance(joint_connections, np.ndarray):
            joint_connections = np.array(joint_connections)

        self.joint_positions = joint_positions
        super(Skeletons, self).__init__(n_frames=self.joint_positions.shape[0], color=color, **kwargs)

        self.joint_rotations = joint_rotations
        self.joint_connections = joint_connections

        # Root nodes are not connected to any other joints, so ignore those.
        non_root = ~np.any(joint_connections == -1, axis=-1)
        self.skeleton = joint_connections[non_root]
        self.joint_idxs = np.unique(self.skeleton)

        # Nodes.
        material = Material(color=color)
        self.spheres = Spheres(joint_positions, radius=radius, material=material)
        self.lines = Lines(lines=self.joint_positions[:, self.skeleton].reshape(len(self), -1, 3),
                           mode='lines', r_base=radius, r_tip=radius / 10.0,material=material)
        self._add_nodes(self.spheres, self.lines, show_in_hierarchy=False)

    @property
    def joint_positions(self):
        return self._joint_positions

    @joint_positions.setter
    def joint_positions(self, joint_positions):
        if len(joint_positions.shape) == 2 and joint_positions.shape[-1] == 3:
            joint_positions = joint_positions[np.newaxis]
        assert len(joint_positions.shape) == 3
        self._joint_positions = joint_positions
        self.n_frames = len(joint_positions)

    def redraw(self):
        self.spheres.sphere_positions = self.joint_positions
        self.lines.lines = self.joint_positions[:, self.skeleton].reshape(len(self), -1, 3)
        super().redraw()

    @property
    def bounds(self):
        return self.get_bounds(self.joint_positions)

    @Node.color.setter
    def color(self, color):
        self.material.color = color
        self.spheres.color = color
        self.lines.color = color

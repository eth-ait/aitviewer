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

from aitviewer.renderables.lines import Lines
from aitviewer.renderables.spheres import Spheres
from aitviewer.scene.node import Node


class BoundingBoxes(Node):
    """
    Draw bounding boxes.
    """

    def __init__(self, vertices, thickness=0.005, color=(0.0, 0.0, 1.0, 1.0), **kwargs):
        """
        Initializer.
        :param vertices: Set of 3D coordinates as a np array of shape (N, 8, 3). The vertices will be connected in the
          following way: 0-1-2-3-0 (bottom) 4-5-6-7-4 (top) 0-4 1-5 2-6 3-7 (vertical connections between bottom
          and top).
        :param thickness: Line thickness.
        :param color: Color of the lines.
        """
        if not isinstance(vertices, np.ndarray):
            vertices = np.array(vertices)
        if len(vertices.shape) == 2:
            vertices = vertices[np.newaxis]
        else:
            assert len(vertices.shape) == 3
        assert vertices.shape[1] == 8
        super(BoundingBoxes, self).__init__(n_frames=len(vertices), color=color, **kwargs)

        self.vertices = vertices

        self.lines = Lines(
            lines=self._get_line_coords(),
            mode="lines",
            r_base=thickness,
            color=self.color,
            cast_shadow=False,
        )
        self.spheres = Spheres(positions=self.vertices, radius=thickness, color=self.color, cast_shadow=False)
        self._add_nodes(self.lines, self.spheres, show_in_hierarchy=False)

    @property
    def bounds(self):
        return self.get_bounds(self.vertices)

    @property
    def current_bounds(self):
        return self.get_bounds(self.vertices[self.current_frame_id])

    @staticmethod
    def from_min_max_diagonal(v_min, v_max, **kwargs):
        """
        Create an axis-aligned bounding box from the 3D diagonal.
        :param v_min: np array of shape (N, 3).
        :param v_max: np array of shape (N, 3).
        :return: BoundingBoxes corresponding to the given diagonals.
        """
        vertices = np.zeros((v_min.shape[0], 8, 3), dtype=v_min.dtype)
        vertices[:, 0:4] = v_min[:, np.newaxis]
        vertices[:, 1, 0] = v_max[:, 0]
        vertices[:, 2, 0:2] = v_max[:, 0:2]
        vertices[:, 3, 1] = v_max[:, 1]

        vertices[:, 4:] = v_max[:, np.newaxis]
        vertices[:, 4, 0:2] = v_min[:, 0:2]
        vertices[:, 7, 0] = v_min[:, 0]
        vertices[:, 5, 1] = v_min[:, 1]

        return BoundingBoxes(vertices, **kwargs)

    def _get_line_coords(self):
        lines = np.zeros((self.n_frames, 12 * 2, 3), dtype=self.vertices.dtype)

        # Bottom 0-1-2-3-0.
        lines[:, 0:2] = self.vertices[:, 0:2]
        lines[:, 2:4] = self.vertices[:, 1:3]
        lines[:, 4:6] = self.vertices[:, 2:4]
        lines[:, 6:8] = self.vertices[:, [3, 0]]

        # Top 4-5-6-7-4.
        lines[:, 8:10] = self.vertices[:, 4:6]
        lines[:, 10:12] = self.vertices[:, 5:7]
        lines[:, 12:14] = self.vertices[:, 6:8]
        lines[:, 14:16] = self.vertices[:, [7, 4]]

        # Vertical Connections.
        lines[:, 16:18] = self.vertices[:, [0, 4]]
        lines[:, 18:20] = self.vertices[:, [1, 5]]
        lines[:, 20:22] = self.vertices[:, [2, 6]]
        lines[:, 22:24] = self.vertices[:, [3, 7]]

        return lines

    @Node.color.setter
    def color(self, color):
        self.material.color = color
        self.lines.color = color
        self.spheres.color = color

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

from aitviewer.scene.node import Node
from aitviewer.renderables.lines import Lines


class Arrows(Node):
    """
    Draw an arrow.
    """

    def __init__(self,
                 origins,
                 tips,
                 r_base=0.01,
                 r_head=0.02,
                 p=0.1,
                 color=(0.0, 0.0, 0.5, 1.0),
                 **kwargs):
        """
        Initializer.
        :param origins: Set of 3D coordinates of the base of the arrows as a np array of shape (N, B, 3).
        :param tips: Set of 3D coordinates denoting the tip of the arrow (N, T, 3).
        :param r_base: Radius of the base cylinder.
        :param r_head: Radius of the tip cylinder.
        :param p: Percentage of arrow head on the entire length.
        :param color: Color of the line (4-tuple).
        """
        assert (origins.shape == tips.shape)
        if len(origins.shape) == 2:
            origins = origins[np.newaxis]
            tips = tips[np.newaxis]
        else:
            assert len(origins.shape) == 3
        super(Arrows, self).__init__(n_frames=len(origins), **kwargs)

        self.origins = origins
        self.tips = tips

        # Percentage of arrow head on entire length
        self.p = p

        # Nodes
        self.material.color = color
        self.bases_r = Lines(lines=self.get_line_coords(self.origins, self.mid_points), mode='lines', r_base=r_base,
                             color=color, cast_shadow=False, is_selectable=False)
        self.arrows_r = Lines(lines=self.get_line_coords(self.mid_points, self.tips), mode='lines', r_base=r_head,
                              r_tip=0.0, color=color, cast_shadow=False, is_selectable=False)

        self._add_nodes(self.bases_r, self.arrows_r, show_in_hierarchy=False)

    @property
    def bounds(self):
        return self.arrows_r.bounds

    @property
    def current_bounds(self):
        return self.arrows_r.current_bounds

    @property
    def current_origins(self):
        idx = self.current_frame_id if self.origins.shape[0] > 1 else 0
        return self.origins[idx]

    @current_origins.setter
    def current_origins(self, origins):
        idx = self.current_frame_id if self.origins.shape[0] > 1 else 0
        self.origins[idx] = origins

    @property
    def current_tips(self):
        idx = self.current_frame_id if self.tips.shape[0] > 1 else 0
        return self.tips[idx]

    @current_tips.setter
    def current_tips(self, tips):
        idx = self.current_frame_id if self.tips.shape[0] > 1 else 0
        self.tips[idx] = tips

    @property
    def mid_points(self):
        return self.origins + (self.tips - self.origins) * (1 - self.p)

    def get_line_coords(self, starts, ends):
        c = np.zeros((len(self), (starts.shape[1] + ends.shape[1]), 3), dtype=starts.dtype)
        c[:, 0::2] = starts
        c[:, 1::2] = ends
        return c

    def redraw(self, **kwargs):
        self.bases_r.lines = self.get_line_coords(self.origins, self.mid_points)
        self.bases_r.redraw(**kwargs)
        self.arrows_r.lines = self.get_line_coords(self.mid_points, self.tips)
        self.arrows_r.redraw(**kwargs)

    @Node.color.setter
    def color(self, color):
        self.material.color = color
        self.bases_r.color = color
        self.arrows_r.color = color

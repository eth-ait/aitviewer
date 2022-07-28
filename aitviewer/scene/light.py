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
from aitviewer.renderables.lines import Lines

from aitviewer.scene.node import Node
from aitviewer.scene.camera_utils import look_at
from aitviewer.scene.camera_utils import orthographic_projection


class Light(Node):
    """Simple point light."""

    def __init__(self, intensity_diffuse=1.0, intensity_ambient=1.0, **kwargs):
        super(Light, self).__init__(**kwargs)

        self.intensity_ambient = intensity_ambient
        self.intensity_diffuse = intensity_diffuse
        self.shadow_map_size = 15.0
        self.shadow_map_near = 5.0
        self.shadow_map_far = 50.0

        self._debug_lines = None
        self._show_debug_lines = False

    def mvp(self):
        """Return a model-view-projection matrix to project vertices into the view of the light."""
        s = self.shadow_map_size
        P = orthographic_projection(s, s, self.shadow_map_near, self.shadow_map_far)
        V = look_at(np.array(self.position), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        return (P @ V).astype('f4')
    
    def _update_debug_lines(self):
        lines = np.array([
            [-1, -1, -1], [-1,  1, -1],
            [-1, -1,  1], [-1,  1,  1],
            [ 1, -1, -1], [ 1,  1, -1],
            [ 1, -1,  1], [ 1,  1,  1],
            
            [-1, -1, -1], [-1, -1, 1],
            [-1,  1, -1], [-1,  1, 1],
            [ 1, -1, -1], [ 1, -1, 1],
            [ 1,  1, -1], [ 1,  1, 1],
            
            [-1, -1, -1], [ 1, -1, -1],
            [-1, -1,  1], [ 1, -1,  1],
            [-1,  1, -1], [ 1,  1, -1],
            [-1,  1,  1], [ 1,  1,  1],
        ])

        world_from_ndc = np.linalg.inv(self.mvp())
        lines = np.apply_along_axis(lambda x: (world_from_ndc @ np.append(x, 1.0))[:3] - self.position, 1, lines)

        if self._debug_lines is None:
            self._debug_lines = Lines(lines, position=self.position, r_base=0.2, mode='lines', cast_shadow=False)
            self.add(self._debug_lines)
        else:
            self._debug_lines.lines = lines
            self._debug_lines.redraw()

    def redraw(self, **kwargs):
        if self._debug_lines:
            self._debug_lines.redraw(**kwargs)

    def gui(self, imgui):
        self.gui_position(imgui)
        self.gui_material(imgui, show_advanced=False)
        # Position control
        # _, self.position = imgui.drag_float3('Position##pos', *self.position, 0.1, format='%.2f')


        # Light controls
        _, self.intensity_ambient = imgui.drag_float('Ambient##ambient', self.intensity_ambient, 0.01, min_value=0.0, max_value=1.0,
                                           format='%.2f')
        _, self.intensity_diffuse = imgui.drag_float('Diffuse##diffuse', self.intensity_diffuse, 0.01, min_value=0.0, max_value=1.0,
                                           format='%.2f')

        u_size, self.shadow_map_size = imgui.drag_float('Shadow Map Size', self.shadow_map_size, 0.1, format='%.2f', min_value=0.01, max_value=100.0)
        u_near, self.shadow_map_near = imgui.drag_float('Shadow Map Near', self.shadow_map_near, 0.1, format='%.2f', min_value=0.01, max_value=100.0)
        u_far, self.shadow_map_far  = imgui.drag_float('Shadow Map Far', self.shadow_map_far, 0.1, format='%.2f', min_value=0.01, max_value=100.0)

        u_show, self._show_debug_lines = imgui.checkbox('Show Shadowmap Frustum', self._show_debug_lines)

        if self._show_debug_lines:
            if self._debug_lines:
                self._debug_lines.enabled = True

            if u_size or u_near or u_far or u_show:
                self._update_debug_lines()
        else:
            if self._debug_lines:
                self._debug_lines.enabled = False
        
        
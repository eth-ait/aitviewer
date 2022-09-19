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
from functools import lru_cache
import numpy as np
from aitviewer.renderables.lines import Lines

from aitviewer.scene.node import Node
from aitviewer.scene.camera_utils import look_at
from aitviewer.scene.camera_utils import orthographic_projection


class Light(Node):
    """Simple point light."""

    def __init__(self, intensity_diffuse=1.0, intensity_ambient=1.0, shadow_enabled=True, **kwargs):
        super(Light, self).__init__(icon='\u0085', **kwargs)

        self.intensity_ambient = intensity_ambient
        self.intensity_diffuse = intensity_diffuse

        self.shadow_enabled = shadow_enabled
        self.shadow_map = None
        self.shadow_map_framebuffer = None

        self.shadow_map_size = 15.0
        self.shadow_map_near = 5.0
        self.shadow_map_far = 50.0

        self._debug_lines = None
        self._show_debug_lines = False

    def create_shadowmap(self, ctx):
        if self.shadow_map is None:
            shadow_map_size = 8192, 8192

            # Setup shadow mapping
            self.shadow_map = ctx.depth_texture(shadow_map_size)
            self.shadow_map.compare_func = '>'
            self.shadow_map.repeat_x = False
            self.shadow_map.repeat_y = False
            self.shadow_map_framebuffer = ctx.framebuffer(depth_attachment=self.shadow_map)

    def use(self, ctx):
        if not self.shadow_map:
            self.create_shadowmap(ctx)

        self.shadow_map_framebuffer.clear()
        self.shadow_map_framebuffer.use()

    @staticmethod
    @lru_cache()
    def _compute_light_matrix(position, size, near, far):
        P = orthographic_projection(size, size, near, far)
        V = look_at(np.array(position), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        return (P @ V).astype('f4')

    def mvp(self):
        """Return a model-view-projection matrix to project vertices into the view of the light."""
        return self._compute_light_matrix(tuple(self.position), self.shadow_map_size, self.shadow_map_near, self.shadow_map_far)

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

    @Node.position.setter
    def position(self, position):
        self._position = position
        for n in self.nodes:
            n.position = position
        self._update_debug_lines()

    def redraw(self, **kwargs):
        if self._debug_lines:
            self._debug_lines.redraw(**kwargs)

    def gui(self, imgui):
        self.gui_position(imgui)
        self.gui_material(imgui, show_advanced=False)

        # Light controls
        _, self.intensity_ambient = imgui.drag_float('Ambient##ambient', self.intensity_ambient, 0.01, min_value=0.0, max_value=1.0,
                                           format='%.2f')
        _, self.intensity_diffuse = imgui.drag_float('Diffuse##diffuse', self.intensity_diffuse, 0.01, min_value=0.0, max_value=1.0,
                                           format='%.2f')

        _, self.shadow_enabled = imgui.checkbox('Enable Shadows', self.shadow_enabled)

        u_size, self.shadow_map_size = imgui.drag_float('Shadow Map Size', self.shadow_map_size, 0.1, format='%.2f', min_value=0.01, max_value=100.0)
        u_near, self.shadow_map_near = imgui.drag_float('Shadow Map Near', self.shadow_map_near, 0.1, format='%.2f', min_value=0.01, max_value=100.0)
        u_far, self.shadow_map_far  = imgui.drag_float('Shadow Map Far', self.shadow_map_far, 0.1, format='%.2f', min_value=0.01, max_value=100.0)

        u_show, self._show_debug_lines = imgui.checkbox('Show Shadow Map Frustum', self._show_debug_lines)

        if self._show_debug_lines:
            if self._debug_lines:
                self._debug_lines.enabled = True

            if u_size or u_near or u_far or u_show:
                self._update_debug_lines()
        else:
            if self._debug_lines:
                self._debug_lines.enabled = False


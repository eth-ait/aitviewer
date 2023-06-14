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
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.scene.camera_utils import look_at, orthographic_projection
from aitviewer.scene.material import Material
from aitviewer.scene.node import Node
from aitviewer.utils.utils import (
    direction_from_spherical_coordinates,
    spherical_coordinates_from_direction,
)


class Light(Node):
    """Simple point light."""

    def __init__(
        self,
        light_color=(1.0, 1.0, 1.0),
        elevation=-90.0,
        azimuth=0.0,
        strength=1.0,
        shadow_enabled=True,
        **kwargs,
    ):
        kwargs["gui_material"] = False
        super(Light, self).__init__(icon="\u0085", **kwargs)

        self._light_color = light_color
        self.strength = strength
        self._azimuth = azimuth
        self._elevation = elevation
        self.update_rotation()

        self.shadow_enabled = shadow_enabled
        self.shadow_map = None
        self.shadow_map_framebuffer = None

        self.shadow_map_size = 15.0
        self.shadow_map_near = 5.0
        self.shadow_map_far = 50.0

        self._debug_lines = None
        self._show_debug_lines = False

        rot = np.eye(3)
        rot[2, 2] = -1
        self.mesh = RigidBodies(
            np.array([[0, 0, 0]], dtype=np.float32),
            np.array([rot], dtype=np.float32),
            radius=0.08,
            length=0.4,
            is_selectable=False,
        )
        self.mesh.spheres.material.diffuse = 0.0
        self.mesh.spheres.material.ambient = 1.0
        self.mesh.spheres.color = (*tuple(light_color), 1.0)
        self.add(self.mesh, show_in_hierarchy=False, enabled=False)

    @classmethod
    def facing_origin(cls, **kwargs):
        pos = np.array(kwargs["position"])
        dir = -pos / np.linalg.norm(pos)
        theta, phi = spherical_coordinates_from_direction(dir, degrees=True)
        return cls(elevation=theta, azimuth=phi, **kwargs)

    @property
    def elevation(self):
        return self._elevation

    @elevation.setter
    def elevation(self, elevation):
        self._elevation = elevation
        self.update_rotation()

    @property
    def azimuth(self):
        return self._azimuth

    @azimuth.setter
    def azimuth(self, azimuth):
        self._azimuth = azimuth
        self.update_rotation()

    @property
    def light_color(self):
        return self._light_color

    @light_color.setter
    def light_color(self, light_color):
        self._light_color = light_color
        self.mesh.spheres.color = (*tuple(light_color), 1.0)

    def update_rotation(self):
        self.rotation = look_at(np.array([0, 0, 0]), self.direction, np.array([0.0, 1.0, 0.0]))[:3, :3].T

    def create_shadowmap(self, ctx):
        if self.shadow_map is None:
            shadow_map_size = 8192, 8192

            # Setup shadow mapping
            self.shadow_map = ctx.depth_texture(shadow_map_size)
            self.shadow_map.compare_func = ">"
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
    def _compute_light_matrix(position, direction, size, near, far):
        P = orthographic_projection(size, size, near, far)
        p = np.array(position)
        d = np.array(direction)
        V = look_at(p, p + d, np.array([0.0, 1.0, 0.0]))
        return (P @ V).astype("f4")

    def mvp(self):
        """Return a model-view-projection matrix to project vertices into the view of the light."""
        return self._compute_light_matrix(
            tuple(self.position),
            tuple(self.direction),
            self.shadow_map_size,
            self.shadow_map_near,
            self.shadow_map_far,
        )

    def _update_debug_lines(self):
        lines = np.array(
            [
                [-1, -1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, 1, -1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
                [-1, -1, -1],
                [1, -1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [-1, 1, -1],
                [1, 1, -1],
                [-1, 1, 1],
                [1, 1, 1],
            ]
        )

        size = self.shadow_map_size
        view_from_ndc = np.linalg.inv(orthographic_projection(size, size, self.shadow_map_near, self.shadow_map_far))
        lines = np.apply_along_axis(lambda x: (view_from_ndc @ np.append(x, 1.0))[:3], 1, lines)

        if self._debug_lines is None:
            self._debug_lines = Lines(lines, r_base=0.05, mode="lines", cast_shadow=False, is_selectable=False)
            self.add(self._debug_lines, show_in_hierarchy=False)
        else:
            self._debug_lines.lines = lines
            self._debug_lines.redraw()

    def render_outline(self, *args, **kwargs):
        if self.mesh.enabled:
            self.mesh.spheres.render_outline(*args, **kwargs)

    @Node.position.setter
    def position(self, position):
        super(Light, self.__class__).position.fset(self, position)
        self._update_debug_lines()

    @property
    def bounds(self):
        return self.mesh.bounds

    @property
    def current_bounds(self):
        return self.mesh.current_bounds

    @property
    def direction(self):
        return direction_from_spherical_coordinates(self.elevation, self.azimuth, degrees=True)

    def redraw(self, **kwargs):
        if self._debug_lines:
            self._debug_lines.redraw(**kwargs)

    def gui_affine(self, imgui):  # Position controls
        up, pos = imgui.drag_float3(
            "Position##pos{}".format(self.unique_name),
            *self.position,
            1e-2,
            format="%.2f",
        )
        if up:
            self.position = pos

    def gui(self, imgui):
        uc, light_color = imgui.color_edit3("Color", *self.light_color)
        if uc:
            self.light_color = light_color
        _, self.strength = imgui.drag_float(
            "Strength",
            self.strength,
            0.01,
            min_value=0.0,
            max_value=10.0,
            format="%.2f",
        )
        u_el, elevation = imgui.drag_float(
            "Elevation",
            self.elevation,
            0.1,
            min_value=-90,
            max_value=90.0,
            format="%.2f",
        )
        if u_el:
            self.elevation = elevation
        u_az, azimuth = imgui.drag_float(
            "Azimuth",
            self.azimuth,
            0.1,
            min_value=-360.0,
            max_value=360.0,
            format="%.2f",
        )
        if u_az:
            self.azimuth = np.remainder(azimuth, 360.0)

        imgui.spacing()
        _, self.shadow_enabled = imgui.checkbox("Enable Shadows", self.shadow_enabled)
        u_size, self.shadow_map_size = imgui.drag_float(
            "Shadowmap size",
            self.shadow_map_size,
            0.1,
            format="%.2f",
            min_value=0.01,
            max_value=100.0,
        )
        u_near, self.shadow_map_near = imgui.drag_float(
            "Shadowmap Near",
            self.shadow_map_near,
            0.1,
            format="%.2f",
            min_value=0.01,
            max_value=100.0,
        )
        u_far, self.shadow_map_far = imgui.drag_float(
            "Shadowmap Far",
            self.shadow_map_far,
            0.1,
            format="%.2f",
            min_value=0.01,
            max_value=100.0,
        )
        _, self.mesh.enabled = imgui.checkbox("Show light", self.mesh.enabled)
        u_show, self._show_debug_lines = imgui.checkbox("Show Frustum", self._show_debug_lines)

        if self._show_debug_lines:
            if self._debug_lines:
                self._debug_lines.enabled = True

            if u_size or u_near or u_far or u_show:
                self._update_debug_lines()
        else:
            if self._debug_lines:
                self._debug_lines.enabled = False

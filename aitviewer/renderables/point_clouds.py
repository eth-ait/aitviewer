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
import moderngl
import numpy as np
from moderngl_window.opengl.vao import VAO

from aitviewer.scene.node import Node
from aitviewer.shaders import (
    get_fragmap_program,
    get_outline_program,
    get_simple_unlit_program,
)
from aitviewer.utils.decorators import hooked


class PointClouds(Node):
    """
    Draw a point clouds man!
    """

    def __init__(
        self,
        points,
        colors=None,
        point_size=5.0,
        color=(0.0, 0.0, 1.0, 1.0),
        z_up=False,
        icon="\u008c",
        pickable=True,
        **kwargs,
    ):
        """
        A sequence of point clouds. Each point cloud can have a varying number of points.
        :param points: Sequence of points (F, P, 3)
        :param colors: Sequence of Colors (F, C, 4) or None. If None, all points are colored according to `color`.
        :param point_size: Initial point size.
        :param color: Default color applied to all points of all frames if `colors` is not provided.
        :param z_up: If true the point cloud rotation matrix is initialized to convert from z-up to y-up data.
        """
        assert isinstance(points, list) or isinstance(points, np.ndarray)
        if colors is not None:
            assert len(colors) == len(points)

        self.points = points
        super(PointClouds, self).__init__(n_frames=len(self.points), color=color, icon=icon, **kwargs)

        self.fragmap = pickable

        # Render passes
        self.outline = True

        self.colors = colors
        self.point_size = point_size
        self.max_n_points = max([p.shape[0] for p in self.points])

        self.vao = VAO("points", mode=moderngl.POINTS)

        if z_up:
            self.rotation = np.matmul(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), self.rotation)

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        self._points = points
        self.n_frames = len(points)
        self.max_n_points = max([p.shape[0] for p in self.points])

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, colors):
        # Colors cannot be empty
        if colors is None:
            self._colors = [self.color]
        elif isinstance(colors, tuple) and len(colors) == 4:
            self._colors = [colors]
        elif isinstance(colors, list):
            assert len(colors) == self.n_frames
            assert colors[0].shape[-1] == 4
            self._colors = colors
        else:
            raise ValueError("Invalid colors: {}".format(colors))

    @Node.color.setter
    def color(self, color):
        """Update the color of the point cloud."""
        # This is a bit ill-defined because point clouds can have per-point colors, in which case we probably do
        # not want to override them with a single uniform color. We disallow this for now. The function is still useful
        # though to change the alpha, even if a point cloud has per-point colors.
        self.material.color = color
        if self.is_renderable:
            single_color = isinstance(self.colors[0], tuple) and len(self.colors[0]) == 4
            if single_color:
                self.colors = tuple(color)
            else:
                # Only update the colors if the alpha changed. Take any frame and any point to check if the alpha
                # changed because we always change every frame and every point.
                alpha_changed = abs(color[-1] - self.colors[0][0, -1]) > 0
                if alpha_changed:
                    for i in range(self.n_frames):
                        self.colors[i][..., -1] = color[-1]
        self.redraw()

    @property
    def current_points(self):
        idx = self.current_frame_id if len(self.points) > 1 else 0
        return self.points[idx]

    @property
    def current_colors(self):
        if len(self.colors) == 1:
            n_points = self.current_points.shape[0]
            return np.full((n_points, 4), self.colors[0])
        else:
            idx = self.current_frame_id if len(self.colors) > 1 else 0
            return self.colors[idx]

    @property
    def bounds(self):
        if len(self.points) == 0:
            return np.array([[0, 0], [0, 0], [0, 0]])

        bounds = np.array([[np.inf, np.NINF], [np.inf, np.NINF], [np.inf, np.NINF]])
        for i in range(len(self.points)):
            b = self.get_bounds(self.points[i])
            bounds[:, 0] = np.minimum(bounds[:, 0], b[:, 0])
            bounds[:, 1] = np.maximum(bounds[:, 1], b[:, 1])
        return bounds

    @property
    def current_bounds(self):
        return self.get_bounds(self.current_points)

    def on_frame_update(self):
        """Called whenever a new frame must be displayed."""
        super().on_frame_update()
        self.redraw()

    def redraw(self, **kwargs):
        """Upload the current frame data to the GPU for rendering."""
        if not self.is_renderable:
            return

        points = self.current_points.astype("f4").tobytes()
        colors = self.current_colors.astype("f4").tobytes()

        # Resize the VBOs if necessary. This can happen if new points are set after the `make_renderable` has been
        # called.
        if self.max_n_points * 3 * 4 > self.vbo_points.size:
            self.vbo_points.orphan(self.max_n_points * 3 * 4)
            self.vbo_colors.orphan(self.max_n_points * 4 * 4)

        self.vbo_points.write(points)
        self.vbo_colors.write(colors)

    def _clear_buffer(self):
        self.vbo_points.clear()
        self.vbo_colors.clear()

    # noinspection PyAttributeOutsideInit
    @Node.once
    def make_renderable(self, ctx):
        ctx.point_size = self.point_size

        self.prog = get_simple_unlit_program()
        self.outline_program = get_outline_program("mesh_positions.vs.glsl")
        self.fragmap_program = get_fragmap_program("mesh_positions.vs.glsl")

        self.vbo_points = ctx.buffer(reserve=self.max_n_points * 3 * 4, dynamic=True)
        self.vbo_colors = ctx.buffer(reserve=self.max_n_points * 4 * 4, dynamic=True)
        self.vbo_points.write(self.current_points.astype("f4").tobytes())
        self.vbo_colors.write(self.current_colors.astype("f4").tobytes())
        self.vao.buffer(self.vbo_points, "3f", ["in_position"])
        self.vao.buffer(self.vbo_colors, "4f", ["in_color"])

        self.positions_vao = VAO("{}:positions".format(self.unique_name), mode=moderngl.POINTS)
        self.positions_vao.buffer(self.vbo_points, "3f", ["in_position"])

    def render(self, camera, **kwargs):
        self.set_camera_matrices(self.prog, camera, **kwargs)
        # Draw only as many points as we have set in the buffer.
        self.vao.render(self.prog, vertices=len(self.current_points))

    def render_positions(self, prog):
        if self.is_renderable:
            self.positions_vao.render(prog, vertices=len(self.current_points))

    @hooked
    def release(self):
        if self.is_renderable:
            self.vao.release()
            self.positions_vao.release(buffer=False)

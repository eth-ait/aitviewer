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
from aitviewer.shaders import get_simple_unlit_program
from moderngl_window.opengl.vao import VAO
import moderngl


class PointClouds(Node):
    """
    Draw a point clouds man!
    """

    def __init__(self, points, colors=None, lengths=None, point_size=5.0, color=(0.0, 0.0, 1.0, 1.0), **kwargs):
        """
        A sequence of point clouds. Each point cloud can have a varying number of points.
        Internally represented as a list of arrays.
        :param points: Sequence of points (F, P, 3)
        :param colors: Sequence of Colors (F, C, 4)
        :param lengths: Length mask for each frame of points denoting the usable part of the array
        :param point_size: Initial point size
        """
        self.points = points
        super(PointClouds, self).__init__(n_frames=len(self.points), color=color, **kwargs)

        self.n_points = self.points.shape[1]
        self.colors = colors
        self.point_size = point_size
        self.lengths = lengths

        self.vao = VAO("points", mode=moderngl.POINTS)

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, points):
        if len(points.shape) == 2 and points.shape[-1] == 3:
            points = points[np.newaxis]
        assert len(points.shape) == 3

        self._points = points
        self.n_frames = len(points)

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, colors):
        # Colors cannot be empty
        if colors is None:
            self._colors = np.full((self.n_frames, self.n_points, 4), self.color)
        elif isinstance(colors, tuple) and len(colors) == 4:
            self._vertex_colors = np.full((self.n_frames, self.n_points, 4), colors)
        else:
            # Ensure batch dimension is in place
            if len(colors.shape) == 2 and colors.shape[-1] == 4:
                colors = colors[np.newaxis]
            assert len(colors.shape) == 3

            self._colors = colors

    @Node.color.setter
    def color(self, color):
        alpha_changed = np.abs((np.array(color) - np.array(self._color)))[-1] > 0
        self.material.color = color
        if self.is_renderable:
            # If alpha changed, don't update all colors
            if alpha_changed:
                self._colors[..., -1] = color[-1]
            else:
                self.colors = color

        self.redraw()

    @property
    def current_points(self):
        return self.points[self.current_frame_id]

    @property
    def current_colors(self):
        return self.colors[self.current_frame_id]

    @property
    def bounds(self):
        return self.get_bounds(self.points)

    def on_frame_update(self):
        """Called whenever a new frame must be displayed."""
        super().on_frame_update()
        self.redraw()

    def redraw(self):
        """Upload the current frame data to the GPU for rendering."""
        if not self.is_renderable:
            return

        # Each write call takes about 1-2 ms
        points = self.current_points
        colors = self.current_colors

        self._clear_buffer()
        self.vbo_points.write(points.astype('f4').tobytes())
        self.vbo_colors.write(colors.astype('f4').tobytes())

    def _clear_buffer(self):
        self.vbo_points.clear()
        self.vbo_colors.clear()

    @Node.once
    def make_renderable(self, ctx):
        ctx.point_size = self.point_size

        self.prog = get_simple_unlit_program()
        self.vbo_points = ctx.buffer(self.current_points.astype('f4').tobytes())
        self.vbo_colors = ctx.buffer(self.current_colors.astype('f4').tobytes())
        self.vao.buffer(self.vbo_points, '3f', ['in_position'])
        self.vao.buffer(self.vbo_colors, '4f', ['in_color'])

    def render(self, camera, **kwargs):
        self.set_camera_matrices(self.prog, camera, **kwargs)
        self.vao.render(self.prog)

    def to_npz(self, path=''):
        np.savez_compressed(path + self.name + '.npz',
                            points=self.points,
                            colors=self.colors,
                            lengths=self.lengths)

    @classmethod
    def from_npz(cls, npz_path):
        """Load a sequence from an npz file. The filename becomes the name of the sequence"""
        data = np.load(npz_path, allow_pickle=True)
        name = os.path.splitext(os.path.basename(npz_path))[0]
        return cls(
            points= data['points'] if 'points' in data and data['points'].size > 1 else None,
            colors=data['colors'] if 'colors' in data and data['colors'].size > 1 else None,
            lengths=data['lengths'] if 'lengths' in data and data['lengths'].dtype != object else None
        )

    def gui(self, imgui):
        super().gui(imgui)
        if imgui.button("Save"):
            self.to_npz()
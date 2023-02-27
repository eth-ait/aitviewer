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
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.spheres import Spheres
from aitviewer.utils.so3 import aa2rot_numpy as aa2rot
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # Draw 10k lines.
    grid_xz = np.mgrid[-5:5:0.1, -5:5:0.1]
    n_lines = grid_xz.shape[1] * grid_xz.shape[2]
    print("Number of lines", n_lines)

    xz_coords = np.reshape(grid_xz, (2, -1)).T
    line_starts = np.concatenate([xz_coords[:, 0:1], np.zeros((n_lines, 1)), xz_coords[:, 1:2]], axis=-1)
    line_ends = line_starts.copy()
    line_ends[:, 1] = 1.0
    line_strip = np.zeros((2 * n_lines, 3))
    line_strip[::2] = line_starts
    line_strip[1::2] = line_ends
    line_renderable = Lines(line_strip, mode="lines")

    # Draw some spheres on top of each line.
    line_dirs = line_ends - line_starts
    sphere_positions = line_ends + 0.1 * (line_dirs / np.linalg.norm(line_dirs, axis=-1, keepdims=True))
    spheres = Spheres(sphere_positions, color=(1.0, 0.0, 1.0, 1.0))

    # Draw rigid bodies on top of each sphere (a rigid body is just a sphere with three axes representing its
    # orientation).
    rb_positions = line_ends + 0.4 * (line_dirs / np.linalg.norm(line_dirs, axis=-1, keepdims=True))
    angles = np.arange(0.0, 2 * np.pi, step=2 * np.pi / n_lines)[:, None]
    axes = np.zeros((n_lines, 3))
    axes[:, 2] = 1.0
    rb_orientations = aa2rot(angles * axes)
    rbs = RigidBodies(rb_positions, rb_orientations)

    # Display in viewer.
    v = Viewer()
    v.scene.add(line_renderable, spheres, rbs)
    v.scene.camera.position = np.array([0.0, 1.3, 5.0])
    v.run()

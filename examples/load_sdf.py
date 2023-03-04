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

from aitviewer.renderables.sdf import SDF
from aitviewer.viewer import Viewer

volume = np.load(open("resources/dragon.npz", "rb"))["volume"]

v = Viewer()

o_shells = np.linspace(0.3, 1.5, 5)
o_shell_colors = np.linspace(np.array([0.7, 0.6, 0.6, 1.0]), np.array([0.7, 0.2, 0.2, 1.0]), len(o_shells))

i_shells = -np.linspace(0.1, 0.8, 5)
i_shell_colors = np.linspace(np.array([0.6, 0.6, 0.7, 1.0]), np.array([0.2, 0.2, 0.7, 1.0]), len(i_shells))

shells = np.hstack((o_shells, i_shells))
shell_colors = np.vstack((o_shell_colors, i_shell_colors))

s = SDF(
    volume,
    size=(np.array(volume.shape) / float(max(volume.shape)) * 6),
    mc_step_size=1,
    level=0.0,
    shells=shells,
    shell_colors=shell_colors,
)
s.clip_extents = (0.5, 1.0, 1.0)
s.mesh.color = (0.7, 0.7, 0.7, 0.5)

v.scene.add(s)
v.scene.camera.position = (10, 4, -2)
v.scene.camera.target = (1, 1, 4)
v.auto_set_camera_target = False
v.run()

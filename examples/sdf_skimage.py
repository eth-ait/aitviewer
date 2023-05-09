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

if __name__ == "__main__":
    # Load a signed distance field from a file and display it with different level sets it in the viewer.
    # This example uses skimage's marching cubes algorithm to mesh the SDF.
    # For a faster alternative that uses a GPU shader see examples\sdf_gpu_shader.py.
    volume = np.load("resources/dragon.npz", "rb")["volume"]

    v = Viewer()

    s = SDF.with_level_sets(
        volume,
        size=(np.array(volume.shape) / float(max(volume.shape)) * 6),
        level=0.0,
        inside_levels=-np.linspace(0.1, 0.8, 5),
        outside_levels=np.linspace(0.3, 1.4, 5),
    )
    s.clip_extents = (0.5, 1.0, 1.0)

    v.scene.add(s)
    v.scene.camera.position = (10, 4, -2)
    v.scene.camera.target = (1, 1, 4)
    v.auto_set_camera_target = False
    v.run()

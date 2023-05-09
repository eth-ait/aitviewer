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

from aitviewer.renderables.volume import Volume
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # Load a signed distance field from a file and mesh with a marching cubes algorithm implemented in a GPU shader.
    volume: np.ndarray = np.load("resources/dragon.npz")["volume"]
    SIZE = np.array(volume.shape[::-1], np.float32) / max(volume.shape) * 6
    LEVEL = 0.0

    vol = Volume(volume, SIZE, LEVEL)

    v = Viewer()
    v.scene.add(vol)
    v.scene.camera.position = (-2, 4, 10)
    v.scene.camera.target = (4, 1, 1)
    v.auto_set_camera_target = False
    v.run()

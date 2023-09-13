# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
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

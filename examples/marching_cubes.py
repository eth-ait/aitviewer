import numpy as np

from aitviewer.renderables.meshes import MarchingCubesMeshes
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    volume: np.ndarray = np.load(open("resources/dragon.npz", "rb"))["volume"]
    SIZE = np.array(volume.shape[::-1], np.float32) / max(volume.shape)
    LEVEL = 0.0

    mc = MarchingCubesMeshes(volume, SIZE, LEVEL)

    v = Viewer()
    v.scene.add(mc)
    v.run()

# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
from aitviewer.streamables.webcam import Webcam
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # Stream from a webcam.
    w = Webcam()

    # Display in viewer.
    v = Viewer()
    v.scene.add(w)
    v.run()

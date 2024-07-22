# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    """Load a BVH file and display it."""
    skeletons = Skeletons.from_bvh(r"./resources/bvh/sample.bvh")
    viewer = Viewer()
    viewer.scene.add(skeletons)

    # Place the camera such that the skeleton is in the center of the view.
    viewer.center_view_on_node(skeletons)
    viewer.auto_set_camera_target = False

    viewer.run_animations = True
    viewer.run()

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

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.spheres import Spheres
from aitviewer.utils.decorators import hooked
from aitviewer.viewer import Viewer


class ClickingViewer(Viewer):
    """
    This viewer just allows to place spheres onto vertices that we clicked on with the mouse.
    This only works if the viewer is in "inspection" mode (Click I).
    """

    title = "Clicking Viewer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_virtual_marker(self, intersection):
        # Create a marker sequence for the entire sequence at once.
        # First get the positions.
        seq = intersection.node
        positions = seq.vertices[:, intersection.vert_id : intersection.vert_id + 1] + seq.position[np.newaxis]

        ms = Spheres(positions, name="{}".format(intersection.vert_id), radius=0.005)
        ms.current_frame_id = seq.current_frame_id
        self.scene.add(ms)

    def mouse_press_event(self, x: int, y: int, button: int):
        if not self.imgui_user_interacting and self.selected_mode == "inspect":
            result = self.mesh_mouse_intersection(x, y)
            if result is not None:
                self.interact_with_sequence(result, button)
        else:
            # Pass the event to the viewer if we didn't handle it.
            super().mouse_press_event(x, y, button)

    def interact_with_sequence(self, intersection, button):
        """
        Called when the user clicked on a mesh while holding ctrl.
        :param intersection: The result of intersecting the user click with the scene.
        :param button: The mouse button the user clicked.
        """
        if button == 1:  # left mouse
            self.add_virtual_marker(intersection)


if __name__ == "__main__":
    # This example shows how we can implement clicking vertices on a mesh.
    # To implement this, we subclass the viewer. This is also a helpful
    # example to show how you can use the viewer in your own project.
    #
    # To enable clicking, put the viewer into "inspection" mode by hitting
    # the `I` key. In this mode, a new window pops up that displays the face
    # and nearest vertex IDs for the current mouse position.
    #
    # To place spheres onto vertices, it might be easier to show the edges
    # by hitting the `E` key.
    v = ClickingViewer()

    # Create a neutral SMPL T Pose.
    # This also works with `smplh` or `smplx` model type (but there's no neutral model for SMPL-H).
    smpl_layer = SMPLLayer(model_type="smpl", gender="neutral", device=C.device)
    poses = np.zeros([1, smpl_layer.bm.NUM_BODY_JOINTS * 3])
    smpl_seq = SMPLSequence(poses, smpl_layer)

    # Display in viewer.
    v.scene.add(smpl_seq)
    v.run()

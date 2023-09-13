# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import PinholeCamera
from aitviewer.utils import path
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # Create a neutral SMPL T Pose.
    smpl_template = SMPLSequence.t_pose(SMPLLayer(model_type="smpl", gender="neutral", device=C.device), name="SMPL")

    d = 10  # Distance from the object at start and end.
    r = 3  # Radius of the circle around the object.
    h = 2  # Height of the circle.

    # Create a path with a line followed by a circle followed by another line.
    first = path.line(start=(r, h, d), end=(r, h, 0), num=100)
    circle = path.circle(
        center=(0, h, 0),
        radius=r,
        num=int(314 * 2 * r / d),
        start_angle=360,
        end_angle=0,
    )
    second = path.line(start=(r, h, 0), end=(r, h, -d), num=100)
    positions = np.vstack((first, circle, second))

    # Use a fixed target.
    targets = np.array([0, 0, 0])

    # Display in viewer.
    v = Viewer()

    # Create a Pihole camera that uses the positions and targets we computed.
    camera = PinholeCamera(positions, targets, v.window_size[0], v.window_size[1], viewer=v)

    # Add the camera and the SMPL sequence to the scene.
    v.scene.add(camera, smpl_template)

    # Set the camera as the current viewer camera.
    v.set_temp_camera(camera)

    v.run()

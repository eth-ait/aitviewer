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

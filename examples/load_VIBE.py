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
import os
import re

import joblib
import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import WeakPerspectiveCamera
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer

# Set to True for rendering in headless mode, no window will be created and
# a video will be exported to 'headless/test.mp4' in the export directory
HEADLESS = False


if __name__ == "__main__":
    # Load camera and SMPL data from the output of the VIBE demo from https://github.com/mkocabas/VIBE
    data = joblib.load(open("resources/vibe/vibe_output.pkl", "rb"))
    camera_info = data[1]["orig_cam"]
    poses = data[1]["pose"]
    betas = data[1]["betas"]

    # Create the viewer, set a size that has 16:9 aspect ratio to match the input data
    if HEADLESS:
        viewer = HeadlessRenderer(size=(1600, 900))
    else:
        viewer = Viewer(size=(1600, 900))

    # Instantiate an SMPL sequence using the parameters from the data file.
    # We rotate the sequence by 180 degrees around the x axis to flip the y and z axis
    # because VIBE outputs the pose in a different coordinate system.
    smpl_layer = SMPLLayer(model_type="smpl", gender="neutral", device=C.device)
    smpl_sequence = SMPLSequence(
        poses_body=poses[:, 3 : 24 * 3],
        poses_root=poses[:, 0:3],
        betas=betas,
        smpl_layer=smpl_layer,
        rotation=aa2rot_numpy(np.array([1, 0, 0]) * np.pi),
    )

    # Size in pixels of the image data.
    cols, rows = 1920, 1080

    # Create a sequence of weak perspective cameras.
    cameras = WeakPerspectiveCamera(camera_info[:, :2], camera_info[:, 2:], cols, rows, far=3, viewer=viewer)

    # Path to the directory containing the video frames.
    images_path = "resources/vibe/frames"

    # Sort images by frame number in the filename.
    regex = re.compile(r"(\d*)$")

    def sort_key(x):
        name = os.path.splitext(x)[0]
        return int(regex.search(name).group(0))

    # Create a billboard.
    billboard = Billboard.from_camera_and_distance(
        cameras,
        cameras.far - 1e-6,
        cols,
        rows,
        [os.path.join(images_path, f) for f in sorted(os.listdir(images_path), key=sort_key)],
    )

    # Add all the objects to the scene.
    viewer.scene.add(smpl_sequence, billboard, cameras)

    # Set the weak perspective cameras as the current camera used by the viewer.
    # This is a temporary setting, moving the camera will result in switching back to the default (pinhole) camera.
    viewer.set_temp_camera(cameras)

    # Viewer settings.
    viewer.auto_set_floor = False
    viewer.playback_fps = 25
    viewer.scene.fps = 25
    viewer.scene.floor.position = np.array([0, -1.15, 0])
    viewer.scene.origin.enabled = False
    viewer.shadows_enabled = False

    if HEADLESS:
        viewer.save_video(video_dir=os.path.join(C.export_dir, "headless/vibe.mp4"), output_fps=25)
    else:
        viewer.run()

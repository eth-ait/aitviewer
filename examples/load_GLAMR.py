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
import pickle
import re

import numpy as np
import torch

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # Create the viewer, set a size that has 16:9 aspect ratio to match the input data
    viewer = Viewer(size=(1600, 900))

    # Load camera and SMPL data from the output of the GLAMR dynamic camera demo from https://github.com/NVlabs/GLAMR
    data = pickle.load(open("resources/glamr/running_seed1.pkl", "rb"))

    person = data["person_data"][0]
    pose = person["smpl_pose"]
    betas = person["smpl_beta"]
    trans = person["root_trans_world"]
    ori = person["smpl_orient_world"]

    # Define a postprocess function for the SMPL sequence,
    # we use this to apply the translation from the data to the root node.
    def post_fk_func(
        self: SMPLSequence,
        vertices: torch.Tensor,
        joints: torch.Tensor,
        current_frame_only: bool,
    ):
        # Select the translation of the current frame if current_frame_only is True, otherwise select all frames.
        t = trans[[self.current_frame_id]] if current_frame_only else trans[:]
        t = torch.from_numpy(t).to(dtype=joints.dtype, device=joints.device)

        # Subtract the position of the root joint from all vertices and joint positions and add the root translation.
        cur_root_trans = joints[:, [0], :]
        vertices = vertices - cur_root_trans + t[:, None, :]
        joints = joints - cur_root_trans + t[:, None, :]
        return vertices, joints

    # Instantiate an SMPL sequence using the parameters from the data file.
    # We set z_up=True because GLAMR data is using z_up coordinates.
    smpl_layer = SMPLLayer(model_type="smpl", gender="neutral", device=C.device)
    smpl_sequence = SMPLSequence(
        poses_body=pose,
        poses_root=ori,
        betas=betas,
        is_rigged=False,
        smpl_layer=smpl_layer,
        color=(149 / 255, 149 / 255, 149 / 255, 0.8),
        z_up=True,
        post_fk_func=post_fk_func,
    )

    # Draw an outline around the SMPL mesh.
    smpl_sequence.mesh_seq.draw_outline = True

    # Transform y_up coordinates to z_up coordinates.
    z_up_from_y_up = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], np.float32).T

    # Camera extrinsics expect z up data, we transform each frame to expect y up data instead
    # since this is the coordinate system used by the viewer.
    Rt = data["cam_pose"]
    for i in range(Rt.shape[0]):
        Rt[i] = Rt[i] @ z_up_from_y_up
    K = person["cam_K"]

    # Create a sequence of cameras from camera extrinsics and intrinsics.
    cols, rows = 3840, 2160
    camera = OpenCVCamera(K, Rt[:, :3, :], cols, rows, viewer=viewer)

    # Path to the directory containing the video frames.
    images_path = "resources/glamr/frames"

    # Sort images by frame number in the filename.
    regex = re.compile(r"(\d*)$")

    def sort_key(x):
        name = os.path.splitext(x)[0]
        return int(regex.search(name).group(0))

    # Create a billboard
    billboard = Billboard.from_camera_and_distance(
        camera,
        3.0,
        cols,
        rows,
        [os.path.join(images_path, f) for f in sorted(os.listdir(images_path), key=sort_key)],
    )

    # Add all objects to the scene.
    viewer.scene.add(smpl_sequence, billboard, camera)

    # Show a visualization of the camera position and orientation over time.
    camera.show_path()

    # Set initial camera position and target
    viewer.scene.camera.position = np.array((11.0, 4.0, -9))
    viewer.scene.camera.target = np.array((-3.3, 0.6, -7.3))

    # Viewer settings
    viewer.scene.floor.enabled = False
    viewer.scene.fps = 30.0
    viewer.playback_fps = 30.0
    viewer.shadows_enabled = False
    viewer.auto_set_camera_target = False

    viewer.run()

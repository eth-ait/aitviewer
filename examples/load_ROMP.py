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
import cv2
import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # Load camera and SMPL parameters estimated by ROMP https://github.com/Arthur151/ROMP.
    # For a discussion on camera parameters see https://github.com/Arthur151/ROMP/issues/344
    results = np.load("resources/romp/romp_output.npz", allow_pickle=True)["results"][()]

    # Load the image that ROMP produced. Note: in order to get perfect alignment, it is important that ROMP was
    # configured to use pyrender to render the results. Otherwise, the image will be slightly different.
    img_path = "resources/romp/input_pyrender_overlaid.jpg"
    input_img = cv2.imread(img_path)
    cols, rows = input_img.shape[1], input_img.shape[0]

    smpl_layer = SMPLLayer(model_type="smpl", gender="male", device=C.device)
    romp_smpl = SMPLSequence(
        poses_body=results["body_pose"],
        smpl_layer=smpl_layer,
        poses_root=results["global_orient"],
        betas=results["smpl_betas"],
        color=(0.0, 106 / 255, 139 / 255, 1.0),
        name="ROMP Estimate",
        rotation=aa2rot_numpy(np.array([1, 0, 0]) * np.pi),
    )

    # Instantiate the viewer.
    viewer = Viewer(size=(cols, rows))

    # When using pyrender with ROMP, an FOV of 60 degrees is used.
    fov = 60
    f = max(cols, rows) / 2.0 * 1.0 / np.tan(np.radians(fov / 2))
    cam_intrinsics = np.array([[f, 0.0, cols / 2], [0.0, f, rows / 2], [0.0, 0.0, 1.0]])

    # The camera extrinsics are assumed to identity rotation and the translation is estimated by ROMP.
    cam_extrinsics = np.eye(4)
    cam_extrinsics[:3, 3] = results["cam_trans"][0]

    # The OpenCVCamera class expects extrinsics with Y pointing down, so we flip both Y and Z axis to keep a
    # positive determinant.
    cam_extrinsics[1:3, :3] *= -1.0

    # Create an OpenCV camera.
    cameras = OpenCVCamera(cam_intrinsics, cam_extrinsics[:3], cols, rows, viewer=viewer)

    # Load the reference image and create a Billboard.
    pc = Billboard.from_camera_and_distance(cameras, 4.0, cols, rows, [img_path])

    # Add all the objects to the scene.
    viewer.scene.add(pc, romp_smpl, cameras)

    # Set the viewing camera as the camera estimated by ROMP.
    # This is a temporary setting, moving the camera will result in switching back to the default (pinhole) camera.
    viewer.set_temp_camera(cameras)

    # We only render an outline of the SMPL model, so we can see the input image underneath.
    romp_smpl.skeleton_seq.enabled = False
    romp_smpl.mesh_seq.color = romp_smpl.mesh_seq.color[:3] + (0.0,)
    romp_smpl.mesh_seq.draw_outline = True

    # Viewer settings.
    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    viewer.shadows_enabled = False

    viewer.run()

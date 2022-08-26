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
from aitviewer.viewer import Viewer
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.scene.camera import WeakPerspectiveCamera
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.models.smpl import SMPLLayer
from aitviewer.configuration import CONFIG as C
from aitviewer.utils.so3 import aa2rot_numpy

import cv2
import numpy as np


# Use the weak perspective to render the result. If False it will convert it the a proper perspective projection.
USE_WEAK_PERSPECTIVE_CAMERA = True


def estimate_translation_cv2(joints_3d, joints_2d, proj_mat=None, cam_dist=None):
    _, _, tvec, inliers = cv2.solvePnPRansac(joints_3d, joints_2d, proj_mat, cam_dist, flags=cv2.SOLVEPNP_EPNP,
                                             reprojectionError=20, iterationsCount=100)

    if inliers is None:
        raise RuntimeError("No inliers found.")
    else:
        return tvec[:, 0]


if __name__ == '__main__':
    # Load camera and SMPL parameters estimated by ROMP https://github.com/Arthur151/ROMP.
    img_path = "resources/romp/input_overlaid.jpg"
    results = np.load("resources/romp/romp_output.npz", allow_pickle=True)['results'][()]

    input_img = cv2.imread(img_path)
    cols, rows = input_img.shape[1], input_img.shape[0]

    smpl_layer = SMPLLayer(model_type='smpl', gender='male', device=C.device)
    romp_smpl = SMPLSequence(poses_body=results['body_pose'],
                             smpl_layer=smpl_layer,
                             poses_root=results['global_orient'],
                             betas=results['smpl_betas'],
                             color=(0.0, 106 / 255, 139 / 255, 1.0),
                             name='ROMP Estimate',
                             rotation=aa2rot_numpy(np.array([1, 0, 0]) * np.pi) if USE_WEAK_PERSPECTIVE_CAMERA else None)

    # Instantiate the viewer.
    viewer = Viewer(size=(cols, cols if USE_WEAK_PERSPECTIVE_CAMERA else rows))

    # Create the camera.
    if USE_WEAK_PERSPECTIVE_CAMERA:
        def pad_to_square(img):
            h, w = img.shape[0], img.shape[1]
            if h > w:
                diff = h - w
                return cv2.copyMakeBorder(img, 0, 0, diff // 2, diff - diff // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            else:
                diff = w - h
                return cv2.copyMakeBorder(img, diff // 2, diff - diff // 2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # The weak perspective camera was estimated assuming square images, cf.
        # https://github.com/Arthur151/ROMP/issues/344
        cam = results['cam'][0]
        cameras = WeakPerspectiveCamera(cam[[0, 0]], 2 * cam[1:], cols, cols, far=3.0, viewer=viewer)

        pc = Billboard.from_camera_and_distance(cameras, cameras.far - 1e-6, cols, cols, [img_path],
                                                image_process_fn=pad_to_square)
    else:
        # We are using the recommended way to transform the weak perspective camera to a perspective camera
        # as described in this issue: https://github.com/Arthur151/ROMP/issues/300. We are using the camera
        # intrinsics as suggested for the AGORA evaluation setting:
        # https://github.com/Arthur151/ROMP/blob/91dac0172c4dc0685b97f96eda9a3a53c626da47/simple_romp/evaluation/eval_AGORA.py#L84
        # This projection will not perfectly align with the visualization that ROMP provides.
        focal_length = 995.55555556
        input_img = cv2.imread(img_path)
        cam_intrinsics = np.array([[focal_length, 0., cols / 2], [0., focal_length, rows / 2], [0., 0., 1.]])
        cam_extrinsics = np.eye(4)

        # Tranform to perspective projection as suggested in the Github issue.
        tra_pred = estimate_translation_cv2(results['joints'][0], results['pj2d_org'][0], proj_mat=cam_intrinsics)
        cam_extrinsics[:3, 3] = tra_pred

        # Create an OpenCV camera.
        cameras = OpenCVCamera(cam_intrinsics, cam_extrinsics[:3], cols, rows, viewer=viewer)

        # Load the reference image and create a Billboard.
        pc = Billboard.from_camera_and_distance(cameras, 4.0, cols, rows, [img_path])

    # Add all the objects to the scene.
    viewer.scene.add(pc, romp_smpl, cameras)

    # Set the viewing camera as the camera estimated by ROMP.
    # This is a temporary setting, moving the camera will result in switching back to the default (pinhole) camera.
    viewer.set_temp_camera(cameras)

    # Viewer settings.
    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    viewer.shadows_enabled = False

    viewer.run()

# Copyright (C) 2024 Max Planck Institute for Intelligent Systems, Marilyn Keller, marilyn.keller@tuebingen.mpg.de

import argparse
import os

import numpy as np
import torch

from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.skel import SKELSequence
from aitviewer.viewer import Viewer

try:
    from skel.skel_model import SKEL
except Exception as e:
    print("Could not import SKEL, make sure you installed the skel repository.")
    raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a SKEL model and display it.")
    parser.add_argument("-s", "--motion_file", type=str, help="Path to a skel motion file", default=None)
    parser.add_argument("-z", "--z_up", help="Rotate the mesh 90 deg", action="store_true")

    args = parser.parse_args()

    skel_model = SKEL(gender="female", model_path=C.skel_models).to(C.device)

    if args.motion_file is None:
        F = 120
        pose = torch.zeros(F, 46).to(C.device)
        betas = torch.zeros(F, 10).to(C.device)
        betas[: F // 2, 0] = torch.linspace(-2, 2, F // 2)  # Vary beta0 between -2 and 2
        betas[F // 2 :, 1] = torch.linspace(-2, 2, F // 2)  # Vary beta1 between -2 and 2

        trans = torch.zeros(F, 3).to(C.device)

        # Test SKEL forward pass
        skel_output = skel_model(pose, betas, trans)

        skel_seq = SKELSequence(
            skel_layer=skel_model,
            betas=betas,
            poses_body=pose,
            poses_type="skel",
            trans=trans,
            is_rigged=True,
            show_joint_angles=True,
            name="SKEL",
            z_up=False,
            skin_coloring="skinning_weights",  # "pose_offsets", "skinning_weights"
        )
        cam_pose = None

    else:
        assert os.path.exists(
            args.motion_file
        ), f"Could not find {args.motion_file}, please provide a valid path to a skel motion file."
        assert args.motion_file.endswith(".pkl"), f"Please provide a .pkl file."

        skel_seq = SKELSequence.from_pkl(args.motion_file, name="SKEL", fps_in=120, fps_out=30, z_up=args.z_up)
        cam_pose = np.array([0, 1.2, 4.0])

    v = Viewer()
    v.playback_fps = 30
    v.scene.add(skel_seq)
    v.run_animations = True
    if cam_pose is not None:
        v.scene.camera.position = cam_pose
    v.run()

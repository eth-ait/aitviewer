# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import pickle as pkl

import torch

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # This is loading the TotalCapture data that can be downloaded from the DIP project website here:
    # https://dip.is.tue.mpg.de/download.php
    # Download the "Reference SMPL pose parameters for TotalCapture dataset" and point the following path to
    # one of the extracted pickle files.
    with open("./s1_acting1.pkl", "rb") as f:
        data = pkl.load(f, encoding="latin1")

    oris = data["ori"]
    poses = data["gt"]

    # Sometimes the poses and raw measurements are off by one frame due to how the data was preprocessed,
    # so correct for this.
    if oris.shape[0] < poses.shape[0]:
        pose = poses[: oris.shape[0]]
    elif oris.shape[0] > poses.shape[0]:
        acc = oris[: poses.shape[0]]

    # Display only the first `max_frames` frames.
    max_frames = 1000
    poses = poses[:max_frames]
    oris = oris[:max_frames]

    # DIP assumed the mean pose and the male SMPL model.
    betas = torch.zeros((poses.shape[0], 10)).float().to(C.device)
    smpl_layer = SMPLLayer(model_type="smpl", gender="male", device=C.device)

    # We need to anchor the IMU orientations somewhere in order to display them.
    # We can do this at the joint locations, so perform one forward pass.
    _, joints = smpl_layer(
        poses_body=torch.from_numpy(poses[:, 3:]).float().to(C.device),
        poses_root=torch.from_numpy(poses[:, :3]).float().to(C.device),
        betas=betas,
    )

    # Place the orientations at the joint positions.
    # The order of sensors is left arm, right arm, left leg, right arm, head, pelvis.
    # This corresponds to the following SMPL joint indices.
    joint_idxs = [18, 19, 4, 5, 15, 0]
    rbs = RigidBodies(joints[:, joint_idxs].cpu().numpy(), oris)

    # Display the SMPL ground-truth.
    smpl_seq = SMPLSequence(poses_body=poses[:, 3:], smpl_layer=smpl_layer, poses_root=poses[:, :3])

    # Add everything to the scene and display.
    v = Viewer()
    v.scene.add(smpl_seq, rbs)
    v.run()

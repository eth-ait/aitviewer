import pickle as pkl

import torch

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # This is loading the DIP-IMU data that can be downloaded from the DIP project website here:
    # https://dip.is.tue.mpg.de/download.php
    # Download the "DIP IMU and others" and point the following path to one of the extracted pickle files.
    with open(r"C:\Users\manuel\Downloads\DIPIMUandOthers\DIP_IMU_and_Others\DIP_IMU\DIP_IMU\s_02\04.pkl", "rb") as f:
        data = pkl.load(f, encoding="latin1")

    # Whether we want to visualize all 17 sensors or just the 6 sensors used by DIP.
    all_sensors = True

    # Get the data.
    oris = data["imu_ori"]
    poses = data["gt"]

    # Subject 6 is female, all others are male (cf. metadata.txt included in the downloaded zip file).
    gender = "male"

    # Downsample to 30 Hz.
    poses = poses[::2]
    oris = oris[::2]

    # DIP has no shape information, assume the mean shape.
    betas = torch.zeros((poses.shape[0], 10)).float().to(C.device)
    smpl_layer = SMPLLayer(model_type="smpl", gender=gender, device=C.device)

    # We need to anchor the IMU orientations somewhere in order to display them.
    # We can do this at the joint locations, so perform one forward pass.
    _, joints = smpl_layer(
        poses_body=torch.from_numpy(poses[:, 3:]).float().to(C.device),
        poses_root=torch.from_numpy(poses[:, :3]).float().to(C.device),
        betas=betas,
    )

    # This is the sensor placement (cf. https://github.com/eth-ait/dip18/issues/16).
    sensor_placement = [
        "head",
        "sternum",
        "pelvis",
        "lshoulder",
        "rshoulder",
        "lupperarm",
        "rupperarm",
        "llowerarm",
        "rlowerarm",
        "lupperleg",
        "rupperleg",
        "llowerleg",
        "rlowerleg",
        "lhand",
        "rhand",
        "lfoot",
        "rfoot",
    ]

    # We manually choose the SMPL joint indices cooresponding to the above sensor placement.
    joint_idxs = [15, 12, 0, 13, 14, 16, 17, 20, 21, 1, 2, 4, 5, 22, 23, 10, 11]

    # Select only the 6 input sensors if configured.
    sensor_sub_idxs = [7, 8, 11, 12, 0, 2] if not all_sensors else list(range(len(joint_idxs)))
    rbs = RigidBodies(joints[:, joint_idxs][:, sensor_sub_idxs].cpu().numpy(), oris[:, sensor_sub_idxs])

    # Display the SMPL ground-truth with a semi-transparent mesh so we can see the IMUs.
    smpl_seq = SMPLSequence(poses_body=poses[:, 3:], smpl_layer=smpl_layer, poses_root=poses[:, :3])
    smpl_seq.mesh_seq.color = smpl_seq.mesh_seq.color[:3] + (0.5,)

    # Add everything to the scene and display at 30 fps.
    v = Viewer()
    v.playback_fps = 30.0

    v.scene.add(smpl_seq, rbs)
    v.run()

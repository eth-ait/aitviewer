import os
from pathlib import Path
from random import random, shuffle

import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.remote.renderables.meshes import RemoteMeshes
from aitviewer.remote.renderables.smpl import RemoteSMPLSequence
from aitviewer.remote.viewer import RemoteViewer
from aitviewer.renderables.smpl import SMPLSequence

# Select subset of AMASS dataset to load.
a_seqs = list(Path(os.path.join(C.datasets.amass, "Dfaust")).rglob("*poses.npz"))
shuffle(a_seqs)

# Create (NxN) Grid.
N = 5
x = np.linspace(-5, 5, N)
z = np.linspace(-5, 5, N)
xv, zv = np.meshgrid(x, z)
xv = xv.reshape(N * N)
zv = zv.reshape(N * N)
yv = np.zeros(xv.shape[0]) + 0.6
positions = np.vstack((xv, yv, zv)).T

# Create Remote Viewer.
v: RemoteViewer = RemoteViewer.create_new_process()

for pos, seq in zip(positions, a_seqs):
    local_smpl = SMPLSequence.from_amass(npz_data_path=seq, fps_out=60.0, end_frame=200, log=False)
    seq_name = os.path.splitext(os.path.basename(seq))[0]

    # Send to Remote Viewer via vertices/faces.
    print("Sending sequence {} to Remote Viewer".format(seq_name))
    remote_meshes = RemoteMeshes(
        viewer=v,
        vertices=local_smpl.vertices,
        faces=local_smpl.faces,
        rotation=np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
        position=pos,
        color=(random(), random(), random(), 1.0),
        name=seq_name,
    )

    # [Alternatively] Send to Remote Viewer via SMPL poses and betas (this will trigger a model inference on the viewer).
    # remote_smpl = RemoteSMPLSequence(
    #     viewer=v,
    #     poses_body=local_smpl.poses_body,
    #     poses_root=local_smpl.poses_root,
    #     poses_left_hand=local_smpl.poses_left_hand,
    #     poses_right_hand=local_smpl.poses_right_hand,
    #     betas=local_smpl.betas,
    #     trans=local_smpl.trans,
    #     model_type=local_smpl.smpl_layer.model_type,
    #     z_up=True
    #     )

    v.set_frame(0)

# Cleanup.
v.close_connection()

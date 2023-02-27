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
import joblib
import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # In this example we show a way to play sequences of data that has missing frames.
    # Here we assume that we have data only for the frames that exist but we also know the indices
    # of missing frames and we want to avoid padding our data with invalid frames.
    # Instead, we create a mask that has ones for each frame for which we have data and zeros
    # everywhere else. Thus the number of ones in the mask must match the number of frames in the data.

    # Load SMPL data from the output of the VIBE demo from https://github.com/mkocabas/VIBE
    data = joblib.load(open("resources/vibe/vibe_output.pkl", "rb"))
    poses = data[1]["pose"]
    betas = data[1]["betas"]

    # Number of total frames, including frames that are missing.
    N = poses.shape[0]

    # Number of existing frames of data. We are going to remove 100 frames to create holes in our data.
    M = N - 100

    # Create a hole of 100 frames in the data, we now have M frames of data.
    # In this example we suppose that we have 25 frames of data followed by a hole of 100 frames.
    p = np.concatenate((poses[:25], poses[125:]))
    b = np.concatenate((betas[:25], betas[125:]))

    # Create a mask with N frames.
    enabled_frames = np.ones(N, dtype=np.bool8)
    # Set to zero the values at the indices of the 100 missing frames. There are now only M ones in the mask.
    enabled_frames[25:125] = 0

    # Instantiate an SMPL sequence using the parameters and the enabled_frames mask.
    smpl_layer = SMPLLayer(model_type="smpl", gender="neutral", device=C.device)
    smpl_seq = SMPLSequence(
        poses_body=p[:, 3 : 24 * 3],
        poses_root=p[:, 0:3],
        betas=b,
        smpl_layer=smpl_layer,
        position=(1, 0, 0),
        rotation=aa2rot_numpy(np.array([1, 0, 0]) * np.pi),
        enabled_frames=enabled_frames,
    )

    # We do the same for a second sequence but removing a different range of frames.
    # Notice how the two sequences remain in sync, since the sequence is only advanced
    # during frames that are enabled (where the mask value is one).
    p2 = np.concatenate((poses[:175], poses[275:]))
    b2 = np.concatenate((betas[:175], betas[275:]))
    enabled_frames2 = np.ones(N, dtype=np.bool8)
    enabled_frames2[175:275] = 0
    smpl_layer2 = SMPLLayer(model_type="smpl", gender="neutral", device=C.device)
    smpl_seq2 = SMPLSequence(
        poses_body=p2[:, 3 : 24 * 3],
        poses_root=p2[:, 0:3],
        betas=b2,
        smpl_layer=smpl_layer2,
        position=(-1, 0, 0),
        rotation=aa2rot_numpy(np.array([1, 0, 0]) * np.pi),
        enabled_frames=enabled_frames2,
    )

    # Run the viewer
    viewer = Viewer(size=(1600, 900))
    viewer.scene.floor.enabled = False
    viewer.scene.camera.position = (0, 1, 5)
    viewer.scene.fps = 25
    viewer.playback_fps = 25
    viewer.scene.add(smpl_seq, smpl_seq2)
    viewer.run()

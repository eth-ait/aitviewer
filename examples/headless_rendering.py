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

import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.smpl import SMPLSequence

if __name__ == "__main__":
    # Load an AMASS sequence.
    smpl_seq = SMPLSequence.from_amass(
        npz_data_path=os.path.join(C.datasets.amass, "ACCAD/Female1Running_c3d/C2 - Run to stand_poses.npz"),
        fps_out=60.0,
        name="AMASS Running",
        show_joint_angles=True,
    )
    smpl_seq.color = smpl_seq.color[:3] + (0.75,)  # Make the sequence a bit transparent.

    # Create the headless renderer and add the sequence.
    v = HeadlessRenderer()
    v.scene.add(smpl_seq)

    # Have the camera automatically follow the SMPL sequence. For every frame, the camera points to the center of the
    # bounding box of the SMPL mesh while keeping a fixed relative distance. The smoothing is optional but ensures that
    # the view is not too jittery.
    v.lock_to_node(smpl_seq, (2, 2, 2), smooth_sigma=5.0)
    v.save_video(video_dir=os.path.join(C.export_dir, "headless/test.mp4"))

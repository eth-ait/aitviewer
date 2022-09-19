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
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.headless import HeadlessRenderer


if __name__ == '__main__':
    # Load an AMASS sequence.
    smpl_seq = SMPLSequence.from_amass(npz_data_path=os.path.join(C.datasets.amass, 'TotalCapture/s2/rom3_poses.npz'),
                                       start_frame=0, end_frame=500, fps_out=60.0)

    # Render to video.
    v = HeadlessRenderer()
    v.scene.add(smpl_seq)
    v.scene.camera.position = np.array([0.0, 0.5, 4.0])
    v.save_video(video_dir=os.path.join(C.export_dir, 'headless/test.mp4'))

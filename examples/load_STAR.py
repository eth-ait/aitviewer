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

from aitviewer.configuration import CONFIG as C
from aitviewer.models.star import STARLayer
from aitviewer.renderables.star import STARSequence
from aitviewer.viewer import Viewer


if __name__ == '__main__':
    star_layer = STARLayer(device=C.device)
    star_template = STARSequence.t_pose(model=star_layer)

    star_seq = STARSequence.from_amass(
        npz_data_path=os.path.join(C.datasets.amass, "ACCAD/Female1Running_c3d/C2 - Run to stand_poses.npz"),
        fps_out=60.0, name="AMASS Running", show_joint_angles=False)

    # Add to scene and render
    v = Viewer()
    v.scene.add(star_template, star_seq)
    v.run()

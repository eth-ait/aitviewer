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
import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # Create a neutral SMPL T Pose.
    # This also works with `smplh` or `smplx` model type (but there's no neutral model for SMPL-H).
    smplh_template = SMPLSequence.t_pose(SMPLLayer(model_type="smplh", gender="neutral", device=C.device), name="SMPL")
    mano_template = SMPLSequence.t_pose(
        SMPLLayer(model_type="mano", gender="neutral", device=C.device),
        position=np.array((-1.0, 0.0, 0.0)),
        name="MANO",
    )
    flame_template = SMPLSequence.t_pose(
        SMPLLayer(model_type="flame", gender="neutral", device=C.device),
        position=np.array((1.0, 0.0, 0.0)),
        name="FLAME",
    )

    # Display in viewer.
    v = Viewer()
    v.scene.add(smplh_template, mano_template, flame_template)
    v.run()

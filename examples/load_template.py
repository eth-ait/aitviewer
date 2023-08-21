# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
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

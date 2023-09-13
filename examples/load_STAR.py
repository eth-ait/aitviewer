# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import os

from aitviewer.configuration import CONFIG as C
from aitviewer.models.star import STARLayer
from aitviewer.renderables.star import STARSequence
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # Instantiate a STAR layer. This requires that the respective repo has been installed via
    # pip install git+https://github.com/ahmedosman/STAR.git and that the model files are available on the path
    # specified in `C.star_models`.
    star_layer = STARLayer(device=C.device)

    # Load a STAR Sequence from AMASS data. BETAs will not be loaded by default and need to be converted.
    star_seq = STARSequence.from_amass(
        npz_data_path=os.path.join(C.datasets.amass, "ACCAD/Female1Running_c3d/C2 - Run to stand_poses.npz"),
        fps_out=60.0,
        name="AMASS Running",
        show_joint_angles=False,
    )

    # Add to scene and render
    v = Viewer()
    v.scene.add(star_seq)
    v.run()

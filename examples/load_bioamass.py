# Copyright (C) 2024 Max Planck Institute for Intelligent Systems, Marilyn Keller, marilyn.keller@tuebingen.mpg.de

import os

import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.osim import OSIMSequence
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    subj_name = "01"
    seq_name = "03"

    c = (149 / 255, 85 / 255, 149 / 255, 0.5)

    to_display = []

    amass_file = os.path.join(C.datasets.amass, f"CMU/{subj_name}/{subj_name}_{seq_name}_poses.npz")
    osim_file = os.path.join(C.datasets.bioamass, f"CMU/{subj_name}/ab_fits/Models/optimized_scale_and_markers.osim")
    mot_file = os.path.join(C.datasets.bioamass, f"CMU/{subj_name}/ab_fits/IK/{seq_name}_ik.mot")

    if os.path.exists(C.datasets.amass):
        seq_amass = SMPLSequence.from_amass(
            npz_data_path=amass_file,
            fps_out=30.0,
            color=c,
            name=f"AMASS {subj_name} {seq_name}",
            show_joint_angles=False,
        )
        to_display.append(seq_amass)
    else:
        seq_amass = None
        print(f"Could not find AMASS dataset at {C.datasets.amass}. Skipping loading SMPL body.")

    osim_seq = OSIMSequence.from_files(
        osim_path=osim_file,
        mot_file=mot_file,
        name=f"BSM {subj_name} {seq_name}",
        fps_out=30,
        color_skeleton_per_part=True,
        show_joint_angles=False,
        is_rigged=False,
    )

    to_display.append(osim_seq)

    # Display in the viewer
    v = Viewer()
    v.run_animations = True
    v.scene.camera.position = np.array([10.0, 2.5, 0.0])
    v.scene.add(*to_display)

    if seq_amass is not None:
        v.lock_to_node(seq_amass, (2, 0.7, 2), smooth_sigma=5.0)
    v.playback_fps = 30

    v.run()

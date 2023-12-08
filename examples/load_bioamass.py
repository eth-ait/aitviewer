# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import os
from aitviewer.renderables.osim import OSIMSequence

import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # Load an AMASS sequence and make sure it's sampled at 60 fps. This automatically loads the SMPL-H model.
    # We set transparency to 0.5 and render the joint coordinates systems.
    subj_name = '01'
    seq_name = '03'
    
    c = (149 / 255, 85 / 255, 149 / 255, 0.5)
    
    seq_amass = SMPLSequence.from_amass(
        npz_data_path=os.path.join(C.datasets.amass, f"CMU/{subj_name}/{subj_name}_{seq_name}_poses.npz"), # AMASS/CMU/01/01_01_poses.npz
        fps_out=30.0,
        color=c,
        name=f"AMASS {subj_name} {seq_name}",
        show_joint_angles=False,
    )

    osim_path = os.path.join(C.datasets.bioamass, f"CMU/{subj_name}/ab_fits/Models/optimized_scale_and_markers.osim") # bioamass_v1.0/CMU/11/ab_fits/Models/optimized_scale_and_markers.osim
    mot_file = os.path.join(C.datasets.bioamass, f"CMU/{subj_name}/ab_fits/IK/{seq_name}_ik.mot") #bioamass_v1.0/CMU/11/ab_fits/IK/01_ik.mot

    osim_seq = OSIMSequence.from_files(osim_path=osim_path, 
                                       mot_file=mot_file, 
                                       name=f'BSM {subj_name} {seq_name}', 
                                       fps_out=30,
                                       color_skeleton_per_part=True, 
                                    show_joint_angles=True, is_rigged=True)
    
    # Display in the viewer.
    v = Viewer()
    v.run_animations = True
    v.scene.camera.position = np.array([10.0, 2.5, 0.0])
    v.scene.add(seq_amass, osim_seq)
    
    v.lock_to_node(seq_amass, (2, 0.7, 2), smooth_sigma=5.0)
    v.playback_fps = 30
    
    v.run()

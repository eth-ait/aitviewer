"""Visualize amass with mocap markers"""

import numpy as np
import os

from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.viewer import Viewer
from aitviewer.configuration import CONFIG as C

from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.markers import Markers


if __name__ == '__main__':

    # Display in the viewer.
    v = Viewer()

    fps_in = 120 #TODO load this from file
    fps_out = 60

    seq_subj = '01'
    seq_trial = '03'

    seq_path = f'CMU/{seq_subj}/{seq_subj}_{seq_trial}_poses.npz'
    c3d_file = f"../AMASS_mocap/CMU/subjects/{seq_subj}/{seq_subj}_{seq_trial}.c3d"
    c3d_file_path = os.path.join(C.datasets.amass, c3d_file)

    c = (85/255, 85/255, 255/255, 1)
    markers_pc = Markers.from_c3d(c3d_file_path, color=c, fps_out=fps_out, point_size=15, nb_markers_expected=41)

    # Amass sequence
    c = (149/255, 150/255, 0/255, 0.9)
    seq_amass = SMPLSequence.from_amass(
        npz_data_path=os.path.join(C.datasets.amass, seq_path),
        fps_out=fps_out, color=c, name="AMASS Running", show_joint_angles=False, is_rigged=False, device='cpu')

    v.run_animations = True
    v.scene.camera.position = np.array([10.0, 2.5, 0.0])
    
    v.scene.add(seq_amass)
    v.scene.add(markers_pc)
    
    v.run()

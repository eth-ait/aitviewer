# Copyright (C) 2023  MPI, Marilyn Keller

import torch
from aitviewer.viewer import Viewer

try:
    from skel.skel_model import SKEL
except:
    print("Could not import SKEL, make sure you installed it.")

if __name__ == "__main__":
    
    skel_model = SKEL()

    pose = torch.rand(1, 49)
    betas = torch.rand(1, 10)
    trans =  torch.rand(1, 3)

    skel_output = skel_model(pose, betas, None)
    
    from aitviewer.renderables.skel import SKELSequence
    skel_seq = SKELSequence(skel_layer=skel_model, betas=betas, poses_body=pose, poses_type='skel', 
                            trans=trans, is_rigged=True, show_joint_angles=True, name='SKEL', z_up=False,
                            skinning_weights_color=False,
                            # position=[0,-0.94,0],
                            )
    
    v = Viewer()
    v.scene.add(SKELSequence.t_pose())
    v.run()

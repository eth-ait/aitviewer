# Copyright (C) 2023  MPI, Marilyn Keller

import torch
from aitviewer.viewer import Viewer
from aitviewer.renderables.skel import SKELSequence

try:
    from skel.skel_model import SKEL
except:
    print("Could not import SKEL, make sure you installed the skel repository.")

if __name__ == "__main__":
    
    skel_model = SKEL()

    F = 120
    pose = torch.zeros(F, 46)
    betas = torch.zeros(F, 10)
    betas[:F//2, 0] = torch.linspace(-2, 2, F//2) # Vary beta0 between -2 and 2
    betas[F//2:, 1] = torch.linspace(-2, 2, F//2) # Vary beta1 between -2 and 2
    
    trans =  torch.zeros(F, 3)

    # Test SKEL forward pass
    skel_output = skel_model(pose, betas, trans)

    skel_seq = SKELSequence(skel_layer=skel_model, betas=betas, poses_body=pose, poses_type='skel', 
                            trans=trans, is_rigged=True, show_joint_angles=True, name='SKEL', z_up=False,
                            skinning_weights_color=False,
                            # position=[0,-0.94,0],
                            )
    
    v = Viewer()
    v.playback_fps = 30
    # v.scene.add(SKELSequence.t_pose(skel_layer=skel_model))
    v.scene.add(skel_seq)
    v.run()

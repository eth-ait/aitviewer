"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See https://skel.is.tue.mpg.de/license.html for licensing and contact information.
"""
import torch
from aitviewer.viewer import Viewer
from aitviewer.renderables.skel import SKELSequence
from aitviewer.configuration import CONFIG as C

try:
    from skel.skel_model import SKEL
except Exception as e:
    print("Could not import SKEL, make sure you installed the skel repository.")
    raise e

if __name__ == "__main__":
    
    skel_model = SKEL(model_path=C.skel_models)

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
                            )
    
    v = Viewer()
    v.playback_fps = 30
    v.scene.add(skel_seq)
    v.run()

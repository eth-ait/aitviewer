from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.utils import to_numpy as c2c
import torch
import numpy as np
from aitviewer.models.star import STARLayer

class STARSequence(SMPLSequence):
    """
    Represents a temporal sequence of SMPL poses using the STAR model.
    """

    def __init__(self,
                 poses_body,
                 smpl_layer,
                 poses_root,
                 betas=None,
                 trans=None,
                 device=C.device,
                 include_root=True,
                 normalize_root=False,
                 is_rigged=True,
                 in_canonical=False,
                 show_joint_angles=False,
                 z_up=False,
                 post_fk_func=None,
                 **kwargs):


        super(STARSequence, self).__init__(poses_body, smpl_layer, poses_root, betas, trans, device=device,
                                           include_root=include_root, normalize_root=normalize_root,
                                           is_rigged=is_rigged, show_joint_angles=show_joint_angles, z_up=z_up,
                                           post_fk_func=post_fk_func,
                                            **kwargs)


    def fk(self, current_frame_only=False):
        """Get joints and/or vertices from the poses_body."""
        poses_root = self.poses_root if self._include_root else None
        trans = self.trans if self._include_root else None
        verts, joints = self.smpl_layer(poses_root=poses_root,
                                        poses_body=self.poses_body,
                                        betas=self.betas,
                                        trans=trans,
                                        normalize_root=self._normalize_root
                                        )
        if self._z_up:
            to_y_up = torch.Tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).to(device=self.betas.device, dtype=self.betas.dtype)
            verts = torch.matmul(to_y_up.unsqueeze(0), verts.unsqueeze(-1)).squeeze(-1)
            joints = torch.matmul(to_y_up.unsqueeze(0), joints.unsqueeze(-1)).squeeze(-1)

        skeleton = self.smpl_layer.skeletons().T
        faces = self.smpl_layer.faces

        if current_frame_only:
            return c2c(verts)[0], c2c(joints)[0], c2c(faces), c2c(skeleton)
        else:
            return c2c(verts), c2c(joints), c2c(faces), c2c(skeleton)


    @classmethod
    def from_amass(cls, npz_data_path, start_frame=None, end_frame=None, sub_frames=None, log=True, fps_out=None, load_betas=True, z_up=True, rest_in_a=False, **kwargs):
        seq = SMPLSequence.from_amass(npz_data_path, start_frame, end_frame, log, fps_out, **kwargs)

        # STAR has no hands, but includes wrists
        poses_body = torch.cat((seq.poses_body, seq.poses_left_hand[:, :3], seq.poses_right_hand[:, :3]), dim=-1)
        poses_root = seq.poses_root
        trans = seq.trans
        betas = seq.betas if load_betas else None

        if sub_frames is not None:
            poses_root = poses_root[sub_frames]
            poses_body = poses_body[sub_frames]
            trans = trans [sub_frames]

        return cls(poses_body=poses_body,
                   smpl_layer=STARLayer(device=C.device),
                   poses_root=poses_root,
                   betas=betas,
                   trans=trans,
                   include_root=seq._include_root,
                   is_rigged=seq._is_rigged,
                   z_up=z_up,
                   color=seq.color,
                   **kwargs)

    @classmethod
    def from_3dpw(cls):
        raise ValueError('STAR does not support loading from 3DPW.')

    @classmethod
    def t_pose(cls, model=None, betas=None, frames=1, **kwargs):
        """Creates a SMPL sequence whose single frame is a SMPL mesh in T-Pose."""

        if model is None:
            model = STARLayer(device=C.device)

        poses_body = np.zeros([frames, model.n_joints_body * 3])
        poses_root = np.zeros([frames, 3])
        return cls(poses_body=poses_body, smpl_layer=model, poses_root=poses_root, betas=betas, **kwargs)



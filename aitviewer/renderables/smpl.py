"""
Copyright (C) 2022  ETH Zurich, Manuel Kaufmann, Velko Vechev

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
import pickle as pkl
import torch

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.scene.node import Node
from aitviewer.utils.so3 import aa2rot_torch as aa2rot
from aitviewer.utils.so3 import rot2aa_torch as rot2aa
from aitviewer.utils.so3 import interpolate_rotations
from aitviewer.utils.so3 import resample_rotations
from aitviewer.utils import resample_positions
from aitviewer.utils import to_torch
from aitviewer.utils import local_to_global
from aitviewer.utils import interpolate_positions
from aitviewer.utils import to_numpy as c2c
from scipy.spatial.transform import Rotation as R


class SMPLSequence(Node):
    """
    Represents a temporal sequence of SMPL poses. Can be loaded from disk or initialized from memory.
    """

    def __init__(self,
                 poses_body,
                 smpl_layer,
                 poses_root=None,
                 betas=None,
                 trans=None,
                 poses_left_hand=None,
                 poses_right_hand=None,
                 device=C.device,
                 dtype=C.f_precision,
                 include_root=True,
                 normalize_root=False,
                 is_rigged=True,
                 show_joint_angles=False,
                 **kwargs):
        """
        Initializer.
        :param poses_body: An array (numpy ar pytorch) of shape (F, N_JOINTS*3) containing the pose parameters of the
          body, i.e. without hands or face parameters.
        :param smpl_layer: The SMPL layer that maps parameters to joint positions and/or dense surfaces.
        :param poses_root: An array (numpy or pytorch) of shape (F, 3) containing the global root orientation.
        :param betas: An array (numpy or pytorch) of shape (N_BETAS, ) containing the shape parameters.
        :param trans: An array (numpy or pytorch) of shape (F, 3) containing the global root translation.
        :param device: The pytorch device for computations.
        :param dtype: The pytorch data type.
        :param include_root: Whether or not to include root information. If False, no root translation and no root
          rotation is applied.
        :param normalize_root: Whether or not to normalize the root. If True, the global root translation in the first
          frame is zero and the global root orientation is the identity.
        :param is_rigged: Whether or not to display the joints as a skeleton.
        :param show_joint_angles: Whether or not the coordinate frames at the joints should be visualized.
        :param kwargs: Remaining arguments for rendering.
        """
        assert len(poses_body.shape) == 2
        super(SMPLSequence, self).__init__(n_frames=poses_body.shape[0], **kwargs)

        self.smpl_layer = smpl_layer

        self.poses_body = to_torch(poses_body, dtype=dtype, device=device)
        self.poses_left_hand = to_torch(poses_left_hand, dtype=dtype, device=device)
        self.poses_right_hand = to_torch(poses_right_hand, dtype=dtype, device=device)

        poses_root = poses_root if poses_root is not None else torch.zeros([len(poses_body), 3])
        betas = betas if betas is not None else torch.zeros([1, self.smpl_layer.num_betas])
        trans = trans if trans is not None else torch.zeros([len(poses_body), 3])

        self.poses_root = to_torch(poses_root, dtype=dtype, device=device)
        self.betas = to_torch(betas, dtype=dtype, device=device)
        self.trans = to_torch(trans, dtype=dtype, device=device)

        if len(self.betas.shape) == 1:
            self.betas = self.betas.unsqueeze(0)

        self._include_root = include_root
        self._normalize_root = normalize_root
        self._show_joint_angles = show_joint_angles
        self._is_rigged = is_rigged or show_joint_angles
        self._render_kwargs = kwargs

        if not self._include_root:
            self.poses_root = torch.zeros_like(self.poses_root)
            self.trans = torch.zeros_like(self.trans)

        if self._normalize_root:
            root_ori = aa2rot(self.poses_root)
            first_root_ori = torch.inverse(root_ori[0:1])
            root_ori = torch.matmul(first_root_ori, root_ori)
            self.poses_root = rot2aa(root_ori)

            trans = torch.matmul(first_root_ori.unsqueeze(0), self.trans.unsqueeze(-1)).squeeze()
            self.trans = trans - trans[0:1]

        # Nodes
        self.vertices, self.joints, self.faces, self.skeleton = self.fk()

        if self._is_rigged:
            # Must first add skeleton, otherwise transparency does not work correctly.
            # Overriding given color with a custom color for the skeleton.
            kwargs = self._render_kwargs.copy()
            kwargs['color'] = (1.0, 177 / 255, 1 / 255, 1.0)
            kwargs['name'] = 'Skeleton'
            self.skeleton_seq = Skeletons(self.joints, self.skeleton, **kwargs)
            self.skeleton_seq.position = self.position
            self.skeleton_seq.rotation = self.rotation

            self._add_node(self.skeleton_seq, gui_elements=['material'])

        if self._show_joint_angles:
            # First convert the relative joint angles to global joint angles in rotation matrix form.
            global_oris = local_to_global(torch.cat([self.poses_root, self.poses_body], dim=-1),
                                          self.skeleton[:, 0], output_format='rotmat')
            global_oris = global_oris.reshape((self.n_frames, -1, 3, 3))

            self.rbs = RigidBodies(self.joints, c2c(global_oris), length=0.1, name='Joint Angles')
            self.rbs.position = self.position
            self.rbs.rotation = self.rotation
            self._add_node(self.rbs)

        kwargs = self._render_kwargs.copy()
        kwargs['name'] = 'Mesh'
        kwargs['color'] = kwargs.get('color', (160 / 255, 160 / 255, 160 / 255, 1.0))
        self.mesh_seq = Meshes(self.vertices, self.faces, **kwargs)
        self.mesh_seq.position = self.position
        self.mesh_seq.rotation = self.rotation
        self._add_node(self.mesh_seq, gui_elements=['material'])

    @classmethod
    def from_amass(cls, npz_data_path, start_frame=None, end_frame=None, log=True, fps_out=None, **kwargs):
        """Load a sequence downloaded from the AMASS website."""
        body_data = np.load(npz_data_path)
        smpl_layer = SMPLLayer(model_type='smplh', gender=body_data['gender'].item(), device=C.device)

        if log:
            print('Data keys available: {}'.format(list(body_data.keys())))
            print('{:>6d} poses of size {:>4d}.'.format(body_data['poses'].shape[0], body_data['poses'].shape[1]))
            print('{:>6d} trans of size {:>4d}.'.format(body_data['trans'].shape[0], body_data['trans'].shape[1]))
            print('{:>6d} shape of size {:>4d}.'.format(1, body_data['betas'].shape[0]))
            print('Gender {}'.format(body_data['gender']))
            print('FPS {}'.format(body_data['mocap_framerate']))

        sf = start_frame or 0
        ef = end_frame or body_data['poses'].shape[0]
        poses = body_data['poses'][sf:ef]
        trans = body_data['trans'][sf:ef]

        if fps_out is not None:
            fps_in = body_data['mocap_framerate'].tolist()
            if fps_in != fps_out:
                ps = np.reshape(poses, [poses.shape[0], -1, 3])
                ps_new = resample_rotations(ps, fps_in, fps_out)
                poses = np.reshape(ps_new, [-1, poses.shape[1]])
                trans = resample_positions(trans, fps_in, fps_out)

        # Transform root orientation and translation into our viewer's coordinate system (where Y is up).
        to_y_up = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        poses_root = R.as_matrix(R.from_rotvec(poses[:, 0:3]))
        poses_root = np.matmul(to_y_up[np.newaxis], poses_root)
        poses[:, 0:3] = R.as_rotvec(R.from_matrix(poses_root))
        trans = np.matmul(to_y_up[np.newaxis], trans[..., np.newaxis]).squeeze()
        i_root_end = 3
        i_body_end = i_root_end + smpl_layer.bm.NUM_BODY_JOINTS*3
        i_left_hand_end = i_body_end + smpl_layer.bm.NUM_HAND_JOINTS*3
        i_right_hand_end = i_left_hand_end + smpl_layer.bm.NUM_HAND_JOINTS*3

        return cls(poses_body=poses[:, i_root_end:i_body_end],
                   poses_root=poses[:, :i_root_end],
                   poses_left_hand=poses[:, i_body_end: i_left_hand_end],
                   poses_right_hand=poses[:, i_left_hand_end: i_right_hand_end],
                   smpl_layer=smpl_layer,
                   betas=body_data['betas'][np.newaxis],
                   trans=trans, **kwargs)

    @classmethod
    def from_3dpw(cls, pkl_data_path, smplx_neutral=False, **kwargs):
        """Load a 3DPW sequence which might contain multiple people."""
        with open(pkl_data_path, 'rb') as p:
            body_data = pkl.load(p, encoding='latin1')
        num_people = len(body_data['poses'])

        if smplx_neutral:
            smpl_layer = SMPLLayer(model_type='smplx', gender='neutral', num_betas=10, flat_hand_mean=True)

        poses_key = 'poses_smplx' if smplx_neutral else 'poses'
        trans_key = 'trans_smplx' if smplx_neutral else 'trans'
        betas_key = 'betas_smplx' if smplx_neutral else 'betas'

        name = kwargs.get('name', '3DPW')

        seqs = []
        for i in range(num_people):
            gender = body_data['genders'][i]
            if not smplx_neutral:
                smpl_layer = SMPLLayer(model_type='smpl', gender='female' if gender == 'f' else 'male', device=C.device,
                                       num_betas=10)

            # Extract the 30 Hz data that is already aligned with the image data.
            poses = body_data[poses_key][i]
            trans = body_data[trans_key][i]
            betas = body_data[betas_key][i]

            if len(betas.shape) == 1:
                betas = betas[np.newaxis]

            poses_body = poses[:, 3:]
            poses_root = poses[:, :3]
            trans_root = trans

            kwargs['name'] = name + " S{}".format(i)
            seq = cls(poses_body=poses_body, poses_root=poses_root, trans=trans_root,
                      smpl_layer=smpl_layer, betas=betas, **kwargs)
            seqs.append(seq)

        return seqs

    @classmethod
    def t_pose(cls, smpl_layer=None, betas=None, frames=1, **kwargs):
        """Creates a SMPL sequence whose single frame is a SMPL mesh in T-Pose."""

        if smpl_layer is None:
            smpl_layer = SMPLLayer(model_type='smplh', gender='neutral')

        poses = np.zeros([frames, smpl_layer.bm.NUM_BODY_JOINTS * 3])  # including hands and global root
        return cls(poses, smpl_layer, betas=betas, **kwargs)

    @property
    def vertex_normals(self):
        return self.mesh_seq.vertex_normals

    @property
    def poses(self):
        return torch.cat((self.poses_root, self.poses_body), dim=-1)

    def fk(self):
        """Get joints and/or vertices from the poses."""
        verts, joints = self.smpl_layer(poses_root=self.poses_root,
                                        poses_body=self.poses_body,
                                        poses_left_hand=self.poses_left_hand,
                                        poses_right_hand=self.poses_right_hand,
                                        betas=self.betas,
                                        trans=self.trans)
        skeleton = self.smpl_layer.skeletons()['body'].T
        faces = self.smpl_layer.bm.faces.astype(np.int64)
        joints = joints[:, :skeleton.shape[0]]
        return c2c(verts), c2c(joints), c2c(faces), c2c(skeleton)

    def interpolate(self, frame_ids):
        """
        Replace the frames at the given frame IDs via an interpolation of its neighbors. Only the body pose as well
        as the root pose and translation are interpolated.
        :param frame_ids: A list of frame ids to be interpolated.
        """
        ids = np.unique(frame_ids)
        all_ids = np.arange(self.n_frames)
        mask_avail = np.ones(self.n_frames, dtype=np.bool)
        mask_avail[ids] = False

        # Interpolate poses.
        all_poses = torch.cat([self.poses_root, self.poses_body], dim=-1)
        ps = np.reshape(all_poses.cpu().numpy(), (self.n_frames, -1, 3))
        ps_interp = interpolate_rotations(ps[mask_avail], all_ids[mask_avail], ids)
        all_poses[ids] = torch.from_numpy(ps_interp.reshape(len(ids), -1)).to(dtype=self.betas.dtype,
                                                                              device=self.betas.device)
        self.poses_root = all_poses[:, :3]
        self.poses_body = all_poses[:, 3:]

        # Interpolate global translation.
        ts = self.trans.cpu().numpy()
        ts_interp = interpolate_positions(ts[mask_avail], all_ids[mask_avail], ids)
        self.trans[ids] = torch.from_numpy(ts_interp).to(dtype=self.betas.dtype, device=self.betas.device)

        self.redraw()

    def redraw(self):
        self.vertices, self.joints, self.faces, self.skeleton = self.fk()
        if self._is_rigged:
            self.skeleton_seq.joint_positions = self.joints
        self.mesh_seq.vertices = self.vertices
        super().redraw()

    def gui(self, imgui):
        super().gui_animation(imgui)
        super().gui_position(imgui)

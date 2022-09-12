"""
Copyright (C) 2022  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos

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
from aitviewer.utils.decorators import hooked
from aitviewer.utils.so3 import aa2euler_numpy, aa2rot_torch as aa2rot, euler2aa_numpy
from aitviewer.utils.so3 import rot2aa_torch as rot2aa
from aitviewer.utils.so3 import interpolate_rotations
from aitviewer.utils.so3 import resample_rotations
from aitviewer.utils import resample_positions
from aitviewer.utils import to_torch
from aitviewer.utils import local_to_global
from aitviewer.utils import interpolate_positions
from aitviewer.utils import to_numpy as c2c
from scipy.spatial.transform import Rotation
from smplx.joint_names import JOINT_NAMES, SMPLH_JOINT_NAMES


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
                 z_up=False,
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
        :param z_up: Whether or not the input data assumes Z is up. If so, the data will be rotated such that Y is up.
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
        self._z_up = z_up

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

        # Edit mode
        self._edit_mode = False
        self._edit_joint = None
        self._edit_pose = None
        self._edit_pose_dirty = False

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

        # First convert the relative joint angles to global joint angles in rotation matrix form.
        if self.smpl_layer.model_type != "flame":
            global_oris = local_to_global(torch.cat([self.poses_root, self.poses_body], dim=-1),
                                          self.skeleton[:, 0], output_format='rotmat')
            global_oris = c2c(global_oris.reshape((self.n_frames, -1, 3, 3)))
        else:
            global_oris = np.tile(np.eye(3), self.joints.shape[:-1])[np.newaxis]
            
        if self._z_up:
            self.rotation = np.matmul(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), self.rotation) 

        self.rbs = RigidBodies(self.joints, global_oris, length=0.1, name='Joint Angles')
        self.rbs.position = self.position
        self.rbs.rotation = self.rotation
        self._add_node(self.rbs, enabled=self._show_joint_angles)

        kwargs = self._render_kwargs.copy()
        kwargs['name'] = 'Mesh'
        kwargs['color'] = kwargs.get('color', (160 / 255, 160 / 255, 160 / 255, 1.0))
        self.mesh_seq = Meshes(self.vertices, self.faces, is_selectable=False, **kwargs)
        self.mesh_seq.position = self.position
        self.mesh_seq.rotation = self.rotation
        self._add_node(self.mesh_seq, gui_elements=['material'])

        # Save view mode state to restore when exiting edit mode.
        self._view_mode_color = self.mesh_seq.color
        self._view_mode_joint_angles = self._show_joint_angles

    @classmethod
    def from_amass(cls, npz_data_path, start_frame=None, end_frame=None, log=True, fps_out=None, z_up=True, **kwargs):
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

        i_root_end = 3
        i_body_end = i_root_end + smpl_layer.bm.NUM_BODY_JOINTS * 3
        i_left_hand_end = i_body_end + smpl_layer.bm.NUM_HAND_JOINTS * 3
        i_right_hand_end = i_left_hand_end + smpl_layer.bm.NUM_HAND_JOINTS * 3

        return cls(poses_body=poses[:, i_root_end:i_body_end],
                   poses_root=poses[:, :i_root_end],
                   poses_left_hand=poses[:, i_body_end: i_left_hand_end],
                   poses_right_hand=poses[:, i_left_hand_end: i_right_hand_end],
                   smpl_layer=smpl_layer,
                   betas=body_data['betas'][np.newaxis],
                   trans=trans, z_up=z_up, **kwargs)

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

    def fk(self, current_frame_only=False):
        """Get joints and/or vertices from the poses."""
        if current_frame_only:
            # Use current frame data.
            if self._edit_mode:
                poses_root = self._edit_pose[:3][None, :]
                poses_body = self._edit_pose[3:][None, :]
            else:
                poses_body = self.poses_body[self.current_frame_id][None, :]
                poses_root = self.poses_root[self.current_frame_id][None, :]

            poses_left_hand = None if self.poses_left_hand is None else self.poses_left_hand[self.current_frame_id][None, :]
            poses_right_hand = None if self.poses_right_hand is None else self.poses_right_hand[self.current_frame_id][None, :]
            trans = self.trans[self.current_frame_id][None, :]

            if self.betas.shape[0] == self.n_frames:
                betas = self.betas[self.current_frame_id][None, :]
            else:
                betas = self.betas
        else:
            # Use the whole sequence.
            if self._edit_mode:
                poses_root = self.poses_root.clone()
                poses_body = self.poses_body.clone()

                poses_root[self.current_frame_id] = self._edit_pose[:3]
                poses_body[self.current_frame_id] = self._edit_pose[3:]
            else:
                poses_body = self.poses_body
                poses_root = self.poses_root

            poses_left_hand = self.poses_left_hand
            poses_right_hand = self.poses_right_hand
            trans = self.trans
            betas = self.betas

        verts, joints = self.smpl_layer(poses_root=poses_root,
                                        poses_body=poses_body,
                                        poses_left_hand=poses_left_hand,
                                        poses_right_hand=poses_right_hand,
                                        betas=betas,
                                        trans=trans)



        skeleton = self.smpl_layer.skeletons()['body'].T
        faces = self.smpl_layer.bm.faces.astype(np.int64)
        joints = joints[:, :skeleton.shape[0]]

        if current_frame_only:
            return c2c(verts)[0], c2c(joints)[0], c2c(faces), c2c(skeleton)
        else:
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

    @hooked
    def on_before_frame_update(self):
        if self._edit_mode and self._edit_pose_dirty:
            self._edit_pose = self.poses[self.current_frame_id].clone()
            self.redraw(current_frame_only=True)
            self._edit_pose_dirty = False

    @hooked
    def on_frame_update(self):
        if self.edit_mode:
            self._edit_pose = self.poses[self.current_frame_id].clone()
            self._edit_pose_dirty = False

    def redraw(self, **kwargs):
        current_frame_only = kwargs.get('current_frame_only', False)

        # Use the edited pose if in edit mode.
        vertices, joints, self.faces, self.skeleton = self.fk(current_frame_only)

        if current_frame_only:
            self.vertices[self.current_frame_id] = vertices
            self.joints[self.current_frame_id] = joints

            if self._is_rigged:
                self.skeleton_seq.current_joint_positions = joints

            # Use current frame data.
            if self._edit_mode:
                pose = self._edit_pose
            else:
                pose = torch.cat([self.poses_root[self.current_frame_id], self.poses_body[self.current_frame_id]],
                                 dim=-1)

            # Update rigid bodies.
            if self.smpl_layer.model_type != 'flame':
                global_oris = local_to_global(pose, self.skeleton[:, 0], output_format='rotmat')
                global_oris = global_oris.reshape((-1, 3, 3))
                self.rbs.current_rb_ori = c2c(global_oris)
            self.rbs.current_rb_pos = self.joints[self.current_frame_id]

            # Update mesh.
            self.mesh_seq.current_vertices = vertices
        else:
            self.vertices = vertices
            self.joints = joints

            # Update skeleton.
            if self._is_rigged:
                self.skeleton_seq.joint_positions = self.joints

            # Extract poses including the edited pose.
            if self._edit_mode:
                poses_root = self.poses_root.clone()
                poses_body = self.poses_body.clone()

                poses_root[self.current_frame_id] = self._edit_pose[:3]
                poses_body[self.current_frame_id] = self._edit_pose[3:]
            else:
                poses_body = self.poses_body
                poses_root = self.poses_root

            # Update rigid bodies.
            if self.smpl_layer.model_type != 'flame':
                global_oris = local_to_global(torch.cat([poses_root, poses_body], dim=-1),
                                              self.skeleton[:, 0], output_format='rotmat')
                global_oris = global_oris.reshape((self.n_frames, -1, 3, 3))
                self.rbs.rb_ori = c2c(global_oris)
            self.rbs.rb_pos = self.joints

            # Update mesh
            self.mesh_seq.vertices = vertices

        super().redraw(**kwargs)

    @property
    def edit_mode(self):
        return self._edit_mode

    @edit_mode.setter
    def edit_mode(self, enabled):
        if enabled == self._edit_mode:
            return

        if not enabled:
            self._edit_mode = False

            self.mesh_seq.backface_fragmap = False
            self.mesh_seq.color = self._view_mode_color

            self.rbs.color = (0, 1, 0.5, 1.0)
            self.rbs.enabled = self._view_mode_joint_angles
            self.rbs.is_selectable = True
        else:
            self._edit_mode = True
            self.rbs.enabled = True
            self.rbs.is_selectable = False
            self._edit_pose = self.poses[self.current_frame_id].clone()

            # Disable picking for the mesh
            self.mesh_seq.backface_fragmap = True
            self.rbs.color = (1, 0, 0.5, 1.0)
            self._view_mode_color = self.mesh_seq.color
            self.mesh_seq.color = (*self._view_mode_color[:3], min(self._view_mode_color[3], 0.5))

        self.redraw(current_frame_only=True)

    def _gui_joint(self, imgui, j, tree=None):
        name = "unknown"
        if self.smpl_layer.model_type == "smplh":
            if j < len(SMPLH_JOINT_NAMES):
                name = SMPLH_JOINT_NAMES[j]
        else:
            if j < len(JOINT_NAMES):
                name = JOINT_NAMES[j]

        if tree:
            e = imgui.tree_node(f'{j} - {name}')
        else:
            e = True
            imgui.text(f'{j} - {name}')

        if e:
            aa = self._edit_pose[j * 3: (j + 1) * 3]
            euler = aa2euler_numpy(aa.cpu().numpy(), degrees=True)
            u, euler = imgui.drag_float3(f'##joint{j}', *euler, 0.1, format='%.2f')
            if u:
                aa = euler2aa_numpy(np.array(euler), degrees=True)
                self._edit_pose[j * 3: (j + 1) * 3] = torch.from_numpy(aa)
                self._edit_pose_dirty = True
                self.redraw(current_frame_only=True)
            if tree:
                for c in tree.get(j, []):
                    self._gui_joint(imgui, c, tree)
                imgui.tree_pop()

    def gui(self, imgui):
        super().gui_animation(imgui)
        super().gui_position(imgui)

        if imgui.radio_button("View mode", not self.edit_mode):
            self.edit_mode = False
        if imgui.radio_button("Edit mode", self.edit_mode):
            self.edit_mode = True

        imgui.spacing()

        if self.edit_mode:
            skel = self.smpl_layer.skeletons()['body'].cpu().numpy()

            tree = {}
            for i in range(skel.shape[1]):
                if skel[0, i] != -1:
                    tree.setdefault(skel[0, i], []).append(skel[1, i])

            if not tree:
                return

            if self._edit_joint is None:
                self._gui_joint(imgui, 0, tree)
            else:
                self._gui_joint(imgui, self._edit_joint)

            if imgui.button("Apply"):
                self.poses_root[self.current_frame_id] = self._edit_pose[:3]
                self.poses_body[self.current_frame_id] = self._edit_pose[3:]
                self._edit_pose_dirty = False
                self.redraw(current_frame_only=True)
            imgui.same_line()
            if imgui.button("Apply to all"):
                edit_rots = Rotation.from_rotvec(np.reshape(self._edit_pose, (-1, 3)))
                base_rots = Rotation.from_rotvec(np.reshape(self.poses[self.current_frame_id], (-1, 3)))
                relative = edit_rots * base_rots.inv()
                for i in range(self.n_frames):
                    root = Rotation.from_rotvec(np.reshape(self.poses_root[i], (-1, 3)))
                    self.poses_root[i] = torch.from_numpy((relative[0] * root).as_rotvec().flatten())

                    body = Rotation.from_rotvec(np.reshape(self.poses_body[i], (-1, 3)))
                    self.poses_body[i] = torch.from_numpy((relative[1:] * body).as_rotvec().flatten())
                self._edit_pose_dirty = False
                self.redraw()
            imgui.same_line()
            if imgui.button("Reset"):
                self._edit_pose = self.poses[self.current_frame_id]
                self._edit_pose_dirty = False
                self.redraw(current_frame_only=True)

    def gui_context_menu(self, imgui):
        if self.edit_mode and self._edit_joint is not None:
            self._gui_joint(imgui, self._edit_joint)
        else:
            if imgui.radio_button("View mode", not self.edit_mode):
                self.edit_mode = False
                imgui.close_current_popup()
            if imgui.radio_button("Edit mode", self.edit_mode):
                self.edit_mode = True
                imgui.close_current_popup()

    def on_selection(self, node, tri_id):
        if self.edit_mode:
            # Find the index of the joint that is currently being edited.
            self._edit_joint = self.rbs.get_index_from_node_and_triangle(node, tri_id)
            if self._edit_joint is not None:
                self.rbs.color_one(self._edit_joint, (0.3, 0.4, 1, 1))
            else:
                # Reset color of all spheres to the default color
                self.rbs.color = self.rbs.color

    def render_outline(self, ctx, camera, prog):
        # Only render outline of the mesh, skipping skeleton and rigid bodies.
        self.mesh_seq.render_outline(ctx, camera, prog)

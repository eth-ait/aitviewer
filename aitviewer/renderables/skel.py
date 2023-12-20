"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See https://skel.is.tue.mpg.de/license.html for licensing and contact information.
"""

import numpy as np
import pickle as pkl
import torch
import tqdm
import trimesh
import os

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
from typing import Union, IO

from aitviewer.utils.colors import vertex_colors_from_weights

SKEL_JOINT_NAMES = [ 
 'pelvis',
 'femur_r',
 'tibia_r',
 'talus_r',
 'calcn_r',
 'toes_r',
 'femur_l',
 'tibia_l',
 'talus_l',
 'calcn_l',
 'toes_l',
 'lumbar_body',
 'thorax',
 'head',
 'scapula_r',
 'humerus_r',
 'ulna_r',
 'radius_r',
 'hand_r',
 'scapula_l',
 'humerus_l',
 'ulna_l',
 'radius_l',
 'hand_l']


class SKELSequence(Node):
    """
    Represents a temporal sequence of SMPL poses. Can be loaded from disk or initialized from memory.
    """

    def __init__(self,
                 poses_body,
                 skel_layer,
                 poses_type='skel',
                 betas=None,
                 trans=None,
                 device=C.device,
                 dtype=C.f_precision,
                 include_root=True,
                 normalize_root=False,
                 is_rigged=False,
                 show_joint_angles=False,
                 z_up=False,
                 fps=None,
                 fps_in = None,
                 show_joint_arrows = True,
                 visual = True,
                 post_fk_func=None,
                 skin_color = (200 / 255, 160 / 255, 160 / 255, 125/255),
                 skel_color = (160 / 255, 160 / 255, 160 / 255, 255/255),
                 skinning_weights_color = False,
                 **kwargs):
        """
        Initializer.
        :param poses_body: An array (numpy ar pytorch) of shape (F, 46) containing the pose parameters of the
          body, i.e. without hands or face parameters.
        :skel_layer: The SKEL layer that maps parameters to joint positions and/or body and meshes surfaces.
        :param betas: An array (numpy or pytorch) of shape (N_BETAS, ) containing the shape parameters.
        :param trans: An array (numpy or pytorch) of shape (F, 3) containing a global translation that is applied to
          all joints and vertices.
        :param device: The pytorch device for computations.
        :param dtype: The pytorch data type.
        :param include_root: Whether or not to include root information. If False, no root translation and no root
          rotation is applied.
        :param normalize_root: Whether or not to normalize the root. If True, the global root translation in the first
          frame is zero and the global root orientation is the identity.
        :param is_rigged: Whether or not to display the joints as a skeleton.
        :param show_joint_angles: Whether or not the coordinate frames at the joints should be visualized.
        :param z_up: Whether or not the input data assumes Z is up. If so, the data will be rotated such that Y is up.
        :param post_fk_func: User specified postprocessing function that is called after evaluating the SMPL model,
          the function signature must be: def post_fk_func(self, vertices, joints, current_frame_only),
          and it must return new values for vertices and joints with the same shapes.
          Shapes are:
            if current_frame_only is False: vertices (F, V, 3) and joints (F, N_JOINTS, 3)
            if current_frame_only is True:  vertices (1, V, 3) and joints (1, N_JOINTS, 3)
        :param kwargs: Remaining arguments for rendering.
        """
        assert len(poses_body.shape) == 2
              
        super(SKELSequence, self).__init__(n_frames=poses_body.shape[0], **kwargs)
        self.skel_layer = skel_layer
        self.post_fk_func = post_fk_func

        self.device = device
        self.fps = fps #fps of this loaded sequence
        self.fps_in = fps_in #original fps of the sequence

        self.poses_body = to_torch(poses_body, dtype=dtype, device=device)
        self.poses_type = poses_type

        betas = betas if betas is not None else torch.zeros([1, self.skel_layer.num_betas])
        trans = trans if trans is not None else torch.zeros([len(poses_body), 3])

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
            self.trans = torch.zeros_like(self.trans)

        # Edit mode
        self.gui_modes.update({'edit': {'title': ' Edit', 'fn': self.gui_mode_edit, 'icon': '\u0081'}})

        self._edit_joint = None
        self._edit_pose = None
        self._edit_pose_dirty = False

        # Nodes
        self.skin_vertices, self.skin_faces, self.skel_vertices, self.skel_faces, self.joints, self.joints_ori, self.skeleton = self.fk()
        
        if self._z_up:
            self.rotation = np.matmul(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), self.rotation)
            
        if visual == False:
            return

        # import ipdb; ipdb.set_trace()
        if self._is_rigged:
            # Must first add skeleton, otherwise transparency does not work correctly.
            # Overriding given color with a custom color for the skeleton.
            kwargs = self._render_kwargs.copy()
            color = (1.0, 177 / 255, 1 / 255, 1.0)
            
            self.skeleton_seq = Skeletons(self.joints, self.skeleton, color=color, name='Kin tree')

            self._add_node(self.skeleton_seq)

            global_oris = self.joints_ori
                
            if show_joint_arrows is False:
                arrow_length = 0
            else:
                arrow_length = 0.1

            self.rbs = RigidBodies(self.joints, global_oris, length=arrow_length, name='Joint Angles', color=(1.0, 177 / 255, 1 / 255, 1.0))
            self._add_node(self.rbs, enabled=self._show_joint_angles)
            
        def skining_weights_to_color(skinning_weights):
            """ Given a skinning weight matrix NvxNj, return a color matrix of shape Nv*3. For each joint Ji i in [0, Nj] , 
            the color is colors[i]"""
            
            joints_ids = np.arange(0, skinning_weights.shape[1])
            colors = vertex_colors_from_weights(joints_ids, scale_to_range_1=True, alpha=skin_color[-1], shuffle=True, seed = 1)
            
            weights_color = np.matmul(skinning_weights, colors)
            return weights_color


        kwargs = self._render_kwargs.copy()
        kwargs['name'] = 'Skin'
        if skinning_weights_color == True:

            skin_colors = skining_weights_to_color(skel_layer.weights.cpu().numpy()) 
            self.skin_mesh_seq = Meshes(self.skin_vertices, self.skin_faces, is_selectable=False, vertex_colors=skin_colors , **kwargs)
        else:
            self.skin_mesh_seq = Meshes(self.skin_vertices, self.skin_faces, is_selectable=False, color=skin_color , **kwargs)
        # self.skin_mesh_seq.position = self.position
        # self.skin_mesh_seq.rotation = self.rotation
        self._add_node(self.skin_mesh_seq)
        
        kwargs = self._render_kwargs.copy()
        kwargs['name'] = 'Bones'
        self.bones_mesh_seq = Meshes(self.skel_vertices, self.skel_faces, is_selectable=False, color=skel_color, **kwargs)
        # self.bones_mesh_seq.position = self.position
        # self.bones_mesh_seq.rotation = self.rotation
        self._add_node(self.bones_mesh_seq)

        # Save view mode state to restore when exiting edit mode.
        self._skin_view_mode_color = self.skin_mesh_seq.color
        self._skel_view_mode_color = self.bones_mesh_seq.color
        self._view_mode_joint_angles = self._show_joint_angles
        
        
        
    def get_rotated_global_joint(self):
        rot_smpl_joints = np.matmul(self.joints, self.rotation.T) 
        
        rot_joints_ori = np.zeros_like(self.joints_ori)
        for joint_idx in range(self.joints_ori.shape[1]):
            rot_joints_ori[:,joint_idx,:,:] = np.matmul( self.rotation, self.joints_ori[:,joint_idx,:,:])    
        
        return rot_smpl_joints, rot_joints_ori     
        
    @property
    def rotated_vertices(self):
        return np.matmul(self.vertices, self.rotation.T) + self.position
    
    @property
    def rotated_skel_vertices(self):
        return np.matmul(self.skel_vertices, self.rotation.T) + self.position
    
    @classmethod
    def from_skel_aligned(cls, skel_layer, fps_in, pkl_data_path, start_frame=None, end_frame=None, log=True, fps_out=None, z_up=False, 
                   device=C.device, poses_type='skel', **kwargs):
        
        """Load a sequence downloaded from the AMASS website."""
        skel_data = pkl.load(open(pkl_data_path, 'rb'))          
        gender = skel_data['gender']

        # import ipdb; ipdb.set_trace()
        assert gender == skel_layer.gender, f"skel layer has gender {skel_layer.gender} while data has gender {gender}"

        sf = start_frame or 0
        ef = end_frame or skel_data['pose'].shape[0]
        poses = skel_data['pose'][sf:ef]
        trans = skel_data['trans'][sf:ef]
        betas = skel_data['betas'][sf:ef]

        # import ipdb; ipdb.set_trace()
        if fps_out is not None:
            if fps_in != fps_out:
                betas = resample_positions(betas, fps_in, fps_out)
                poses = resample_positions(poses, fps_in, fps_out) # Linear interpolation
                print("WARNING: poses resampled with linear interpolation, this is wrong but of for int fps ratio")
                trans = resample_positions(trans, fps_in, fps_out)
        else:
            fps_out = fps_in
            
        i_beta_end = skel_layer.num_betas 
        return cls(poses_body=poses,
                   skel_layer=skel_layer,
                   betas=betas[:, :i_beta_end],
                   trans=trans,
                   z_up=z_up,
                   gender=gender,
                   device = device,
                   fps = fps_out,
                   fps_in = fps_in,
                   poses_type = poses_type,
                    **kwargs)           



    @classmethod
    def t_pose(cls, skel_layer, betas=None, frames=1, is_rigged=True, show_joint_angles=False, device=C.device, **kwargs):
        """Creates a SKEL sequence whose single frame is a SKEL mesh in T-Pose."""
        
        if betas is not None:
            assert betas.shape[0] == 1

        poses = np.zeros([frames, skel_layer.num_q_params]) 
        return cls(poses, skel_layer=skel_layer, betas=betas, is_rigged=is_rigged, show_joint_angles=show_joint_angles, device=device, **kwargs)


    @property
    def color(self):
        return self.mesh_seq.color

    @color.setter
    def color(self, color):
        self.mesh_seq.color = color

    @property
    def bounds(self):
        return self.skin_mesh_seq.bounds

    @property
    def current_bounds(self):
        return self.skin_mesh_seq.current_bounds
    
    @property
    def vertex_normals(self):
        return self.skin_mesh_seq.vertex_normals

    @property
    def poses(self):
        return self.poses_body
    
    @property
    def _edit_mode(self):
        return self.selected_mode == 'edit'
    
    def fk(self, current_frame_only=False):
        """Get joints and/or vertices from the poses."""
        if current_frame_only:
            # Use current frame data.
            if self._edit_mode:
                poses_body = self._edit_pose[None, :]
            else:
                poses_body = self.poses_body[self.current_frame_id][None, :]


            trans = self.trans[self.current_frame_id][None, :]

            if self.betas.shape[0] == self.n_frames:
                betas = self.betas[self.current_frame_id][None, :]
            else:
                betas = self.betas
        else:
            # Use the whole sequence.
            if self._edit_mode:
                poses_body = self.poses_body.clone()
                poses_body[self.current_frame_id] = self._edit_pose
            else:
                poses_body = self.poses_body

            trans = self.trans
            betas = self.betas

        skel_output = self.skel_layer(poses=poses_body,
                                        betas=betas,
                                        trans=trans,
                                        poses_type=self.poses_type)
        
        skin_verts = skel_output.skin_verts
        skel_verts = skel_output.skel_verts
        joints = skel_output.joints
        joints_ori = skel_output.joints_ori
        
        skeleton = self.skel_layer.kintree_table.T 
        skeleton[0, 0] = -1     
                
        skin_f = self.skel_layer.skin_f
        skel_f = self.skel_layer.skel_f

        if current_frame_only:
            return c2c(skin_verts)[0], c2c(skin_f)[0], c2c(skel_verts)[0], c2c(skel_f)[0], c2c(joints)[0], c2c(joints_ori)[0], c2c(skeleton)
        else:
            return c2c(skin_verts), c2c(skin_f), c2c(skel_verts), c2c(skel_f), c2c(joints), c2c(joints_ori), c2c(skeleton)


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
        skin_vertices, skin_faces, skel_vertices, skel_faces, joints, joints_ori, skeleton = self.fk(current_frame_only)

        if current_frame_only:
            self.skin_vertices[self.current_frame_id] = skin_vertices
            self.skel_vertices[self.current_frame_id] = skel_vertices
            self.joints[self.current_frame_id] = joints

            if self._is_rigged:
                self.skeleton_seq.current_joint_positions = joints

            # Use current frame data.
            if self._edit_mode:
                pose = self._edit_pose
            else:
                pose = self.poses_body[self.current_frame_id]

            # Update rigid bodies.
            global_oris = joints_ori
            self.rbs.current_rb_ori = c2c(global_oris)
            self.rbs.current_rb_pos = self.joints[self.current_frame_id]

            # Update mesh.
            self.skin_mesh_seq.current_vertices = skin_vertices
            self.bones_mesh_seq.current_vertices = skel_vertices
        else:
            self.skin_vertices = skin_vertices
            self.skel_vertices = skel_vertices
            self.joints = joints

            # Update skeleton.
            if self._is_rigged:
                self.skeleton_seq.joint_positions = self.joints

            # Extract poses including the edited pose.
            if self._edit_mode:
                poses_body = self.poses_body.clone()
                poses_body[self.current_frame_id] = self._edit_pose
            else:
                poses_body = self.poses_body
                poses_root = self.poses_root

            # Update rigid bodies.
            global_oris = joints_ori
            self.rbs.rb_ori = c2c(global_oris)
            self.rbs.rb_pos = self.joints

            # Update mesh
            self.skin_mesh_seq.vertices = skin_vertices
            self.bones_mesh_seq.vertices = skel_vertices

        super().redraw(**kwargs)

    @property
    def edit_mode(self):
        return self._edit_mode

    @property
    def selected_mode(self):
        return self._selected_mode

    @selected_mode.setter
    def selected_mode(self, selected_mode):
        if self._selected_mode == selected_mode:
            return
        self._selected_mode = selected_mode

        if self.selected_mode == 'edit':
            self.rbs.enabled = True
            self.rbs.is_selectable = False
            self._edit_pose = self.poses[self.current_frame_id].clone()

            # Disable picking for the mesh
            self.skin_mesh_seq.backface_fragmap = True
            self.bones_mesh_seq.backface_fragmap = True
            self.rbs.color = (1, 0, 0.5, 1.0)
            
            self._skin_view_mode_color = self.skin_mesh_seq.color
            self.skin_mesh_seq.color = (*self._skin_view_mode_color[:3], min(self._skin_view_mode_color[3], 0.5))
            
            self._skel_view_mode_color = self.bones_mesh_seq.color
            self.bones_mesh_seq.color = (*self._skel_view_mode_color[:3], min(self._skel_view_mode_color[3], 0.5))

        self.redraw(current_frame_only=True)

    def _gui_joint(self, imgui, j, tree=None):
        name = "unknown"
        if j < len(SKEL_JOINT_NAMES):
            name = SKEL_JOINT_NAMES[j]

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

    def gui_mode_edit(self, imgui):
        kin_skel = self.skeleton

        tree = {}
        for i in range(kin_skel.shape[1]):
            if kin_skel[0, i] != -1:
                tree.setdefault(kin_skel[0, i], []).append(kin_skel[1, i])

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
            edit_rots = Rotation.from_rotvec(np.reshape(self._edit_pose.cpu().numpy(), (-1, 3)))
            base_rots = Rotation.from_rotvec(np.reshape(self.poses[self.current_frame_id].cpu().numpy(), (-1, 3)))
            relative = edit_rots * base_rots.inv()
            for i in range(self.n_frames):
                root = Rotation.from_rotvec(np.reshape(self.poses_root[i].cpu().numpy(), (-1, 3)))
                self.poses_root[i] = torch.from_numpy((relative[0] * root).as_rotvec().flatten())

                body = Rotation.from_rotvec(np.reshape(self.poses_body[i].cpu().numpy(), (-1, 3)))
                self.poses_body[i] = torch.from_numpy((relative[1:] * body).as_rotvec().flatten())
            self._edit_pose_dirty = False
            self.redraw()
        imgui.same_line()
        if imgui.button("Reset"):
            self._edit_pose = self.poses[self.current_frame_id]
            self._edit_pose_dirty = False
            self.redraw(current_frame_only=True)

    def gui_io(self, imgui):
        if imgui.button("Export sequence to NPZ"):
            dir = os.path.join(C.export_dir, "SMPL")
            os.makedirs(dir, exist_ok=True)
            path = os.path.join(dir, self.name + ".npz")
            self.export_to_npz(path)
            print(f'Exported SMPL sequence to "{path}"')

    def gui_context_menu(self, imgui):
        if self.edit_mode and self._edit_joint is not None:
            self._gui_joint(imgui, self._edit_joint)
        else:
            if imgui.radio_button("View mode", not self.edit_mode):
                self.selected_mode = 'view'
                imgui.close_current_popup()
            if imgui.radio_button("Edit mode", self.edit_mode):
                self.selected_mode = 'edit'
                imgui.close_current_popup()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            super().gui_context_menu(imgui)
                
    def on_selection(self, node, instance_id, tri_id):
        if self.edit_mode:
            # Index of the joint that is currently being edited.
            if node != self.skin_mesh_seq and node != self.bones_mesh_seq:
                self._edit_joint = instance_id
                self.rbs.color_one(self._edit_joint, (0.3, 0.4, 1, 1))
            else:
                self._edit_joint = None
                # Reset color of all spheres to the default color
                self.rbs.color = self.rbs.color

    def render_outline(self, *args, **kwargs):
        # Only render outline of the mesh, skipping skeleton and rigid bodies.
        self.skin_mesh_seq.render_outline(*args, **kwargs)
        self.bones_mesh_seq.render_outline(*args, **kwargs)
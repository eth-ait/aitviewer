"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See https://skel.is.tue.mpg.de/license.html for licensing and contact information.
"""

import os
import pickle as pkl

import numpy as np
import torch
import tqdm
import trimesh
from scipy.spatial.transform import Rotation

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.scene.node import Node
from aitviewer.utils import interpolate_positions, local_to_global, resample_positions
from aitviewer.utils import to_numpy as c2c
from aitviewer.utils import to_torch
from aitviewer.utils.decorators import hooked
from aitviewer.utils.so3 import aa2euler_numpy
from aitviewer.utils.so3 import aa2rot_torch as aa2rot
from aitviewer.utils.so3 import (
    euler2aa_numpy,
    interpolate_rotations,
    resample_rotations,
)
from aitviewer.utils.so3 import rot2aa_torch as rot2aa

try:
    from skel.kin_skel import skel_joints_name
    from skel.skel_model import SKEL
except ImportError as e:
    raise ImportError(f"Could not import SKEL. Please install it from https://github.com/MarilynKeller/skel.git")

from aitviewer.utils.colors import skining_weights_to_color, vertex_colors_from_weights


class SKELSequence(Node):
    """
    Represents a temporal sequence of SMPL poses. Can be loaded from disk or initialized from memory.
    """

    def __init__(
        self,
        poses_body,
        skel_layer,
        poses_type="skel",
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
        fps_in=None,
        show_joint_arrows=True,
        visual=True,
        post_fk_func=None,
        skin_color=(200 / 255, 160 / 255, 160 / 255, 125 / 255),
        skel_color=(160 / 255, 160 / 255, 160 / 255, 255 / 255),
        skin_coloring=None,  # "pose_offsets", "skinning_weights"
        skel_coloring=None,  # "skinning_weights", "bone_label"
        **kwargs,
    ):
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
        :param skin_coloring: Coloring the skin mesh of SKEL per vertex. Must be in ['skinning_weights', 'pose_offsets'].
        :param skel_coloring: Coloring the bones mesh of SKEL per vertex. Must be in ['skinning_weights', 'bone_label'].
        :param kwargs: Remaining arguments for rendering.
        """
        assert len(poses_body.shape) == 2

        super(SKELSequence, self).__init__(n_frames=poses_body.shape[0], **kwargs)
        self.skel_layer = skel_layer
        self.post_fk_func = post_fk_func

        self.device = device
        self.fps = fps  # fps of this loaded sequence
        self.fps_in = fps_in  # original fps of the sequence

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
        self.gui_modes.update({"edit": {"title": " Edit", "fn": self.gui_mode_edit, "icon": "\u0081"}})

        self._edit_joint = None
        self._edit_pose = None
        self._edit_pose_dirty = False

        # Nodes
        skel_output = self.fk()

        self.skin_vertices = skel_output.skin_verts
        self.skel_vertices = skel_output.skel_verts
        self.skin_faces = skel_output.skin_f
        self.skel_faces = skel_output.skel_f
        self.joints = skel_output.joints
        self.joints_ori = skel_output.joints_ori
        self.skeleton = skel_output.skeleton

        self.skel_output = skel_output

        if self._z_up:
            self.rotation = np.matmul(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), self.rotation)

        if visual == False:
            return

        if self._is_rigged:
            # Must first add skeleton, otherwise transparency does not work correctly.
            # Overriding given color with a custom color for the skeleton.
            kwargs = self._render_kwargs.copy()
            color = (1.0, 177 / 255, 1 / 255, 1.0)

            self.skeleton_seq = Skeletons(self.joints, self.skeleton, gui_affine=False, color=color, name="Kin tree")
            # self.skeleton_seq.position = self.position
            # self.skeleton_seq.rotation = self.rotation
            self._add_node(self.skeleton_seq)

            global_oris = self.joints_ori

            if show_joint_arrows is False:
                arrow_length = 0
            else:
                arrow_length = 0.1

            self.rbs = RigidBodies(
                self.joints, global_oris, length=arrow_length, name="Joint Angles", color=(1.0, 177 / 255, 1 / 255, 1.0)
            )
            self._add_node(self.rbs, enabled=self._show_joint_angles)

        # Instantiate the Skin submesh with proper colouring
        kwargs = self._render_kwargs.copy()
        mesh_name = "Skin"
        skin_colors = None
        if skin_coloring == "skinning_weights":
            skin_colors = skining_weights_to_color(
                skel_layer.skin_weights.to_dense().cpu().numpy(), alpha=skin_color[-1]
            )
        elif skin_coloring == "pose_offsets":
            values = self.skel_output.pose_offsets.cpu().numpy()
            skin_colors = values / np.max(np.abs(values))
            # append an alpha channel
            skin_colors = np.concatenate(
                [skin_colors, np.ones([skin_colors.shape[0], skin_colors.shape[1], 1])], axis=-1
            )
            self.skin_mesh_seq = Meshes(
                self.skin_vertices, self.skin_faces, gui_affine=False, is_selectable=False, vertex_colors=skin_colors
            )

        if skin_colors is None:
            self.skin_mesh_seq = Meshes(
                self.skin_vertices,
                self.skin_faces,
                gui_affine=False,
                is_selectable=False,
                color=skin_color,
                name=mesh_name,
            )
        else:
            self.skin_mesh_seq = Meshes(
                self.skin_vertices,
                self.skin_faces,
                gui_affine=False,
                is_selectable=False,
                vertex_colors=skin_colors,
                name=mesh_name,
            )

        self._add_node(self.skin_mesh_seq)

        # Instantiate the Bones submesh with proper colouring
        mesh_name = "Bones"
        skel_colors = None
        if skel_coloring == "skinning_weights":
            skel_colors = skining_weights_to_color(skel_layer.skel_weights.cpu().numpy(), alpha=255)
        elif skel_coloring == "bone_label":
            skel_colors = skining_weights_to_color(skel_layer.skel_weights_rigid.cpu().numpy(), alpha=255)

        if skel_colors is None:
            self.bones_mesh_seq = Meshes(
                self.skel_vertices,
                self.skel_faces,
                gui_affine=False,
                is_selectable=False,
                color=skel_color,
                name=mesh_name,
            )
        else:
            draw_bones_outline = False
            if skin_coloring == "skinning_weights" and (
                skel_coloring == "skinning_weights" or skel_coloring == "bone_label"
            ):
                draw_bones_outline = True  # For better visibility of the bones
            self.bones_mesh_seq = Meshes(
                self.skel_vertices,
                self.skel_faces,
                gui_affine=False,
                is_selectable=False,
                vertex_colors=skel_colors,
                name=mesh_name,
                draw_outline=draw_bones_outline,
            )

        self._add_node(self.bones_mesh_seq)

        # Save view mode state to restore when exiting edit mode.
        self._skin_view_mode_color = self.skin_mesh_seq.color
        self._skel_view_mode_color = self.bones_mesh_seq.color
        self._view_mode_joint_angles = self._show_joint_angles

    def get_rotated_global_joint(self):
        rot_smpl_joints = np.matmul(self.joints, self.rotation.T)

        rot_joints_ori = np.zeros_like(self.joints_ori)
        for joint_idx in range(self.joints_ori.shape[1]):
            rot_joints_ori[:, joint_idx, :, :] = np.matmul(self.rotation, self.joints_ori[:, joint_idx, :, :])

        return rot_smpl_joints, rot_joints_ori

    @property
    def rotated_vertices(self):
        return np.matmul(self.vertices, self.rotation.T) + self.position

    @property
    def rotated_skel_vertices(self):
        return np.matmul(self.skel_vertices, self.rotation.T) + self.position

    @classmethod
    def from_file(
        cls,
        skel_seq_file,
        fps_in,
        start_frame=None,
        end_frame=None,
        log=True,
        fps_out=None,
        z_up=False,
        device=C.device,
        poses_type="skel",
        **kwargs,
    ):
        """Load a SKEL sequence from a pkl."""

        if skel_seq_file.endswith(".pkl"):
            skel_data = pkl.load(open(skel_seq_file, "rb"))
        elif skel_seq_file.endswith(".npz"):
            # Compatibility with PS fitting pipeline
            skel_data = np.load(skel_seq_file)
            skel_data = {key: skel_data[key] for key in skel_data.files}
            if "poses" not in skel_data and "pose" in skel_data and "global_orient" in skel_data:
                skel_data["poses"] = np.concatenate(
                    [
                        skel_data["global_orient"],
                        skel_data["pose"],
                    ],
                    axis=1,
                )
            if "trans" not in skel_data and "transl" in skel_data:
                skel_data["trans"] = skel_data["transl"]
                del skel_data["transl"]
            if "gender" not in skel_data:
                print("Warning: no gender found in the npz file, assuming female.")
                skel_data["gender"] = "female"
            if skel_data["betas"].shape[0] == 1:
                skel_data["betas"] = skel_data["betas"].repeat(skel_data["poses"].shape[0], axis=0)

            import ipdb

            ipdb.set_trace()

        else:
            raise ValueError(f"skel_seq_file must be a pkl or npz file, got {skel_seq_file}")

        for key in ["poses", "trans", "betas", "gender"]:
            assert (
                key in skel_data
            ), f"The loaded skel sequence dictionary must contain {key}. Loaded dictionary has keys: {skel_data.keys()}"

        gender = skel_data["gender"]
        skel_layer = SKEL(model_path=C.skel_models, gender=gender)

        assert gender == skel_layer.gender, f"skel layer has gender {skel_layer.gender} while data has gender {gender}"

        sf = start_frame or 0
        ef = end_frame or skel_data["poses"].shape[0]
        poses = skel_data["poses"][sf:ef]
        trans = skel_data["trans"][sf:ef]
        betas = skel_data["betas"][sf:ef]

        if fps_out is not None:
            if fps_in != fps_out:
                betas = resample_positions(betas, fps_in, fps_out)
                poses = resample_positions(poses, fps_in, fps_out)  # Linear interpolation
                print("WARNING: poses resampled with linear interpolation, this is wrong but ok for int fps ratio")
                trans = resample_positions(trans, fps_in, fps_out)
        else:
            fps_out = fps_in

        i_beta_end = skel_layer.num_betas
        return cls(
            poses_body=poses,
            skel_layer=skel_layer,
            betas=betas[:, :i_beta_end],
            trans=trans,
            z_up=z_up,
            device=device,
            fps=fps_out,
            fps_in=fps_in,
            poses_type=poses_type,
            **kwargs,
        )

    @classmethod
    def t_pose(
        cls, skel_layer, betas=None, frames=1, is_rigged=True, show_joint_angles=False, device=C.device, **kwargs
    ):
        """Creates a SKEL sequence whose single frame is a SKEL mesh in T-Pose."""

        if betas is not None:
            assert betas.shape[0] == 1

        poses = np.zeros([frames, skel_layer.num_q_params])
        return cls(
            poses,
            skel_layer=skel_layer,
            betas=betas,
            is_rigged=is_rigged,
            show_joint_angles=show_joint_angles,
            device=device,
            **kwargs,
        )

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
        return self.selected_mode == "edit"

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

        skel_output = self.skel_layer(poses=poses_body, betas=betas, trans=trans, poses_type=self.poses_type)

        # skin_verts = skel_output.skin_verts
        # skel_verts = skel_output.skel_verts
        # joints = skel_output.joints
        # joints_ori = skel_output.joints_ori

        if current_frame_only:
            # return c2c(skin_verts)[0], c2c(skin_f)[0], c2c(skel_verts)[0], c2c(skel_f)[0], c2c(joints)[0], c2c(joints_ori)[0], c2c(skeleton)
            for att in ["skin_verts", "skel_verts", "joints", "joints_ori"]:
                att_value = getattr(skel_output, att)
                setattr(skel_output, att, c2c(att_value)[0])
        else:
            # return c2c(skin_verts), c2c(skin_f), c2c(skel_verts), c2c(skel_f), c2c(joints), c2c(joints_ori), c2c(skeleton)
            for att in ["skin_verts", "skel_verts", "joints", "joints_ori"]:
                try:
                    att_value = getattr(skel_output, att)
                    setattr(skel_output, att, c2c(att_value))
                except:
                    import ipdb

                    ipdb.set_trace()

        skel_output.skin_f = c2c(self.skel_layer.skin_f)
        skel_output.skel_f = c2c(self.skel_layer.skel_f)

        skeleton = self.skel_layer.kintree_table.T
        skeleton[0, 0] = -1
        skel_output.skeleton = c2c(skeleton)

        return skel_output

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
        all_poses[ids] = torch.from_numpy(ps_interp.reshape(len(ids), -1)).to(
            dtype=self.betas.dtype, device=self.betas.device
        )
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
        current_frame_only = kwargs.get("current_frame_only", False)

        # Use the edited pose if in edit mode.
        # skin_vertices, skin_faces, skel_vertices, skel_faces, joints, joints_ori, skeleton = self.fk(current_frame_only)
        skel_output = self.fk(current_frame_only)

        if current_frame_only:
            self.skin_vertices[self.current_frame_id] = skel_output.skin_verts
            self.skel_vertices[self.current_frame_id] = skel_output.skel_verts
            self.joints[self.current_frame_id] = skel_output.joints

            if self._is_rigged:
                self.skeleton_seq.current_joint_positions = skel_output.joints

            # Use current frame data.
            if self._edit_mode:
                pose = self._edit_pose
            else:
                pose = self.poses_body[self.current_frame_id]

            # Update rigid bodies.
            global_oris = skel_output.joints_ori
            self.rbs.current_rb_ori = c2c(global_oris)
            self.rbs.current_rb_pos = self.joints[self.current_frame_id]

            # Update mesh.
            self.skin_mesh_seq.current_vertices = skel_output.skin_verts
            self.bones_mesh_seq.current_vertices = skel_output.skel_verts
        else:
            self.skin_vertices = skel_output.skin_verts
            self.skel_vertices = skel_output.skel_verts
            self.joints = skel_output.joints

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
            global_oris = skel_output.joints_ori
            self.rbs.rb_ori = c2c(global_oris)
            self.rbs.rb_pos = self.joints

            # Update mesh
            self.skin_mesh_seq.vertices = skel_output.skin_verts
            self.bones_mesh_seq.vertices = skel_output.skel_verts

        self.skel_output = skel_output

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

        if self.selected_mode == "edit":
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
        if j < len(skel_joints_name):
            name = skel_joints_name[j]

        if tree:
            e = imgui.tree_node(f"{j} - {name}")
        else:
            e = True
            imgui.text(f"{j} - {name}")

        if e:
            start_param = [0, 3, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 20, 23, 26, 29, 32, 33, 34, 36, 39, 42, 43, 44, 46]
            aa = self._edit_pose[start_param[j] : start_param[j + 1]]
            if len(aa) == 1:
                angle = np.degrees(aa.cpu().numpy())
                u, angle = imgui.drag_float(f"##joint{j}", angle, 0.1, format="%.2f")
                if u:
                    aa = np.array(np.radians(angle))
                    self._edit_pose[start_param[j] : start_param[j + 1]] = torch.from_numpy(aa)
                    self._edit_pose_dirty = True
                    self.redraw(current_frame_only=True)
            elif len(aa) == 2:
                angles = np.degrees(aa.cpu().numpy())
                u, angles = imgui.drag_float2(f"##joint{j}", *angles, 0.1, format="%.2f")
                if u:
                    aa = np.radians(np.array(angles))
                    self._edit_pose[start_param[j] : start_param[j + 1]] = torch.from_numpy(aa)
                    self._edit_pose_dirty = True
                    self.redraw(current_frame_only=True)
            elif len(aa) == 3:
                euler = aa2euler_numpy(aa.cpu().numpy(), degrees=True)
                u, euler = imgui.drag_float3(f"##joint{j}", *euler, 0.1, format="%.2f")
                if u:
                    aa = euler2aa_numpy(np.array(euler), degrees=True)
                    self._edit_pose[start_param[j] : start_param[j + 1]] = torch.from_numpy(aa)
                    self._edit_pose_dirty = True
                    self.redraw(current_frame_only=True)
            if tree:
                for c in tree.get(j, []):
                    self._gui_joint(imgui, c, tree)
                imgui.tree_pop()
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

    def gui_context_menu(self, imgui, x: int, y: int):
        if self.edit_mode and self._edit_joint is not None:
            self._gui_joint(imgui, self._edit_joint)
        else:
            if imgui.radio_button("View mode", not self.edit_mode):
                self.selected_mode = "view"
                imgui.close_current_popup()
            if imgui.radio_button("Edit mode", self.edit_mode):
                self.selected_mode = "edit"
                imgui.close_current_popup()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            super().gui_context_menu(imgui, x, y)

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

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
import os
import pickle as pkl
from typing import IO, Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from smplx.joint_names import JOINT_NAMES, SMPLH_JOINT_NAMES

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


class SMPLSequence(Node):
    """
    Represents a temporal sequence of SMPL poses. Can be loaded from disk or initialized from memory.
    """

    def __init__(
        self,
        poses_body,
        smpl_layer,
        poses_root=None,
        betas=None,
        trans=None,
        poses_left_hand=None,
        poses_right_hand=None,
        device=None,
        dtype=None,
        include_root=True,
        normalize_root=False,
        is_rigged=True,
        show_joint_angles=False,
        z_up=False,
        post_fk_func=None,
        icon="\u0093",
        **kwargs,
    ):
        """
        Initializer.
        :param poses_body: An array (numpy ar pytorch) of shape (F, N_JOINTS*3) containing the pose parameters of the
          body, i.e. without hands or face parameters.
        :param smpl_layer: The SMPL layer that maps parameters to joint positions and/or dense surfaces.
        :param poses_root: An array (numpy or pytorch) of shape (F, 3) containing the global root orientation.
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

        # Set model icon
        if smpl_layer.model_type == "mano":
            icon = "\u0092"
        elif smpl_layer.model_type == "flame":
            icon = "\u0091"

        if device is None:
            device = C.device
        if dtype is None:
            dtype = C.f_precision

        super(SMPLSequence, self).__init__(n_frames=poses_body.shape[0], icon=icon, gui_material=False, **kwargs)

        self.smpl_layer = smpl_layer
        self.post_fk_func = post_fk_func
        self.dtype = dtype
        self.device = device

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
        self.gui_modes.update({"edit": {"title": " Edit", "fn": self.gui_mode_edit, "icon": "\u0081"}})

        self._edit_joint = None
        self._edit_pose = None
        self._edit_pose_dirty = False
        self._edit_local_axes = True

        # Nodes
        self.vertices, self.joints, self.faces, self.skeleton = self.fk()

        if self._is_rigged:
            self.skeleton_seq = Skeletons(
                self.joints,
                self.skeleton,
                gui_affine=False,
                color=(1.0, 177 / 255, 1 / 255, 1.0),
                name="Skeleton",
            )
            self._add_node(self.skeleton_seq)

        # First convert the relative joint angles to global joint angles in rotation matrix form.
        if self.smpl_layer.model_type != "flame":
            global_oris = local_to_global(
                torch.cat([self.poses_root, self.poses_body], dim=-1),
                self.skeleton[:, 0],
                output_format="rotmat",
            )
            global_oris = c2c(global_oris.reshape((self.n_frames, -1, 3, 3)))
        else:
            global_oris = np.tile(np.eye(3), self.joints.shape[:-1])[np.newaxis]

        if self._z_up:
            self.rotation = np.matmul(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), self.rotation)

        self.rbs = RigidBodies(self.joints, global_oris, length=0.1, gui_affine=False, name="Joint Angles")
        self._add_node(self.rbs, enabled=self._show_joint_angles)

        self.mesh_seq = Meshes(
            self.vertices,
            self.faces,
            is_selectable=False,
            gui_affine=False,
            color=kwargs.get("color", (160 / 255, 160 / 255, 160 / 255, 1.0)),
            name="Mesh",
        )
        self._add_node(self.mesh_seq)

        # Save view mode state to restore when exiting edit mode.
        self._view_mode_color = self.mesh_seq.color
        self._view_mode_joint_angles = self._show_joint_angles

    @classmethod
    def from_amass(
        cls,
        npz_data_path,
        smpl_layer=None,
        start_frame=None,
        end_frame=None,
        log=True,
        fps_out=None,
        z_up=True,
        **kwargs,
    ):
        """Load a sequence downloaded from the AMASS website."""

        body_data = np.load(npz_data_path)
        if smpl_layer is None:
            smpl_layer = SMPLLayer(model_type="smplh", gender=body_data["gender"].item(), device=C.device)

        if log:
            print("Data keys available: {}".format(list(body_data.keys())))
            print("{:>6d} poses of size {:>4d}.".format(body_data["poses"].shape[0], body_data["poses"].shape[1]))
            print("{:>6d} trans of size {:>4d}.".format(body_data["trans"].shape[0], body_data["trans"].shape[1]))
            print("{:>6d} shape of size {:>4d}.".format(1, body_data["betas"].shape[0]))
            print("Gender {}".format(body_data["gender"]))
            print("FPS {}".format(body_data["mocap_framerate"]))

        sf = start_frame or 0
        ef = end_frame or body_data["poses"].shape[0]
        poses = body_data["poses"][sf:ef]
        trans = body_data["trans"][sf:ef]

        if fps_out is not None:
            fps_in = body_data["mocap_framerate"].tolist()
            if fps_in != fps_out:
                ps = np.reshape(poses, [poses.shape[0], -1, 3])
                ps_new = resample_rotations(ps, fps_in, fps_out)
                poses = np.reshape(ps_new, [-1, poses.shape[1]])
                trans = resample_positions(trans, fps_in, fps_out)

        i_root_end = 3
        i_body_end = i_root_end + smpl_layer.bm.NUM_BODY_JOINTS * 3
        i_left_hand_end = i_body_end + smpl_layer.bm.NUM_HAND_JOINTS * 3
        i_right_hand_end = i_left_hand_end + smpl_layer.bm.NUM_HAND_JOINTS * 3

        return cls(
            poses_body=poses[:, i_root_end:i_body_end],
            poses_root=poses[:, :i_root_end],
            poses_left_hand=poses[:, i_body_end:i_left_hand_end],
            poses_right_hand=poses[:, i_left_hand_end:i_right_hand_end],
            smpl_layer=smpl_layer,
            betas=body_data["betas"][np.newaxis],
            trans=trans,
            z_up=z_up,
            **kwargs,
        )

    @classmethod
    def from_3dpw(cls, pkl_data_path, **kwargs):
        """Load a 3DPW sequence which might contain multiple people."""
        with open(pkl_data_path, "rb") as p:
            body_data = pkl.load(p, encoding="latin1")
        num_people = len(body_data["poses"])

        name = kwargs.get("name", "3DPW")

        seqs = []
        for i in range(num_people):
            gender = body_data["genders"][i]
            smpl_layer = SMPLLayer(
                model_type="smpl",
                gender="female" if gender == "f" else "male",
                device=C.device,
                num_betas=10,
            )

            # Extract the 30 Hz data that is already aligned with the image data.
            poses = body_data["poses"][i]
            trans = body_data["trans"][i]
            betas = body_data["betas"][i]

            if len(betas.shape) == 1:
                betas = betas[np.newaxis]

            poses_body = poses[:, 3:]
            poses_root = poses[:, :3]
            trans_root = trans

            kwargs["name"] = name + " S{}".format(i)
            seq = cls(
                poses_body=poses_body,
                poses_root=poses_root,
                trans=trans_root,
                smpl_layer=smpl_layer,
                betas=betas,
                **kwargs,
            )
            seqs.append(seq)

        # Load camera poses.
        camera_data = {
            "intrinsics": body_data["cam_intrinsics"],
            "extrinsics": body_data["cam_poses"],
            "campose_valid": body_data["campose_valid"],
        }

        return seqs, camera_data

    @classmethod
    def t_pose(cls, smpl_layer=None, betas=None, frames=1, **kwargs):
        """Creates a SMPL sequence whose single frame is a SMPL mesh in T-Pose."""

        if smpl_layer is None:
            smpl_layer = SMPLLayer(model_type="smplh", gender="neutral")

        poses = np.zeros([frames, smpl_layer.bm.NUM_BODY_JOINTS * 3])  # including hands and global root
        return cls(poses, smpl_layer, betas=betas, **kwargs)

    @classmethod
    def from_npz(cls, file: Union[IO, str], smpl_layer: SMPLLayer = None, **kwargs):
        """Creates a SMPL sequence from a .npz file exported through the 'export' function."""
        if smpl_layer is None:
            smpl_layer = SMPLLayer(model_type="smplh", gender="neutral")

        data = np.load(file)

        return cls(
            smpl_layer=smpl_layer,
            poses_body=data["poses_body"],
            poses_root=data["poses_root"],
            betas=data["betas"],
            trans=data["trans"],
            **kwargs,
        )

    def export_to_npz(self, file: Union[IO, str]):
        np.savez(
            file,
            poses_body=c2c(self.poses_body),
            poses_root=c2c(self.poses_root),
            betas=c2c(self.betas),
            trans=c2c(self.trans),
        )

    @property
    def color(self):
        return self.mesh_seq.color

    @color.setter
    def color(self, color):
        self.mesh_seq.color = color

    @property
    def bounds(self):
        return self.mesh_seq.bounds

    @property
    def current_bounds(self):
        return self.mesh_seq.current_bounds

    @property
    def vertex_normals(self):
        return self.mesh_seq.vertex_normals

    @property
    def poses(self):
        return torch.cat((self.poses_root, self.poses_body), dim=-1)

    @property
    def _edit_mode(self):
        return self.selected_mode == "edit"

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

            poses_left_hand = (
                None if self.poses_left_hand is None else self.poses_left_hand[self.current_frame_id][None, :]
            )
            poses_right_hand = (
                None if self.poses_right_hand is None else self.poses_right_hand[self.current_frame_id][None, :]
            )
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

        verts, joints = self.smpl_layer(
            poses_root=poses_root,
            poses_body=poses_body,
            poses_left_hand=poses_left_hand,
            poses_right_hand=poses_right_hand,
            betas=betas,
            trans=trans,
        )

        # Apply post_fk_func if specified.
        if self.post_fk_func:
            verts, joints = self.post_fk_func(self, verts, joints, current_frame_only)

        skeleton = self.smpl_layer.skeletons()["body"].T
        faces = self.smpl_layer.bm.faces.astype(np.int64)
        joints = joints[:, : skeleton.shape[0]]

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
                pose = torch.cat(
                    [
                        self.poses_root[self.current_frame_id],
                        self.poses_body[self.current_frame_id],
                    ],
                    dim=-1,
                )

            # Update rigid bodies.
            if self.smpl_layer.model_type != "flame":
                global_oris = local_to_global(pose, self.skeleton[:, 0], output_format="rotmat")
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
            if self.smpl_layer.model_type != "flame":
                global_oris = local_to_global(
                    torch.cat([poses_root, poses_body], dim=-1),
                    self.skeleton[:, 0],
                    output_format="rotmat",
                )
                global_oris = global_oris.reshape((self.n_frames, -1, 3, 3))
                self.rbs.rb_ori = c2c(global_oris)
            self.rbs.rb_pos = self.joints

            # Update mesh
            self.mesh_seq.vertices = vertices

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
            self.mesh_seq.backface_fragmap = True
            self.rbs.color = (1, 0, 0.5, 1.0)
            self._view_mode_color = self.mesh_seq.color
            self.mesh_seq.color = (
                *self._view_mode_color[:3],
                min(self._view_mode_color[3], 0.5),
            )
        else:
            self.mesh_seq.backface_fragmap = False
            self.mesh_seq.color = self._view_mode_color

            self.rbs.color = (0, 1, 0.5, 1.0)
            self.rbs.enabled = self._view_mode_joint_angles
            self.rbs.is_selectable = True

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
            e = imgui.tree_node(f"{j} - {name}")
        else:
            e = True
            imgui.text(f"{j} - {name}")

        if e:
            # Euler angles sliders.
            aa = self._edit_pose[j * 3 : (j + 1) * 3].cpu().numpy()
            euler = aa2euler_numpy(aa, degrees=True)

            _, self._edit_local_axes = imgui.checkbox("Local axes", self._edit_local_axes)

            # If we are editing local axes generate an empty slider on top
            # of the euler angle sliders to capture the input of the slider
            # without modifying the euler angle values.
            if self._edit_local_axes:
                # Get the current draw position.
                pos = imgui.get_cursor_position()

                # Make the next widget transparent.
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.0)
                u, new_euler = imgui.drag_float3(f"", 0, 0, 0, 0.003, format="")
                imgui.pop_style_var()

                if u:
                    base = Rotation.from_rotvec(aa)
                    for i in range(3):
                        delta = new_euler[i]
                        if delta == 0:
                            continue

                        # Get the world coordinates of the current axis from the
                        # respective column of the rotation matrix.
                        axis = Rotation.as_matrix(base)[:, i]

                        # Create a rotation of 'delta[i]' radians around the axis.
                        rot = Rotation.from_rotvec(axis * delta)

                        # Rotate the current joint and convert back to axis angle.
                        aa = Rotation.as_rotvec(rot * base)

                        self._edit_pose[j * 3 : (j + 1) * 3] = torch.from_numpy(aa)
                        self._edit_pose_dirty = True
                        self.redraw(current_frame_only=True)

                # Reset the draw position so that the next slider is drawn on top of this.
                imgui.set_cursor_pos(pos)

            name = "Local XYZ" if self._edit_local_axes else "Euler XYZ"
            u, euler = imgui.drag_float3(f"{name}##joint{j}", *euler, 0.1, format="%.3f")
            if not self._edit_local_axes and u:
                aa = euler2aa_numpy(np.array(euler), degrees=True)
                self._edit_pose[j * 3 : (j + 1) * 3] = torch.from_numpy(aa)
                self._edit_pose_dirty = True
                self.redraw(current_frame_only=True)

            if tree:
                for c in tree.get(j, []):
                    self._gui_joint(imgui, c, tree)
                imgui.tree_pop()

    def gui_mode_edit(self, imgui):
        skel = self.smpl_layer.skeletons()["body"].cpu().numpy()

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
                self.selected_mode = "view"
                imgui.close_current_popup()
            if imgui.radio_button("Edit mode", self.edit_mode):
                self.selected_mode = "edit"
                imgui.close_current_popup()

            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            super().gui_context_menu(imgui)

    def on_selection(self, node, instance_id, tri_id):
        if self.edit_mode:
            # Index of the joint that is currently being edited.
            if node != self.mesh_seq:
                self._edit_joint = instance_id
                self.rbs.color_one(self._edit_joint, (0.3, 0.4, 1, 1))
            else:
                self._edit_joint = None
                # Reset color of all spheres to the default color
                self.rbs.color = self.rbs.color

    def render_outline(self, *args, **kwargs):
        # Only render outline of the mesh, skipping skeleton and rigid bodies.
        self.mesh_seq.render_outline(*args, **kwargs)

    def add_frames(self, poses_body, poses_root=None, trans=None, betas=None):
        # Append poses_body.
        if len(poses_body.shape) == 1:
            poses_body = poses_body[np.newaxis]
        self.poses_body = torch.cat((self.poses_body, to_torch(poses_body, self.dtype, self.device)))

        # Append poses_root or zeros.
        if poses_root is None:
            poses_root = torch.zeros([len(poses_body), 3])
        elif len(poses_root.shape) == 1:
            poses_root = poses_root[np.newaxis]
        self.poses_root = torch.cat((self.poses_root, to_torch(poses_root, self.dtype, self.device)))

        # Append trans or zeros.
        if trans is None:
            trans = torch.zeros([len(poses_body), 3])
        elif len(trans.shape) == 1:
            trans = trans[np.newaxis]
        self.trans = torch.cat((self.trans, to_torch(trans, self.dtype, self.device)))

        # Append betas or zeros .
        if betas is None:
            # If we have only 1 frame of betas we don't need to append zeros, as the first
            # frame of betas will be broadcasted to all frames.
            if betas.shape[0] > 1:
                self.betas = torch.cat(
                    (
                        self.betas,
                        to_torch(
                            torch.zeros([1, self.smpl_layer.num_betas]),
                            self.dtype,
                            self.device,
                        ),
                    )
                )
        else:
            if len(betas.shape) == 1:
                betas = betas[np.newaxis]
            self.betas = torch.cat((self.betas, to_torch(betas, self.dtype, self.device)))

        self.n_frames = len(self.poses_body)
        self.redraw()

    def update_frames(self, poses_body, frames, poses_root=None, trans=None, betas=None):
        self.poses_body[frames] = to_torch(poses_body, self.dtype, self.device)
        if poses_root is not None:
            self.poses_root[frames] = to_torch(poses_root, self.dtype, self.device)
        if trans is not None:
            self.trans[frames] = to_torch(trans, self.dtype, self.device)
        if betas is not None:
            self.betas[frames] = to_torch(betas, self.dtype, self.device)
        self.redraw()

    def remove_frames(self, frames):
        frames_to_keep = torch.from_numpy(np.setdiff1d(np.arange(self.n_frames), frames)).to(
            dtype=torch.long, device=self.device
        )

        self.poses_body = self.poses_body[frames_to_keep]
        self.poses_root = self.poses_root[frames_to_keep]
        self.trans = self.trans[frames_to_keep]
        if self.betas.shape != 1:
            self.betas = self.betas[frames_to_keep]

        self.n_frames = len(self.poses_body)
        self.redraw()

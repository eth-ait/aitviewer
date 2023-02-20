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
import collections
from abc import ABC

import numpy as np
import smplx
import torch
import torch.nn as nn

from aitviewer.configuration import CONFIG as C
from aitviewer.utils.so3 import aa2rot_torch as aa2rot
from aitviewer.utils.so3 import rot2aa_torch as rot2aa
from aitviewer.utils.utils import compute_vertex_and_face_normals_torch


class SMPLLayer(nn.Module, ABC):
    """A wrapper for the various SMPL body models."""

    def __init__(
        self,
        model_type="smpl",
        gender="neutral",
        num_betas=10,
        device=None,
        dtype=None,
        **smpl_model_params,
    ):
        """
        Initializer.
        :param model_type: Which type of SMPL model to load, currently SMPL, SMPL-H and SMPL-X are supported.
        :param gender: Which gender to load.
        :param num_betas: Number of shape components.
        :param device: CPU or GPU.
        :param dtype: The pytorch floating point data type.
        :param smpl_model_params: Other keyword arguments that can be passed to smplx.create.
        """
        assert model_type in ["smpl", "smplh", "smplx", "mano", "flame"]
        assert gender in ["male", "female", "neutral"]
        if model_type == "smplh" and gender == "neutral":
            gender = "female"  # SMPL-H has no neutral gender.

        super(SMPLLayer, self).__init__()
        self.num_betas = num_betas

        smpl_model_params["use_pca"] = smpl_model_params.get("use_pca", False)
        smpl_model_params["flat_hand_mean"] = smpl_model_params.get("flat_hand_mean", True)

        self.bm = smplx.create(
            C.smplx_models,
            model_type=model_type,
            num_betas=self.num_betas,
            gender=gender,
            **smpl_model_params,
        )

        if device is None:
            device = C.device
        if dtype is None:
            dtype = C.f_precision
        self.bm.to(device=device, dtype=dtype)

        self.model_type = model_type
        self._parents = None
        self._children = None
        self._closest_joints = None
        self._vertex_faces = None
        self._faces = None

    @property
    def faces(self):
        """Return the definition of the faces."""
        if self._faces is None:
            self._faces = torch.from_numpy(self.bm.faces.astype(np.int32))
        return self._faces

    @property
    def parents(self):
        """Return how the joints are connected in the kinematic chain where parents[i, 0] is the parent of
        joint parents[i, 1]."""
        if self._parents is None:
            self._parents = self.bm.kintree_table.transpose(0, 1).cpu().numpy()
        return self._parents

    @property
    def joint_children(self):
        """Return the children of each joint in the kinematic chain."""
        if self._children is None:
            self._children = collections.defaultdict(list)
            for bone in self.parents:
                if bone[0] != -1:
                    self._children[bone[0]].append(bone[1])
        return self._children

    def vertex_faces(self, vertices):
        """Return a matrix that returns a list of faces each vertex is contributing to. `vertices` should have
        have shape (V, 3)."""
        if self._vertex_faces is None:
            import trimesh

            mesh = trimesh.Trimesh(vertices.detach().cpu().numpy(), self.faces.cpu().numpy(), process=False)
            self._vertex_faces = torch.from_numpy(np.copy(mesh.vertex_faces)).to(
                dtype=torch.long, device=vertices.device
            )
        return self._vertex_faces

    def vertex_normals(self, vertices, output_vertex_ids=None):
        """
        Return the unnormalized vertex normals at the provided vertex IDs.
        :param vertices: A tensor of shape (N, V, 3).
        :param output_vertex_ids: An optional list of integers indexing into the 2nd dimension of `vertices`.
        :return: A tensor of shape (N, V', 3) where V' is either V or len(output_vertex_ids).
        """
        normals, _ = compute_vertex_and_face_normals_torch(vertices, self.faces, self.vertex_faces(vertices[0]))
        if output_vertex_ids is not None:
            return normals[:, output_vertex_ids]
        else:
            return normals

    def skeletons(self):
        """Return how the joints are connected in the kinematic chain where skeleton[0, i] is the parent of
        joint skeleton[1, i]."""
        parents = torch.stack(
            [
                self.bm.parents,
                torch.arange(0, len(self.bm.parents), device=self.bm.parents.device),
            ]
        )
        return {
            "all": parents,
            "body": parents[:, : self.bm.NUM_BODY_JOINTS + 1],
            "hands": parents[:, self.bm.NUM_BODY_JOINTS + 1 :],
        }

    def fk(
        self,
        poses_body,
        betas,
        poses_root=None,
        trans=None,
        normalize_root=False,
        poses_left_hand=None,
        poses_right_hand=None,
        poses_jaw=None,
        poses_leye=None,
        poses_reye=None,
        expression=None,
    ):
        """
        Convert body pose data (joint angles and shape parameters) to positional data (joint and mesh vertex positions).
        :param poses_body: A tensor of shape (N, N_JOINTS*3), i.e. joint angles in angle-axis format. This contains all
          body joints which are not the root, i.e. possibly including hands and face depending on the underlying body
          model.
        :param betas: A tensor of shape (N, N_BETAS) containing the betas/shape parameters, i.e. shape parameters can
          differ for every sample. If N_BETAS > self.num_betas, the excessive shape parameters will be ignored.
        :param poses_root: Orientation of the root or None. If specified expected shape is (N, 3).
        :param trans: Translation that is applied to vertices and joints or None, this is the 'transl' parameter
          of the SMPL Model. If specified expected shape is (N, 3).
        :param normalize_root: If set, it will normalize the root such that its orientation is the identity in the
          first frame and its position starts at the origin.
        :param poses_left_hand: A tensor of shape (N, N_JOINTS_HANDS*3) or None. Only relevant if this body model
          supports hands.
        :param poses_right_hand: A tensor of shape (N, N_JOINTS_HANDS*3) or None. Only relevant if this body model
          supports hands.
        :param poses_jaw: A tensor of shape (N, 3) or None. Only relevant if this body model supports faces.
        :param poses_leye: A tensor of shape (N, 3) or None. Only relevant if this body model supports faces.
        :param poses_reye: A tensor of shape (N, 3) or None. Only relevant if this body model supports faces.
        :param expression: A tensor of shape (N, N_EXPRESSIONS) or None. Only relevant if this body model supports
          facial expressions.
        :return: The resulting vertices and joints.
        """
        assert poses_body.shape[1] == self.bm.NUM_BODY_JOINTS * 3

        has_hands = hasattr(self.bm, "NUM_HAND_JOINTS")
        has_face = hasattr(self.bm, "NUM_FACE_JOINTS")
        if has_hands:
            if self.bm.use_pca:
                dof_per_hand = self.bm.num_pca_comps
                assert poses_left_hand is None or poses_left_hand.shape[1] == dof_per_hand
                assert poses_right_hand is None or poses_right_hand.shape[1] == dof_per_hand
            else:
                dof_per_hand = self.bm.NUM_HAND_JOINTS * 3
                assert poses_left_hand is None or poses_left_hand.shape[1] == dof_per_hand
                assert poses_right_hand is None or poses_right_hand.shape[1] == dof_per_hand
        else:
            dof_per_hand = 0  # Silencing the might not be initialized warning.

        batch_size = poses_body.shape[0]
        device = poses_body.device

        if poses_root is None:
            poses_root = torch.zeros([batch_size, 3]).to(dtype=poses_body.dtype, device=device)
        if trans is None:
            trans = torch.zeros([batch_size, 3]).to(dtype=poses_body.dtype, device=device)

        if has_hands and poses_left_hand is None:
            poses_left_hand = torch.zeros([batch_size, dof_per_hand]).to(dtype=poses_body.dtype, device=device)
        if has_hands and poses_right_hand is None:
            poses_right_hand = torch.zeros([batch_size, dof_per_hand]).to(dtype=poses_body.dtype, device=device)
        if has_face and poses_jaw is None:
            poses_jaw = torch.zeros([batch_size, 3]).to(dtype=poses_body.dtype, device=device)
        if has_face and poses_leye is None:
            poses_leye = torch.zeros([batch_size, 3]).to(dtype=poses_body.dtype, device=device)
        if has_face and poses_reye is None:
            poses_reye = torch.zeros([batch_size, 3]).to(dtype=poses_body.dtype, device=device)
        if has_face and expression is None:
            expression = torch.zeros([batch_size, self.bm.num_expression_coeffs]).to(
                dtype=poses_body.dtype, device=device
            )

        # Batch shapes if they don't match batch dimension.
        if len(betas.shape) == 1 or betas.shape[0] == 1:
            betas = betas.repeat(poses_body.shape[0], 1)
        betas = betas[:, : self.num_betas]

        if normalize_root:
            # Make everything relative to the first root orientation.
            root_ori = aa2rot(poses_root)
            first_root_ori = torch.inverse(root_ori[0:1])
            root_ori = torch.matmul(first_root_ori, root_ori)
            poses_root = rot2aa(root_ori)
            trans = torch.matmul(first_root_ori.unsqueeze(0), trans.unsqueeze(-1)).squeeze()
            trans = trans - trans[0:1]

        output = self.bm(
            body_pose=poses_body,
            betas=betas,
            global_orient=poses_root,
            transl=trans,
            left_hand_pose=poses_left_hand,
            right_hand_pose=poses_right_hand,
            jaw_pose=poses_jaw,
            leye_pose=poses_leye,
            reye_pose=poses_reye,
            expression=expression,
        )

        return output.vertices, output.joints

    def forward(self, *args, **kwargs):
        """
        Forward pass using forward kinematics
        """
        return self.fk(*args, **kwargs)

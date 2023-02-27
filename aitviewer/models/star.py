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
import os

import torch
import torch.nn.functional as F

try:
    from star.config import cfg
    from star.pytorch.star import STAR
except Exception as e:
    raise ImportError(f"Cannot import STAR. Please run `pip install git+https://github.com/ahmedosman/STAR.git`\n{e}")

from aitviewer.configuration import CONFIG as C
from aitviewer.utils.so3 import aa2rot_torch as aa2rot
from aitviewer.utils.so3 import rot2aa_torch as rot2aa


class STARLayer(STAR):
    """Wraps the publicly available STAR model to match SMPLX model interface"""

    def __init__(self, gender="male", num_betas=300, device=None, dtype=None):
        """
        Initializer.
        :param gender: Which gender to load.
        :param num_betas: Number of shape components. Keep in mind STAR and SMPL shape components are not the same
        (i.e. the first 10 components of STAR cannot simply be assigned from the first 10 components of SMPL)
        :param device: CPU or GPU.
        :param dtype: The pytorch floating point data type.
        """
        # Configure STAR model before initializing
        cfg.data_type = dtype if dtype is not None else C.f_precision
        cfg.path_male_star = os.path.join(C.star_models, "male/model.npz")
        cfg.path_female_star = os.path.join(C.star_models, "female/model.npz")
        cfg.path_neutral_star = os.path.join(C.star_models, "neutral/model.npz")

        super(STARLayer, self).__init__(gender=gender, num_betas=num_betas)

        self.device = device if device is not None else C.device
        self.model_type = "star"
        self._parents = None
        self._children = None

    @property
    def parents(self):
        """Return how the joints are connected in the kinematic chain where parents[i, 0] is the parent of
        joint parents[i, 1]."""
        if self._parents is None:
            self._parents = self.kintree_table.transpose(0, 1).cpu().numpy()
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

    def skeletons(self):
        """Return how the joints are connected in the kinematic chain where skeleton[0, i] is the parent of
        joint skeleton[1, i]."""
        kintree_table = self.kintree_table
        kintree_table[:, 0] = -1
        return {
            "all": kintree_table,
            "body": kintree_table[:, : self.n_joints_body + 1],
        }

    @property
    def n_joints_body(self):
        return self.parent.shape[0]

    @property
    def n_joints_total(self):
        return self.n_joints_body + 1

    def forward(self, poses_body, betas=None, poses_root=None, trans=None, normalize_root=False):
        """
        forwards the model
        :param poses_body: Pose parameters.
        :param poses_root: Pose parameters for the root joint.
        :param beta: Beta parameters.
        :param trans: Root translation.
        :param normalize_root: Makes poses relative to the root joint (useful for globally rotated captures).
        :return: Deformed surface vertices, transformed joints
        """
        poses, betas, trans = self.preprocess(poses_body, betas, poses_root, trans, normalize_root)

        # STAR repo currently hardcodes floats.
        v = super().forward(pose=poses.float(), betas=betas.float(), trans=trans.float())
        J = v.J_transformed
        return v, J

    def preprocess(self, poses_body, betas=None, poses_root=None, trans=None, normalize_root=False):
        batch_size = poses_body.shape[0]

        if poses_root is None:
            poses_root = torch.zeros([batch_size, 3]).to(dtype=poses_body.dtype, device=self.device)
        if trans is None:
            # If we don't supply the root translation explicitly, it falls back to using self.bm.trans
            # which might not be zero since it is a trainable param that can get updated.
            trans = torch.zeros([batch_size, 3]).to(dtype=poses_body.dtype, device=self.device)

        if normalize_root:
            # Make everything relative to the first root orientation.
            root_ori = aa2rot(poses_root)
            first_root_ori = torch.inverse(root_ori[0:1])
            root_ori = torch.matmul(first_root_ori, root_ori)
            poses_root = rot2aa(root_ori)
            trans = torch.matmul(first_root_ori.unsqueeze(0), trans.unsqueeze(-1)).squeeze()
            trans = trans - trans[0:1]

        poses = torch.cat((poses_root, poses_body), dim=1)

        if betas is None:
            betas = torch.zeros([batch_size, self.num_betas]).to(dtype=poses_body.dtype, device=self.device)

        # Batch shapes if they don't match batch dimension.
        if betas.shape[0] != batch_size:
            betas = betas.repeat(batch_size, 1)

        # Lower bound betas
        if betas.shape[1] < self.num_betas:
            betas = torch.nn.functional.pad(betas, [0, self.num_betas - betas.shape[1]])

        # Upper bound betas
        betas = betas[:, : self.num_betas]

        return poses, betas, trans

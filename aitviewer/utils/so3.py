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
import roma
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline


def rot2aa_torch(rotation_matrices):
    """
    Convert rotation matrices to rotation vectors (angle-axis representation).
    :param rotation_matrices: A torch tensor of shape (..., 3, 3).
    :return: A torch tensor of shape (..., 3).
    """
    assert isinstance(rotation_matrices, torch.Tensor)
    return roma.rotmat_to_rotvec(rotation_matrices)


def aa2rot_torch(rotation_vectors):
    """
    Convert rotation vectors (angle-axis representation) to rotation matrices.
    :param rotation_vectors: A torch tensor of shape (..., 3).
    :return: A torch tensor of shape (..., 3, 3).
    """
    assert isinstance(rotation_vectors, torch.Tensor)
    return roma.rotvec_to_rotmat(rotation_vectors)


def rot2aa_numpy(rotation_matrices):
    """
    Convert rotation matrices to rotation vectors (angle-axis representation).
    :param rotation_matrices: A numpy array of shape (..., 3, 3).
    :return: A numpy array of shape (..., 3).
    """
    assert isinstance(rotation_matrices, np.ndarray)
    ori_shape = rotation_matrices.shape[:-2]
    rots = np.reshape(rotation_matrices, (-1, 3, 3))
    aas = R.as_rotvec(R.from_matrix(rots))
    rotation_vectors = np.reshape(aas, ori_shape + (3,))
    return rotation_vectors


def aa2rot_numpy(rotation_vectors):
    """
    Convert rotation vectors (angle-axis representation) to rotation matrices.
    :param rotation_vectors: A numpy array of shape (..., 3).
    :return: A numpy array of shape (..., 3, 3).
    """
    assert isinstance(rotation_vectors, np.ndarray)
    ori_shape = rotation_vectors.shape[:-1]
    aas = np.reshape(rotation_vectors, (-1, 3))
    rots = R.as_matrix(R.from_rotvec(aas))
    rotation_matrices = np.reshape(rots, ori_shape + (3, 3))
    return rotation_matrices


def euler2aa_numpy(euler_angles, degrees=False):
    """
    Convert euler angles (XYZ order) to rotation vectors (angle-axis representation)
    :param euler_angles: A numpy array of shape (..., 3).
    :param degrees: Must be True if euler_angles are degrees and False if they are radians.
    :return: A numpy array of shape (..., 3).
    """
    assert isinstance(euler_angles, np.ndarray)
    ori_shape = euler_angles.shape[:-1]
    rots = np.reshape(euler_angles, (-1, 3))
    aas = R.as_rotvec(R.from_euler("XYZ", rots, degrees=degrees))
    rotation_vectors = np.reshape(aas, ori_shape + (3,))
    return rotation_vectors


def aa2euler_numpy(rotation_vectors, degrees=False):
    """
    Convert rotation vectors (angle-axis representation) to euler angles (XYZ order)
    :param rotation_vectors: A numpy array of shape (..., 3).
    :param degrees: The return value is in degrees if True and in radians otherwise.
    :return: A numpy array of shape (..., 3).
    """
    assert isinstance(rotation_vectors, np.ndarray)
    ori_shape = rotation_vectors.shape[:-1]
    aas = np.reshape(rotation_vectors, (-1, 3))
    rots = R.as_euler(R.from_rotvec(aas), "XYZ", degrees=degrees)
    euler_angles = np.reshape(rots, ori_shape + (3,))
    return euler_angles


def euler2rot_numpy(euler_angles, degrees=False):
    """
    Convert euler angles (XYZ order) to rotation matrices.
    :param euler_angles: A numpy array of shape (..., 3).
    :param degrees: Must be True if euler_angles are degrees and False if they are radians.
    :return: A numpy array of shape (..., 3, 3).
    """
    assert isinstance(euler_angles, np.ndarray)
    ori_shape = euler_angles.shape[:-1]
    rots = np.reshape(euler_angles, (-1, 3))
    rots = R.as_matrix(R.from_euler("XYZ", rots, degrees=degrees))
    rotation_matrices = np.reshape(rots, ori_shape + (3, 3))
    return rotation_matrices


def rot2euler_numpy(rotation_matrices, degrees=False):
    """
    Convert rotation matrices to euler angles (XYZ order)
    :param rotation_matrices: A numpy array of shape (..., 3, 3).
    :param degrees: The return value is in degrees if True and in radians otherwise.
    :return: A numpy array of shape (..., 3).
    """
    assert isinstance(rotation_matrices, np.ndarray)
    ori_shape = rotation_matrices.shape[:-2]
    rots = np.reshape(rotation_matrices, (-1, 3, 3))
    rots = R.as_euler(R.from_matrix(rots), "XYZ", degrees=degrees)
    euler_angles = np.reshape(rots, ori_shape + (3,))
    return euler_angles


def interpolate_rotations(rotations, ts_in, ts_out):
    """
    Interpolate rotations given at timestamps `ts_in` to timestamps given at `ts_out`. This performs the equivalent
    of cubic interpolation in SO(3).
    :param rotations: A numpy array of rotations of shape (F, N, 3), i.e. rotation vectors.
    :param ts_in: Timestamps corresponding to the given rotations, len(ts_in) == F
    :param ts_out: The desired output timestamps.
    :return: A numpy array of shape (len(ts_out), N, 3).
    """
    out = []
    for j in range(rotations.shape[1]):
        rs = R.from_rotvec(rotations[:, j])
        spline = RotationSpline(ts_in, rs)
        rs_interp = spline(ts_out).as_rotvec()
        out.append(rs_interp[:, np.newaxis])
    return np.concatenate(out, axis=1)


def resample_rotations(rotations, fps_in, fps_out):
    """
    Resample a motion sequence from `fps_in` to `fps_out`.
    :param rotations: A numpy array of shape (F, N, 3), i.e. in angle-axis form.
    :param fps_in: The frequency of the input sequence.
    :param fps_out: The desired frequency of the output sequence.
    :return: A numpy array of shape (F', N, 3) where F is adjusted according to the new fps.
    """
    n_frames = rotations.shape[0]
    assert n_frames > 1, "We need at least two rotations for a resampling to make sense."
    duration = n_frames / fps_in
    ts_in = np.arange(0, duration, 1 / fps_in)[:n_frames]
    ts_out = np.arange(0, duration, 1 / fps_out)
    return interpolate_rotations(rotations, ts_in, ts_out)

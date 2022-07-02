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
import torch
import os
import subprocess

from aitviewer.utils.so3 import aa2rot_torch as aa2rot
from aitviewer.utils.so3 import rot2aa_torch as rot2aa
from scipy.interpolate import CubicSpline


def to_torch(x, dtype, device):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype, device=device)
    return x.to(dtype=dtype, device=device)


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def images_to_video(frame_dir, video_path, frame_format='frame_%06d.png', input_fps=60, output_fps=60,
                    start_frame=0):
    """Convert the rendered images into a video. The video path format determines whether this will be rendered as
    a GIF or an MP4 (default)."""
    if not os.path.exists(frame_dir):
        raise ValueError("Could not find directory containing frames {}".format(frame_dir))

    is_mp4 = video_path.endswith('.mp4')
    is_gif = video_path.endswith('.gif')
    if not (is_mp4 or is_gif):
        video_path += '.mp4'
        is_mp4 = True

    suffix = '.gif' if is_gif else '.mp4'

    dir_of_file = os.path.dirname(os.path.abspath(video_path))
    if not os.path.exists(dir_of_file):
        os.makedirs(dir_of_file)

    # Make sure we don't override an existing video.
    counter = 0
    video_path_candidate = video_path.replace(suffix, '_{}{}'.format(counter, suffix))
    while os.path.exists(video_path_candidate):
        counter += 1
        video_path_candidate = video_path.replace(suffix, '_{}{}'.format(counter, suffix))

    print("Rendering to video {}".format(os.path.abspath(video_path_candidate)))

    if is_mp4:
        command = ['ffmpeg',
                   '-framerate', str(input_fps),  # must be this early in the command, otherwise it is not applied
                   '-start_number', str(start_frame),
                   '-i', os.path.join(frame_dir, frame_format),
                   '-c:v', 'libx264',
                   '-preset', 'slow',
                   '-profile:v', 'high',
                   '-level:v', '4.0',
                   '-pix_fmt', 'yuv420p',
                   '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',  # Avoid error when image res is not divisible by 2.
                   '-r', str(output_fps),
                   '-y',
                   video_path_candidate]
    elif is_gif:
        command = ['ffmpeg',
                   '-start_number', str(start_frame),
                   '-framerate', str(input_fps),  # must be this early in the command, otherwise it is not applied
                   '-loglevel', 'panic',
                   '-i', os.path.join(frame_dir, frame_format),
                   '-y',
                   '-filter_complex', "[0:v] fps=15,scale=w=1080:h=-1,split [a][b];[a] palettegen=stats_mode=single [p];[b][p] paletteuse=new=1",
                   '-r', str(output_fps),
                   video_path_candidate]
    else:
        print("Unknown video format, only '.mp4' or '.gif' is supported.")
        return

    FNULL = open(os.devnull, 'w')
    subprocess.Popen(command, stdout=FNULL).wait()
    FNULL.close()


def interpolate_positions(positions, ts_in, ts_out):
    """
    Interpolate positions given at timestamps `ts_in` to timestamps given at `ts_out` with a cubic spline.
    :param positions: A numpy array of shape (F, N, 3), i.e. in angle-axis form.
    :param ts_in: Timestamps corresponding to the given positions, len(ts_in) == F
    :param ts_out: The desired output timestamps.
    :return: A numpy array of shape (len(ts_out), N, 3).
    """
    cs = CubicSpline(ts_in, positions, axis=0)
    new_positions = cs(ts_out)
    return new_positions


def resample_positions(positions, fps_in, fps_out):
    """
    Resample 3D positions from `fps_in` to `fps_out`.
    :param positions: A numpy array of shape (F, ...).
    :param fps_in: The frequency of the input sequence.
    :param fps_out: The desired output frequency.
    :return: A numpy array of shape (F', ...) where F is adjusted according to the new fps.
    """
    n_frames = positions.shape[0]
    assert n_frames > 1, "Resampling with one data point does not make sense."
    duration = n_frames / fps_in
    ts_in = np.arange(0, duration, 1 / fps_in)[:n_frames]
    ts_out = np.arange(0, duration, 1 / fps_out)
    return interpolate_positions(positions, ts_in, ts_out)


def compute_vertex_and_face_normals_torch(vertices, faces, vertex_faces, normalize=False):
    """
    Compute (unnormalized) vertex normals for the given vertices.
    :param vertices: A tensor of shape (N, V, 3).
    :param faces: A tensor of shape (F, 3) indexing into `vertices`.
    :param vertex_faces: A tensor of shape (V, MAX_VERTEX_DEGREE) that lists the face IDs each vertex is a part of.
    :return: The vertex and face normals as tensors of shape (N, V, 3) and (N, F, 3) respectively.
    """
    vs = vertices[:, faces.to(dtype=torch.long)]
    face_normals = torch.cross(vs[:, :, 1] - vs[:, :, 0], vs[:, :, 2] - vs[:, :, 0], dim=-1)  # (N, F, 3)

    ns_all_faces = face_normals[:, vertex_faces]  # (N, V, MAX_VERTEX_DEGREE, 3)
    ns_all_faces[:, vertex_faces == -1] = 0.0
    vertex_degrees = (vertex_faces > -1).sum(dim=-1).to(dtype=ns_all_faces.dtype)
    vertex_normals = ns_all_faces.sum(dim=-2) / vertex_degrees[None, :, None]  # (N, V, 3)

    if normalize:
        face_normals = face_normals / torch.norm(face_normals, dim=-1).unsqueeze(-1)
        vertex_normals = vertex_normals / torch.norm(vertex_normals, dim=-1).unsqueeze(-1)

    return vertex_normals, face_normals


def compute_vertex_and_face_normals(vertices, faces, vertex_faces, normalize=False):
    """
    Compute (unnormalized) vertex normals for the given vertices. This is a rather expensive operation despite it being
    fully optimized with numpy. For a typical SMPL mesh this can take 5-10ms per call. Not normalizing the resulting
    normals speeds things up considerably, so we are omitting this as OpenGL shaders can do this much more efficiently.
    For more speedup, consider caching the result of this function.

    :param vertices: A numpy array of shape (N, V, 3).
    :param faces: A numpy array of shape (F, 3) indexing into `vertices`.
    :param vertex_faces: A numpy array of shape (V, MAX_VERTEX_DEGREE) that lists the face IDs each vertex is a part of.
    :param normalize: Whether to normalize the normals or not.
    :return: The vertex and face normals as a np arrays of shape (N, V, 3) and (N, F, 3) respectively.
    """
    vs = vertices[:, faces]
    face_normals = np.cross(vs[:, :, 1] - vs[:, :, 0], vs[:, :, 2] - vs[:, :, 0], axis=-1)  # (N, F, 3)

    ns_all_faces = face_normals[:, vertex_faces]  # (N, V, MAX_VERTEX_DEGREE, 3)
    ns_all_faces[:, vertex_faces == -1] = 0.0
    vertex_degrees = np.sum(vertex_faces > -1, axis=-1)
    vertex_normals = np.sum(ns_all_faces, axis=-2) / vertex_degrees[np.newaxis, :, np.newaxis]  # (N, V, 3)

    if normalize:
        face_normals = face_normals / np.linalg.norm(face_normals, axis=-1)[..., np.newaxis]
        vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=-1)[..., np.newaxis]

    return vertex_normals, face_normals


def set_lights_in_program(prog, lights):
    """Set program lighting from scene lights"""
    for i, light in enumerate(lights):
        prog['dirLight[{}].pos'.format(i)].value = light.position
        prog['dirLight[{}].color'.format(i)].value = light.color[:3]
        prog['dirLight[{}].intensity_ambient'.format(i)].value = light.intensity_ambient
        prog['dirLight[{}].intensity_diffuse'.format(i)].value = light.intensity_diffuse


def set_material_properties(prog, material):
    prog['diffuse_coeff'].value = material.diffuse
    prog['ambient_coeff'].value = material.ambient


def local_to_global(poses, parents, output_format='aa', input_format='aa'):
    """
    Convert relative joint angles to global ones by unrolling the kinematic chain.
    :param poses: A tensor of shape (N, N_JOINTS*3) defining the relative poses in angle-axis format.
    :param parents: A list of parents for each joint j, i.e. parent[j] is the parent of joint j.
    :param output_format: 'aa' or 'rotmat'.
    :param input_format: 'aa' or 'rotmat'
    :return: The global joint angles as a tensor of shape (N, N_JOINTS*DOF).
    """
    assert output_format in ['aa', 'rotmat']
    assert input_format in ['aa', 'rotmat']
    dof = 3 if input_format == 'aa' else 9
    n_joints = poses.shape[-1] // dof
    if input_format == 'aa':
        local_oris = aa2rot(poses.reshape((-1, 3)))
    else:
        local_oris = poses
    local_oris = local_oris.reshape((-1, n_joints, 3, 3))
    global_oris = torch.zeros_like(local_oris)

    for j in range(n_joints):
        if parents[j] < 0:
            # root rotation
            global_oris[..., j, :, :] = local_oris[..., j, :, :]
        else:
            parent_rot = global_oris[..., parents[j], :, :]
            local_rot = local_oris[..., j, :, :]
            global_oris[..., j, :, :] = torch.matmul(parent_rot, local_rot)

    if output_format == 'aa':
        global_oris = rot2aa(global_oris.reshape((-1, 3, 3)))
        res = global_oris.reshape((-1, n_joints * 3))
    else:
        res = global_oris.reshape((-1, n_joints * 3 * 3))
    return res

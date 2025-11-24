"""Camera trajectory control panel for aitviewer viewers."""

from __future__ import annotations

import os
import time
from datetime import datetime
from glob import glob
from typing import List, Optional, Sequence, Tuple

import imgui
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from scipy.interpolate import make_interp_spline

from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.spheres import Spheres
from aitviewer.scene.camera import ViewerCamera
from aitviewer.utils.utils import get_video_paths, video_to_gif

__all__ = ["CameraTrajectoryPanel"]


# --------------------------------------------------------------------------- #
# Quaternion and vector helpers
# --------------------------------------------------------------------------- #
def _normalize_vector(vec: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def _look_at_rotation(forward: NDArray[np.float64], up_hint: NDArray[np.float64]) -> NDArray[np.float64]:
    f = _normalize_vector(forward)
    if np.linalg.norm(f) < 1e-6:
        f = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up_proj = up_hint - np.dot(up_hint, f) * f
    up = _normalize_vector(up_proj)
    if np.linalg.norm(up) < 1e-6:
        if abs(f[2]) < 0.9:
            up = _normalize_vector(np.cross(f, np.array([0.0, 0.0, 1.0])))
        else:
            up = _normalize_vector(np.cross(f, np.array([1.0, 0.0, 0.0])))
    right = _normalize_vector(np.cross(up, f))
    up = np.cross(f, right)
    return np.stack([right, up, f], axis=1)


def _matrix_to_quaternion(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    m = mat
    tr = np.trace(m)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / S
            x = 0.25 * S
            y = (m[0, 1] + m[1, 0]) / S
            z = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / S
            x = (m[0, 1] + m[1, 0]) / S
            y = 0.25 * S
            z = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / S
            x = (m[0, 2] + m[2, 0]) / S
            y = (m[1, 2] + m[2, 1]) / S
            z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quaternion_multiply(q1: NDArray[np.float64], q2: NDArray[np.float64]) -> NDArray[np.float64]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quaternion_conjugate(q: NDArray[np.float64]) -> NDArray[np.float64]:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64)


def _quaternion_normalize(q: NDArray[np.float64]) -> NDArray[np.float64]:
    return q / np.linalg.norm(q)


def _quaternion_log(q: NDArray[np.float64]) -> NDArray[np.float64]:
    q = _quaternion_normalize(q)
    w = np.clip(q[0], -1.0, 1.0)
    v = q[1:]
    v_norm = np.linalg.norm(v)
    theta = np.arccos(w)
    if v_norm < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return v / v_norm * theta


def _quaternion_exp(v: NDArray[np.float64]) -> NDArray[np.float64]:
    theta = np.linalg.norm(v)
    if theta < 1e-8:
        return np.array([1.0, v[0], v[1], v[2]], dtype=np.float64)
    axis = v / theta
    sin_theta = np.sin(theta)
    return np.array([np.cos(theta), *(axis * sin_theta)], dtype=np.float64)


def _quaternion_slerp(q1: NDArray[np.float64], q2: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    q1 = _quaternion_normalize(q1)
    q2 = _quaternion_normalize(q2)
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return _quaternion_normalize(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    return q1 * s1 + q2 * s2


def _quaternion_rotate(q: NDArray[np.float64], vec: NDArray[np.float64]) -> NDArray[np.float64]:
    q = _quaternion_normalize(q)
    v_quat = np.array([0.0, vec[0], vec[1], vec[2]], dtype=np.float64)
    rotated = _quaternion_multiply(_quaternion_multiply(q, v_quat), _quaternion_conjugate(q))
    return rotated[1:]


def _ensure_quaternion_continuity(quats: Sequence[NDArray[np.float64]]) -> NDArray[np.float64]:
    arr = np.array(quats, dtype=np.float64)
    for i in range(1, len(arr)):
        if np.dot(arr[i - 1], arr[i]) < 0.0:
            arr[i] = -arr[i]
    return arr


def _quaternion_squad(
    q0: NDArray[np.float64],
    s0: NDArray[np.float64],
    s1: NDArray[np.float64],
    q1: NDArray[np.float64],
    t: float,
) -> NDArray[np.float64]:
    slerp1 = _quaternion_slerp(q0, q1, t)
    slerp2 = _quaternion_slerp(s0, s1, t)
    return _quaternion_slerp(slerp1, slerp2, 2 * t * (1 - t))


# --------------------------------------------------------------------------- #
# Panel implementation
# --------------------------------------------------------------------------- #
class CameraTrajectoryPanel:
    """ImGui panel that allows pose management, trajectory playback, and video export."""

    def __init__(
        self,
        viewer,
        pose_dir: str,
        default_steps: int = 30,
        default_fps: float = 24.0,
        video_dir: Optional[str] = None,
        video_fps: float = 30.0,
        play_scene_animation: bool = False,
    ) -> None:
        self.viewer = viewer
        self.pose_dir = pose_dir
        os.makedirs(self.pose_dir, exist_ok=True)
        self.pose_dir_input = self.pose_dir

        self.pose_name = "pose_001"
        self.selected_pose_idx = 0
        self.steps_per_segment = max(1, int(default_steps))
        self.segment_steps: List[int] = []
        self.playback_fps = max(1.0, float(default_fps))
        self.loop_playback = False
        self.path_smooth_iters = 2
        self.segment_duration_input = ""
        self.rotation_interp_mode = "slerp"
        self.position_interp_mode = "linear"

        self.video_dir = video_dir or os.path.join(os.getcwd(), "camera_videos")
        os.makedirs(self.video_dir, exist_ok=True)
        self.video_name = "trajectory.mp4"
        self.video_fps = max(1.0, float(video_fps))
        self.record_scene_animation = bool(play_scene_animation)
        self.rotation_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.lock_rotation_center = False
        self.rotation_use_pointcloud = True
        self.rotation_ray_radius = 0.2
        self.rotation_ray_max_distance = 50.0
        self.rotation_ray_min_distance = 0.05
        self.rotation_ray_steps = 200
        self._pivot_indicator = None
        self._init_pivot_indicator()

        self.pose_files: List[str] = []
        self.refresh_pose_files()

        self._trajectory_positions: Optional[NDArray[np.float32]] = None
        self._trajectory_targets: Optional[NDArray[np.float32]] = None
        self._trajectory_ups: Optional[NDArray[np.float32]] = None
        self._trajectory_fovs: Optional[NDArray[np.float32]] = None
        self._trajectory_frame = 0
        self._playback_active = False
        self._last_time = None

        self._status_message = ""
        self._status_color = (0.8, 0.8, 0.8)
        self._status_timestamp = 0.0
        self._status_duration = 3.0

        self.trajectory_save_path = os.path.join(self.pose_dir, "trajectory.npz")


    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def attach(self) -> None:
        """Register panel so viewer renders it automatically."""
        self.viewer.gui_controls["camera_panel"] = self.render

    def _init_pivot_indicator(self) -> None:
        try:
            indicator = Spheres(
                centers=np.zeros((1, 1, 3), dtype=np.float32),
                radius=0.05,
                color=(1.0, 0.6, 0.1, 1.0),
                name="RotationPivot",
            )
            indicator.enabled = False
            self.viewer.scene.add(indicator)
            self._pivot_indicator = indicator
        except Exception:
            self._pivot_indicator = None

    def refresh_pose_files(self) -> None:
        pattern = os.path.join(self.pose_dir, "*.npz")
        self.pose_files = sorted(glob(pattern))
        if self.pose_files:
            self.selected_pose_idx = min(self.selected_pose_idx, len(self.pose_files) - 1)
        else:
            self.selected_pose_idx = 0
        self._sync_segment_steps(max(0, len(self.pose_files) - 1))

    def _sync_segment_steps(self, num_segments: int) -> None:
        num_segments = max(0, num_segments)
        if len(self.segment_steps) < num_segments:
            self.segment_steps.extend([self.steps_per_segment] * (num_segments - len(self.segment_steps)))
        elif len(self.segment_steps) > num_segments:
            self.segment_steps = self.segment_steps[:num_segments]

    def _current_camera(self) -> Optional[ViewerCamera]:
        cam = getattr(self.viewer.scene, "camera", None)
        if isinstance(cam, ViewerCamera):
            return cam
        return None

    def _set_status(self, message: str, level: str = "info") -> None:
        colors = {
            "info": (0.7, 0.9, 0.7),
            "warn": (0.95, 0.75, 0.45),
            "error": (0.95, 0.45, 0.45),
        }
        self._status_message = message
        self._status_color = colors.get(level, colors["info"])
        self._status_timestamp = time.time()

    def _load_pose(self, path: str) -> dict:
        data = np.load(path)
        pose = {
            "position": data["position"].astype(np.float32),
            "target": data["target"].astype(np.float32),
            "up": data["up"].astype(np.float32),
            "fov": float(data["fov"]),
        }
        data.close()
        return pose

    def _apply_pose(self, pose: dict) -> None:
        camera = self._current_camera()
        if camera is None:
            self._set_status("ViewerCamera required to apply pose", "warn")
            return
        camera.position = pose["position"]
        if hasattr(camera, "target"):
            camera.target = pose["target"]
        camera.up = pose["up"]
        camera.fov = pose["fov"]

    # ------------------------------------------------------------------ #
    # File operations
    # ------------------------------------------------------------------ #
    def save_current_pose(self) -> None:
        camera = self._current_camera()
        if camera is None:
            self._set_status("ViewerCamera required to save pose", "warn")
            return

        pose = {
            "position": np.array(camera.position, dtype=np.float32),
            "target": np.array(camera.target, dtype=np.float32),
            "up": np.array(camera.up, dtype=np.float32),
            "fov": np.array([camera.fov], dtype=np.float32),
        }

        name = self.pose_name.strip() or datetime.now().strftime("pose_%Y%m%d_%H%M%S")
        root, ext = os.path.splitext(name)
        if ext.lower() != ".npz":
            ext = ".npz"
        filename = os.path.join(self.pose_dir, root + ext)
        suffix = 1
        while os.path.exists(filename):
            filename = os.path.join(self.pose_dir, f"{root}_{suffix:02d}{ext}")
            suffix += 1
        np.savez(filename, **pose)
        self.refresh_pose_files()
        self._set_status(f"Saved pose: {os.path.basename(filename)}")

    def load_selected_pose(self) -> None:
        if not self.pose_files:
            self._set_status("No saved poses available", "warn")
            return
        idx = min(max(self.selected_pose_idx, 0), len(self.pose_files) - 1)
        pose = self._load_pose(self.pose_files[idx])
        self._apply_pose(pose)
        self._set_status(f"Loaded pose: {os.path.basename(self.pose_files[idx])}")

    def delete_selected_pose(self) -> None:
        if not self.pose_files:
            self._set_status("No saved poses to delete", "warn")
            return
        idx = min(max(self.selected_pose_idx, 0), len(self.pose_files) - 1)
        path = self.pose_files[idx]
        try:
            os.remove(path)
        except OSError as exc:
            self._set_status(f"Failed to delete pose: {exc}", "error")
            return
        self.refresh_pose_files()
        if self.pose_files:
            self.selected_pose_idx = min(idx, len(self.pose_files) - 1)
        else:
            self.selected_pose_idx = 0
        self._set_status(f"Deleted pose: {os.path.basename(path)}")

    # ------------------------------------------------------------------ #
    # Trajectory logic
    # ------------------------------------------------------------------ #
    def build_trajectory(self) -> None:
        if len(self.pose_files) < 2:
            self._set_status("At least two poses are required to build a trajectory", "warn")
            return

        keyframes = [self._load_pose(f) for f in self.pose_files]
        num_segments = len(keyframes) - 1
        segment_steps = self._parse_segment_steps(num_segments)

        positions = np.asarray([pose["position"] for pose in keyframes], dtype=np.float64)
        targets = np.asarray([pose["target"] for pose in keyframes], dtype=np.float64)
        ups = np.asarray([pose["up"] for pose in keyframes], dtype=np.float64)
        fovs = np.asarray([pose["fov"] for pose in keyframes], dtype=np.float64)

        target_offsets = targets - positions
        view_norm = np.linalg.norm(target_offsets, axis=1)
        view_norm[view_norm < 1e-4] = 1.0

        rotation_mats = []
        for i in range(len(keyframes)):
            forward = target_offsets[i]
            up_hint = ups[i]
            rotation_mats.append(_look_at_rotation(forward, up_hint))
        quats = np.stack([_matrix_to_quaternion(R) for R in rotation_mats], axis=0)
        for i in range(1, len(quats)):
            if np.dot(quats[i - 1], quats[i]) < 0.0:
                quats[i] = -quats[i]

        use_squad = self.rotation_interp_mode.lower() == "squad"
        squad_tangents = self._compute_squad_tangents(quats) if use_squad else None

        traj_positions = self._interpolate_positions(positions, segment_steps)
        traj_distances = [float(view_norm[0])]
        traj_quats = [quats[0].copy()]
        traj_fovs = [float(fovs[0])]

        for seg in range(len(keyframes) - 1):
            dist_a = view_norm[seg]
            dist_b = view_norm[seg + 1]
            fov_a = fovs[seg]
            fov_b = fovs[seg + 1]
            quat_a = quats[seg]
            quat_b = quats[seg + 1]
            seg_steps = segment_steps[seg]
            for step in range(1, seg_steps + 1):
                t = step / seg_steps
                traj_distances.append(float((1.0 - t) * dist_a + t * dist_b))
                traj_fovs.append(float((1.0 - t) * fov_a + t * fov_b))
                if use_squad and squad_tangents is not None:
                    traj_quats.append(
                        _quaternion_squad(
                            quat_a,
                            squad_tangents[seg],
                            squad_tangents[seg + 1],
                            quat_b,
                            t,
                        )
                    )
                else:
                    traj_quats.append(_quaternion_slerp(quat_a, quat_b, t))

        traj_positions = np.asarray(traj_positions, dtype=np.float32)
        traj_distances = np.asarray(traj_distances, dtype=np.float32)
        traj_fovs = np.asarray(traj_fovs, dtype=np.float32)
        traj_quats = np.asarray(traj_quats, dtype=np.float64)

        smooth_iters = max(0, int(self.path_smooth_iters))
        if smooth_iters > 0 and traj_positions.shape[0] > 2:
            traj_positions = self._smooth_vector_sequence(traj_positions, smooth_iters).astype(np.float32)
            traj_distances = self._smooth_vector_sequence(traj_distances, smooth_iters).astype(np.float32)
            traj_fovs = self._smooth_vector_sequence(traj_fovs, smooth_iters).astype(np.float32)
            traj_quats = self._smooth_quaternion_sequence(traj_quats, smooth_iters)
        traj_distances = np.clip(traj_distances, 1e-4, None)

        traj_targets = []
        traj_ups = []
        for idx in range(traj_positions.shape[0]):
            quat = traj_quats[idx]
            forward = _normalize_vector(_quaternion_rotate(quat, np.array([0.0, 0.0, 1.0])))
            up_vec = _normalize_vector(_quaternion_rotate(quat, np.array([0.0, 1.0, 0.0])))
            distance = max(1e-4, traj_distances[idx])
            pos = traj_positions[idx]
            traj_targets.append(pos + forward * distance)
            traj_ups.append(up_vec)

        self._trajectory_positions = traj_positions
        self._trajectory_targets = np.asarray(traj_targets, dtype=np.float32)
        self._trajectory_ups = np.asarray(traj_ups, dtype=np.float32)
        self._trajectory_fovs = traj_fovs.astype(np.float32)
        self._trajectory_frame = 0
        self._playback_active = False

        self._set_status(f"Trajectory built from {len(keyframes)} poses ({len(self._trajectory_positions)} frames)")

    def _smooth_vector_sequence(self, data: NDArray[np.float32], iterations: int) -> NDArray[np.float64]:
        arr = np.asarray(data, dtype=np.float64)
        squeeze = False
        if arr.ndim == 1:
            arr = arr[:, None]
            squeeze = True
        for _ in range(max(0, int(iterations))):
            if arr.shape[0] <= 2:
                break
            padded = np.vstack([arr[0], arr, arr[-1]])
            smoothed = 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]
            smoothed[0] = arr[0]
            smoothed[-1] = arr[-1]
            arr = smoothed
        if squeeze:
            return arr[:, 0]
        return arr

    def _smooth_quaternion_sequence(self, quats: NDArray[np.float64], iterations: int) -> NDArray[np.float64]:
        arr = _ensure_quaternion_continuity(quats)
        for _ in range(max(0, int(iterations))):
            if arr.shape[0] <= 2:
                break
            new_arr = arr.copy()
            for i in range(1, arr.shape[0] - 1):
                left = _quaternion_slerp(arr[i - 1], arr[i], 0.75)
                right = _quaternion_slerp(arr[i], arr[i + 1], 0.25)
                new_arr[i] = _quaternion_slerp(left, right, 0.5)
            arr = _ensure_quaternion_continuity(new_arr)
        return arr

    def _parse_segment_steps(self, num_segments: int) -> List[int]:
        if num_segments <= 0:
            return []
        self._sync_segment_steps(num_segments)
        steps = [max(1, int(s)) for s in self.segment_steps[:num_segments]]
        return steps

    def _interpolate_positions(self, key_positions: NDArray[np.float64], segment_steps: Sequence[int]) -> NDArray[np.float32]:
        total_frames = 1 + sum(segment_steps)
        if total_frames <= 1:
            return np.asarray(key_positions[:1], dtype=np.float32)

        sample_params = [0.0]
        for seg, steps in enumerate(segment_steps):
            for step in range(1, steps + 1):
                sample_params.append(seg + step / steps)
        sample_params = np.array(sample_params, dtype=np.float64)

        mode = self.position_interp_mode.lower()
        if mode == "bspline" and key_positions.shape[0] >= 4:
            try:
                spline = make_interp_spline(np.arange(len(key_positions)), key_positions, k=3, axis=0)
                samples = spline(sample_params)
                return samples.astype(np.float32)
            except Exception as exc:
                self._set_status(f"B-spline interpolation failed ({exc}). Falling back to linear.", "warn")

        traj_positions = [key_positions[0].astype(np.float32)]
        for seg, steps in enumerate(segment_steps):
            a = key_positions[seg]
            b = key_positions[seg + 1]
            for step in range(1, steps + 1):
                t = step / steps
                traj_positions.append(((1.0 - t) * a + t * b).astype(np.float32))
        return np.asarray(traj_positions, dtype=np.float32)

    def _compute_squad_tangents(self, quats: NDArray[np.float64]) -> NDArray[np.float64]:
        quats = _ensure_quaternion_continuity(quats)
        tangents = []
        for i in range(len(quats)):
            q = quats[i]
            if i == 0:
                q_prev = q
            else:
                q_prev = quats[i - 1]
            if i == len(quats) - 1:
                q_next = q
            else:
                q_next = quats[i + 1]
            inv_q = _quaternion_conjugate(q)
            log_left = _quaternion_log(_quaternion_multiply(inv_q, q_prev))
            log_right = _quaternion_log(_quaternion_multiply(inv_q, q_next))
            exp_term = _quaternion_exp(-0.25 * (log_left + log_right))
            tangents.append(_quaternion_multiply(q, exp_term))
        return np.asarray(tangents, dtype=np.float64)

    def _set_rotation_center_from_view(self) -> None:
        camera = self._current_camera()
        if camera is None:
            self._set_status("ViewerCamera required to lock rotation center", "warn")
            return
        ray = self._get_viewport_ray()
        if ray is None:
            self._set_status("Unable to compute viewport ray", "warn")
            return
        origin, direction = ray
        tree_data = self._build_pointcloud_tree()
        if tree_data is None:
            self._set_status("No point cloud available for hit-test", "warn")
            return
        if self.rotation_use_pointcloud:
            hit_point = self._ray_pointcloud_hit(origin, direction, tree_data)
        else:
            hit_point = None
        if hit_point is None:
            fallback = self._closest_point_on_cloud(origin, direction, tree_data)
            if fallback is not None:
                hit_point = fallback
                msg = "Ray missed, using closest cloud point"
            else:
                self._set_status("Failed to determine rotation center", "warn")
                return
        else:
            msg = "Rotation center locked to hit point"
        self.rotation_center = hit_point.astype(np.float32)
        camera.target = self.rotation_center.copy()
        self._update_pivot_indicator()
        self._set_status(msg)

    def _set_rotation_center_scene_center(self) -> None:
        try:
            bounds = self.viewer.scene.bounds_without_floor
        except Exception:
            bounds = None
        if bounds is None:
            self._set_status("Scene bounds unavailable", "warn")
            return
        center = bounds.mean(-1)
        self.rotation_center = center.astype(np.float32)
        self._apply_rotation_center_lock(force=True)
        self._update_pivot_indicator()
        self._set_status("Rotation center snapped to scene center")

    def _apply_rotation_center_lock(self, force: bool = False) -> None:
        if not (self.lock_rotation_center or force):
            return
        cam = self._current_camera()
        if cam is None or self.rotation_center is None:
            return
        cam.target = np.array(self.rotation_center, dtype=np.float32)
        self._update_pivot_indicator()

    def _update_pivot_indicator(self) -> None:
        if self._pivot_indicator is None or self.rotation_center is None:
            return
        centers = np.asarray(self.rotation_center, dtype=np.float32).reshape(1, 1, 3)
        try:
            self._pivot_indicator.sphere_positions = centers
            self._pivot_indicator.enabled = True
            self._pivot_indicator.redraw()
        except Exception:
            pass


    def _ray_pointcloud_hit(
        self,
        origin: NDArray[np.float32],
        direction: NDArray[np.float32],
        tree_data: Optional[Tuple[cKDTree, NDArray[np.float32], NDArray[np.float32]]] = None,
    ) -> Optional[NDArray[np.float32]]:
        if tree_data is None:
            tree_data = self._build_pointcloud_tree()
            if tree_data is None:
                return None
        tree, world_points, _ = tree_data
        steps = max(10, int(self.rotation_ray_steps))
        distances = np.linspace(
            max(0.0, self.rotation_ray_min_distance),
            self.rotation_ray_max_distance,
            steps,
        )
        best_point = None
        best_t = None
        radius = max(1e-4, self.rotation_ray_radius)
        for t in distances:
            sample = origin + direction * t
            dist, idx = tree.query(sample, k=1, distance_upper_bound=radius)
            if np.isinf(dist):
                continue
            candidate = world_points[idx]
            proj = np.dot(candidate - origin, direction)
            if proj < 0 or proj > self.rotation_ray_max_distance:
                continue
            if best_t is None or proj < best_t:
                best_point = candidate
                best_t = proj
                if proj <= self.rotation_ray_min_distance:
                    break
        return None if best_point is None else best_point

    def _closest_point_on_cloud(
        self,
        origin: NDArray[np.float32],
        direction: NDArray[np.float32],
        tree_data: Optional[Tuple[cKDTree, NDArray[np.float32], NDArray[np.float32]]] = None,
    ) -> Optional[NDArray[np.float32]]:
        if tree_data is None:
            tree_data = self._build_pointcloud_tree()
        if tree_data is None:
            return None
        tree, world_points, centroid = tree_data
        dist, idx = tree.query(origin, k=1)
        if np.isinf(dist):
            return centroid
        return world_points[idx]

    def _build_pointcloud_tree(self) -> Optional[Tuple[cKDTree, NDArray[np.float32], NDArray[np.float32]]]:
        try:
            nodes = self.viewer.scene.collect_nodes(obj_type=PointClouds)
        except Exception:
            nodes = []
        if not nodes:
            self._set_status("Scene contains no point clouds for picking", "warn")
            return None
        all_points = []
        for node in nodes:
            if not getattr(node, "enabled", True):
                continue
            pts = getattr(node, "current_points", None)
            if pts is None or pts.size == 0:
                continue
            mat = getattr(node, "model_matrix", None)
            if mat is None:
                world = pts
            else:
                ones = np.ones((pts.shape[0], 1))
                homo = np.hstack([pts, ones])
                world = (mat @ homo.T).T[:, :3]
            all_points.append(world.astype(np.float32))
        if not all_points:
            self._set_status("No pickable points available", "warn")
            return None
        try:
            stacked = np.vstack(all_points)
        except Exception:
            stacked = np.concatenate(all_points, axis=0)
        try:
            tree = cKDTree(stacked)
        except Exception as exc:
            self._set_status(f"Failed to build point cloud index: {exc}", "warn")
            return None
        return tree, stacked, stacked.mean(axis=0)

    def _vector_input(self, label: str, values: NDArray[np.float32]) -> Tuple[bool, NDArray[np.float32]]:
        changed_x, x = imgui.input_float(f"{label} X", float(values[0]))
        changed_y, y = imgui.input_float(f"{label} Y", float(values[1]))
        changed_z, z = imgui.input_float(f"{label} Z", float(values[2]))
        changed = changed_x or changed_y or changed_z
        return changed, np.array([x, y, z], dtype=np.float32)

    def _get_viewport_ray(self) -> Optional[Tuple[NDArray[np.float32], NDArray[np.float32]]]:
        camera = self._current_camera()
        if camera is None or not self.viewer.viewports:
            return None
        viewport = self.viewer.viewports[0]
        w, h = viewport.extents[2], viewport.extents[3]
        if w <= 0 or h <= 0:
            return None
        px = viewport.extents[0] + w / 2
        py = viewport.extents[1] + h / 2
        vx, vy = self.viewer._mouse_to_viewport(int(px), int(py), viewport)
        if vx < 0 or vy < 0 or vx >= w or vy >= h:
            vx = w // 2
            vy = h // 2
        try:
            return camera.get_ray(vx, vy, w, h)
        except Exception:
            return None

    def _resolve_video_path(self) -> str:
        name = (self.video_name or "").strip()
        if not name:
            name = datetime.now().strftime("trajectory_%Y%m%d_%H%M%S.mp4")
        root, ext = os.path.splitext(name)
        if not ext:
            ext = ".mp4"
        return os.path.join(self.video_dir, root + ext)

    def record_trajectory_video(self) -> None:
        if self._trajectory_positions is None or self._trajectory_targets is None:
            self._set_status("Please build a trajectory before exporting", "warn")
            return

        output_path = self._resolve_video_path()
        try:
            final_path = self._render_trajectory_video(output_path)
        except ImportError:
            self._set_status("skvideo is required to export videos", "error")
            return
        except Exception as exc:
            self._set_status(f"Failed to export video: {exc}", "error")
            return

        self._set_status(f"Video saved: {os.path.basename(final_path)}")

    def _render_trajectory_video(self, base_path: str) -> str:
        import skvideo.io
        from tqdm import tqdm

        camera = self._current_camera()
        if camera is None:
            raise RuntimeError("ViewerCamera required to export video")

        path_video, path_gif, is_gif = get_video_paths(base_path, ensure_no_overwrite=True)
        pix_fmt = "yuv420p"
        writer = skvideo.io.FFmpegWriter(
            path_video,
            inputdict={"-framerate": str(self.video_fps)},
            outputdict={
                "-pix_fmt": pix_fmt,
                "-vf": "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-r": str(self.video_fps),
            },
        )

        saved_camera_state = {
            "position": np.array(camera.position),
            "target": np.array(camera.target),
            "up": np.array(camera.up),
            "fov": float(camera.fov),
        }
        saved_frame = self.viewer.scene.current_frame_id
        saved_run_anim = self.viewer.run_animations
        has_last_time = hasattr(self.viewer, "_last_frame_rendered_at")
        saved_last_time = getattr(self.viewer, "_last_frame_rendered_at", 0.0)

        self.viewer.run_animations = False
        dt = 1.0 / self.video_fps
        time_cursor = 0.0
        scene_frame = saved_frame
        n_frames = len(self._trajectory_positions)

        try:
            for idx in tqdm(range(n_frames), desc="Recording trajectory video"):
                camera.position = self._trajectory_positions[idx]
                camera.target = self._trajectory_targets[idx]
                camera.up = self._trajectory_ups[idx]
                camera.fov = float(self._trajectory_fovs[idx])

                if self.record_scene_animation and self.viewer.scene.n_frames > 0:
                    self.viewer.scene.current_frame_id = scene_frame % self.viewer.scene.n_frames
                    scene_frame += 1

                self.viewer.render(time_cursor, time_cursor + dt, export=True, transparent_background=False)
                image = self.viewer.get_current_frame_as_image(alpha=False)
                frame = np.array(image)
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                writer.writeFrame(frame)
                time_cursor += dt
        finally:
            writer.close()
            camera.position = saved_camera_state["position"]
            camera.target = saved_camera_state["target"]
            camera.up = saved_camera_state["up"]
            camera.fov = saved_camera_state["fov"]
            self.viewer.scene.current_frame_id = saved_frame
            self.viewer.run_animations = saved_run_anim
            if has_last_time:
                self.viewer._last_frame_rendered_at = saved_last_time

        if is_gif:
            video_to_gif(path_video, path_gif, remove=False)
            return path_gif
        return path_video

    def start_playback(self) -> None:
        if self._trajectory_positions is None:
            self._set_status("Please build a trajectory first", "warn")
            return
        self._trajectory_frame = 0
        self._playback_active = True
        self._last_time = None
        self._apply_trajectory_frame(force=True)
        self._set_status("Trajectory playback started")

    def stop_playback(self) -> None:
        if self._playback_active:
            self._playback_active = False
            self._set_status("Trajectory playback stopped")

    def save_trajectory(self, path: Optional[str] = None) -> None:
        if self._trajectory_positions is None:
            self._set_status("Build a trajectory before saving", "warn")
            return
        path = path or self.trajectory_save_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            np.savez(
                path,
                positions=self._trajectory_positions,
                targets=self._trajectory_targets,
                ups=self._trajectory_ups,
                fovs=self._trajectory_fovs,
                segment_steps=np.asarray(self.segment_steps, dtype=np.int32),
                pose_files=np.asarray(self.pose_files),
            )
            self._set_status(f"Saved trajectory: {os.path.basename(path)}")
        except Exception as exc:
            self._set_status(f"Failed to save trajectory ({exc})", "error")

    def _apply_trajectory_frame(self, force: bool = False) -> None:
        if self._trajectory_positions is None:
            return
        camera = self._current_camera()
        if camera is None:
            return
        idx = min(self._trajectory_frame, len(self._trajectory_positions) - 1)
        camera.position = self._trajectory_positions[idx]
        camera.target = self._trajectory_targets[idx]
        camera.up = self._trajectory_ups[idx]
        camera.fov = float(self._trajectory_fovs[idx])

    def _update_playback(self) -> None:
        if not self._playback_active or self._trajectory_positions is None:
            return
        now = time.time()
        frame_dt = 1.0 / max(self.playback_fps, 1.0)
        if self._last_time is None:
            self._last_time = now
            self._apply_trajectory_frame()
            return

        advanced = False
        while now - self._last_time >= frame_dt:
            self._last_time += frame_dt
            self._trajectory_frame += 1
            advanced = True
            if self._trajectory_frame >= len(self._trajectory_positions):
                if self.loop_playback:
                    self._trajectory_frame = 0
                else:
                    self._playback_active = False
                    break

        if advanced and self._playback_active:
            self._apply_trajectory_frame()

    # ------------------------------------------------------------------ #
    # GUI
    # ------------------------------------------------------------------ #
    def render(self) -> None:
        self._update_playback()
        self._apply_rotation_center_lock()

        imgui.set_next_window_position(30, 60, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(360, 420, imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("Camera Panel", True)
        if not expanded:
            imgui.end()
            return

        camera = self._current_camera()
        if camera is None:
            imgui.text_colored(
                "Active viewport camera is not a ViewerCamera.\nSwitch to the main camera to continue.",
                1.0,
                0.4,
                0.4,
                1.0,
            )
            imgui.end()
            return

        imgui.text("Pose folder:")
        imgui.same_line()
        changed_folder, self.pose_dir_input = imgui.input_text("##pose_dir_input", self.pose_dir_input, 256)
        imgui.same_line()
        if imgui.button("Set folder"):
            new_dir = self.pose_dir_input.strip()
            if new_dir:
                self.pose_dir = new_dir
                os.makedirs(self.pose_dir, exist_ok=True)
                self.refresh_pose_files()
                self.trajectory_save_path = os.path.join(self.pose_dir, "trajectory.npz")

        imgui.spacing()

        current_fov = float(camera.fov)
        changed, new_fov = imgui.slider_float("FOV (deg)", current_fov, 5.0, 160.0, "%.1f")
        if changed:
            camera.fov = float(new_fov)

        imgui.spacing()
        _, self.pose_name = imgui.input_text("Pose name", self.pose_name, 128)
        if imgui.button("Save current pose"):
            self.save_current_pose()
        imgui.same_line()
        if imgui.button("Refresh list"):
            self.refresh_pose_files()

        imgui.spacing()
        labels = [os.path.basename(f) for f in self.pose_files]
        imgui.columns(2, "poses_segments", False)
        imgui.text("Saved poses")
        imgui.next_column()
        imgui.text("Frames / segment")
        imgui.next_column()
        if labels:
            clicked, idx = imgui.listbox("Saved poses##list", self.selected_pose_idx, labels, min(len(labels), 8))
            if clicked:
                self.selected_pose_idx = idx
        else:
            imgui.text("No poses saved")
        imgui.next_column()
        num_segments = max(0, len(self.pose_files) - 1)
        self._sync_segment_steps(num_segments)
        for seg in range(num_segments):
            label = f"{seg} -> {seg + 1}"
            imgui.text(label)
            imgui.same_line()
            changed, val = imgui.input_int(f"##seg_{seg}", int(self.segment_steps[seg]), step=1, step_fast=5)
            if changed:
                self.segment_steps[seg] = max(1, val)
        imgui.columns(1)

        if labels:
            if imgui.button("Load selected"):
                self.load_selected_pose()
            imgui.same_line()
            if imgui.button("Delete selected"):
                self.delete_selected_pose()

        imgui.spacing()
        _, self.playback_fps = imgui.input_float("Playback FPS", self.playback_fps)
        self.playback_fps = max(1.0, self.playback_fps)
        _, self.path_smooth_iters = imgui.slider_int("Path smoothing iterations", self.path_smooth_iters, 0, 10)
        _, self.loop_playback = imgui.checkbox("Loop trajectory", self.loop_playback)
        pos_modes = [("linear", "Linear"), ("bspline", "B-spline")]
        current_pos_idx = 0
        for idx, (key, _) in enumerate(pos_modes):
            if key == self.position_interp_mode:
                current_pos_idx = idx
                break
        changed_pos_interp, new_pos_idx = imgui.combo("Position interpolation", current_pos_idx, [label for _, label in pos_modes])
        if changed_pos_interp:
            self.position_interp_mode = pos_modes[new_pos_idx][0]
        interp_modes = ["SLERP", "SQUAD"]
        current_interp_idx = 1 if self.rotation_interp_mode.lower() == "squad" else 0
        changed_interp, new_interp_idx = imgui.combo("Rotation interpolation", current_interp_idx, interp_modes)
        if changed_interp:
            self.rotation_interp_mode = interp_modes[new_interp_idx].lower()

        if imgui.button("Build trajectory (all poses)"):
            self.build_trajectory()
        imgui.same_line()
        if imgui.button("Play trajectory"):
            self.start_playback()
        imgui.same_line()
        if imgui.button("Stop"):
            self.stop_playback()
        if imgui.button("Save trajectory"):
            self.save_trajectory()

        if self._trajectory_positions is not None:
            imgui.spacing()
            max_frame = len(self._trajectory_positions) - 1
            changed, idx = imgui.slider_int(
                "Trajectory frame", int(self._trajectory_frame), min_value=0, max_value=max_frame
            )
            if changed:
                self._trajectory_frame = idx
                self._apply_trajectory_frame(force=True)

        imgui.spacing()
        imgui.separator()
        imgui.text(f"Video folder: {self.video_dir}")
        _, self.video_name = imgui.input_text("Video name", self.video_name, 128)
        _, self.video_fps = imgui.input_float("Video FPS", self.video_fps)
        self.video_fps = max(1.0, self.video_fps)
        imgui.text(
            "Recording mode: " + ("Playing animation" if self.record_scene_animation else "Static frame")
        )
        if imgui.button("Toggle recording mode"):
            self.record_scene_animation = not self.record_scene_animation
        if imgui.button("Record trajectory video"):
            self.record_trajectory_video()

        imgui.spacing()
        imgui.separator()
        imgui.text("Rotation center locking")
        current_center = list(self.rotation_center)
        edited = False
        changed_x, current_center[0] = imgui.input_float("Center X", current_center[0])
        changed_y, current_center[1] = imgui.input_float("Center Y", current_center[1])
        changed_z, current_center[2] = imgui.input_float("Center Z", current_center[2])
        if changed_x or changed_y or changed_z:
            self.rotation_center = np.array(current_center, dtype=np.float32)
            edited = True
        if imgui.button("Snap to current view"):
            self._set_rotation_center_from_view()
        imgui.same_line()
        if imgui.button("Use scene center"):
            self._set_rotation_center_scene_center()
        _, self.rotation_use_pointcloud = imgui.checkbox("Use point cloud hit-test", self.rotation_use_pointcloud)
        _, self.rotation_ray_radius = imgui.input_float("Ray radius", self.rotation_ray_radius)
        self.rotation_ray_radius = max(1e-4, self.rotation_ray_radius)
        _, self.rotation_ray_min_distance = imgui.input_float("Ray min distance", self.rotation_ray_min_distance)
        self.rotation_ray_min_distance = max(0.0, self.rotation_ray_min_distance)
        _, self.rotation_ray_max_distance = imgui.input_float("Ray max distance", self.rotation_ray_max_distance)
        self.rotation_ray_max_distance = max(self.rotation_ray_min_distance + 0.1, self.rotation_ray_max_distance)
        _, self.rotation_ray_steps = imgui.slider_int("Ray samples", int(self.rotation_ray_steps), 10, 1000)
        self.rotation_ray_steps = max(10, int(self.rotation_ray_steps))
        prev_lock = self.lock_rotation_center
        _, self.lock_rotation_center = imgui.checkbox("Lock rotation center", self.lock_rotation_center)
        if self.lock_rotation_center and not prev_lock:
            self._apply_rotation_center_lock(force=True)
        if edited and self.lock_rotation_center:
            self._apply_rotation_center_lock(force=True)

        imgui.spacing()
        self._draw_status()
        imgui.end()

    def _draw_status(self) -> None:
        if not self._status_message:
            return
        if time.time() - self._status_timestamp > self._status_duration:
            return
        imgui.text_colored(self._status_message, *self._status_color, 1.0)

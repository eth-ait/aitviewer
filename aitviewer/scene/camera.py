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
import joblib
import os

from aitviewer.configuration import CONFIG as C
from aitviewer.scene.camera_utils import look_at
from aitviewer.scene.camera_utils import orthographic_projection
from aitviewer.scene.camera_utils import perspective_projection
from trimesh.transformations import rotation_matrix

def _transform_vector(transform, vector):
    """Apply affine transformation (4-by-4 matrix) to a 3D vector."""
    return (transform @ np.concatenate([vector, np.array([1])]))[:3]


class PinholeCamera(object):
    """
    Your classic pinhole camera.
    """

    def __init__(self, fov=45, orthographic=None, znear=C.znear, zfar=C.zfar):
        self.fov = fov
        self.is_ortho = orthographic is not None
        self.ortho_size = 1.0 if orthographic is None else orthographic

        # Default camera settings.
        self.position = np.array([0.0, 0.0, 2.5])
        self.target = np.array([0.0, 0.0, 0.0])
        self.up = np.array([0.0, 1.0, 0.0])

        self.ZOOM_FACTOR = 4
        self.ROT_FACTOR = 0.0025
        self.PAN_FACTOR = 0.01

        self.near = znear
        self.far = zfar

    @property
    def dir(self):
        dir = self.target - self.position
        return dir / np.linalg.norm(dir)

    def view(self):
        """Return the current view matrix."""
        return look_at(self.position, self.target, self.up)

    def save_cam(self):
        """Saves the current camera parameters"""
        cam_dir = C.export_dir + '/camera_params/'
        if not os.path.exists(cam_dir):
            os.makedirs(cam_dir)

        cam_dict = {}
        cam_dict['position'] = self.position
        cam_dict['target'] = self.target
        cam_dict['up'] = self.up
        cam_dict['ZOOM_FACTOR'] = self.ZOOM_FACTOR
        cam_dict['ROT_FACTOR'] = self.ROT_FACTOR
        cam_dict['PAN_FACTOR'] = self.PAN_FACTOR
        cam_dict['near'] = self.near
        cam_dict['far'] = self.far

        joblib.dump(cam_dict, cam_dir+'cam_params.pkl')

    def load_cam(self):
        """Loads the camera parameters"""
        cam_dir = C.export_dir + '/camera_params/'
        if not os.path.exists(cam_dir):
            print('camera config does not exist')
        else:
            cam_dict = joblib.load(cam_dir + 'cam_params.pkl')
            self.position = cam_dict['position']
            self.target = cam_dict['target']
            self.up = cam_dict['up']
            self.ZOOM_FACTOR = cam_dict['ZOOM_FACTOR']
            self.ROT_FACTOR = cam_dict['ROT_FACTOR']
            self.PAN_FACTOR = cam_dict['PAN_FACTOR']
            self.near = cam_dict['near']
            self.far = cam_dict['far']

    def get_projection_matrix(self, width, height):
        """
        Returns the matrix that projects 3D coordinates in camera space to the image plane.
        :param width: Width of the image in pixels.
        :param height: Height of the image in pixels.
        :return: The camera projection matrix as a 4x4 np array.
        """
        if self.is_ortho:
            yscale = self.ortho_size
            xscale = width / height * yscale
            return orthographic_projection(xscale, yscale, self.near, self.far)
        else:
            return perspective_projection(np.deg2rad(self.fov), width / height, self.near, self.far)

    def get_view_projection_matrix(self, width, height):
        """
        Returns the view-projection matrix, i.e. the 4x4 matrix that maps from homogenous world coordinates to image
        space.
        :param width: Width of the image in pixels.
        :param height: Height of the image in pixels.
        :return: The view-projection matrix as a 4x4 np array.
        """
        V = self.view()
        P = self.get_projection_matrix(width, height)
        return np.matmul(P, V)

    def dolly_zoom(self, speed):
        """Zoom by moving the camera along its view direction."""
        # We update both the orthographic and perspective projection so that the transition is seamless when
        # transitioning between them.
        self.ortho_size -= 0.1 * np.sign(speed)
        self.ortho_size = max(0.0001, self.ortho_size)

        # Scale the speed in proportion to the norm (i.e. camera moves slower closer to the target)
        norm = np.linalg.norm(self.position - self.target)
        fwd = self.dir / np.linalg.norm(self.dir)

        # Adjust speed according to config
        speed *= C.camera_zoom_speed

        self.position += fwd * speed * norm

    def pan(self, mouse_dx, mouse_dy):
        """Move the camera in the image plane."""
        sideways = np.cross(self.dir, self.up)
        up = np.cross(sideways, self.dir)

        speed_x = mouse_dx * self.PAN_FACTOR
        speed_y = mouse_dy * self.PAN_FACTOR

        self.position -= sideways * speed_x
        self.target -= sideways * speed_x

        self.position += up * speed_y
        self.target += up * speed_y

    def rotate_azimuth_elevation(self, mouse_dx, mouse_dy):
        """Rotate the camera left-right and up-down (roll is not allowed)."""
        cam_pose = np.linalg.inv(self.view())

        z_axis = cam_pose[:3, 2]
        dot = np.dot(z_axis, self.up)
        rot = np.eye(4)

        # Avoid singularity when z axis of camera is aligned with the up axis of the scene.
        if not (mouse_dy > 0 and dot > 0 and 1 - dot < 0.001) and not (mouse_dy < 0 and dot < 0 and 1 + dot < 0.001):
            # We are either hovering exactly below or above the scene's target but we want to move away or we are
            # not hitting the singularity anyway.
            x_axis = cam_pose[:3, 0]
            rot_x = rotation_matrix(self.ROT_FACTOR * -mouse_dy, x_axis, self.target)
            rot = rot_x @ rot

        y_axis = cam_pose[:3, 1]
        x_speed = self.ROT_FACTOR / 10 if 1 - np.abs(dot) < 0.01 else self.ROT_FACTOR
        rot = rotation_matrix(x_speed * -mouse_dx, y_axis, self.target) @ rot

        self.position = _transform_vector(rot, self.position)

    def rotate_azimuth(self, angle):
        """Rotate around camera's up-axis by given angle (in radians)."""
        if np.abs(angle) < 1e-8:
            return
        cam_pose = np.linalg.inv(self.view())
        y_axis = cam_pose[:3, 1]
        rot = rotation_matrix(angle, y_axis, self.target)
        self.position = _transform_vector(rot, self.position)

    def get_ray(self, x, y, width, height):
        """Construct a ray going through the middle of the given pixel."""
        w, h = width, height

        # Pixel in (-1, 1) range.
        screen_x = (2 * (x + 0.5) / w - 1)
        screen_y = (1 - 2 * (y + 0.5) / h)

        # Scale to actual image plane size.
        scale = self.ortho_size if self.is_ortho else np.tan(np.deg2rad(self.fov) / 2)
        screen_x *= scale * w / h
        screen_y *= scale

        pixel_2d = np.array([screen_x, screen_y, 0 if self.is_ortho else -1])
        cam2world = np.linalg.inv(self.view())
        pixel_3d = _transform_vector(cam2world, pixel_2d)
        if self.is_ortho:
            ray_origin = pixel_3d
            ray_dir = self.dir
        else:
            eye_origin = np.zeros(3)
            ray_origin = _transform_vector(cam2world, eye_origin)
            ray_dir = pixel_3d - ray_origin
        ray_dir = ray_dir / np.linalg.norm(ray_dir)

        return ray_origin, ray_dir

    def gui(self, imgui):
        _, self.is_ortho = imgui.checkbox('Orthographic Camera', self.is_ortho)
        _, self.fov = imgui.slider_float('Camera FOV##fov', self.fov, 0.1, 180.0, '%.1f')

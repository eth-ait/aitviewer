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
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.scene.node import Node
from aitviewer.shaders import get_screen_texture_program
from aitviewer.utils.decorators import hooked
from typing import List

import cv2
import pickle
import moderngl
import numpy as np


class Billboard(Node):
    """ A billboard for displaying a sequence of images as an object in the world"""

    def __init__(self,
                 vertices,
                 texture_paths,
                 img_process_fn=None,
                 **kwargs):
        """ Initializer.
        :param vertices:
            A np array of 4 billboard vertices in world space coordinates of shape (4, 3)
            or an array of shape (N, 4, 3) containing 4 vertices for each frame of the sequence
        :param texture_paths: A list of length N containing paths to the textures as image files.
        """
        super(Billboard, self).__init__(n_frames=len(texture_paths), **kwargs)
        
        if len(vertices.shape) == 2:
            vertices = vertices[np.newaxis]
        else:
            assert vertices.shape[0] == 1 or vertices.shape[0] == len(texture_paths), "the length of the sequence of vertices must be 1 or match the number of textures"

        self.vertices = vertices
        self.img_process_fn = (lambda img: img) if img_process_fn is None else img_process_fn

        # Tile the uv buffer to match the size of the vertices buffer,
        # we do this so that we can use the same vertex array for all draws
        self.uvs = np.repeat(np.array([[
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]], np.float32), self.vertices.shape[0], axis=0)

        self.texture_paths = texture_paths

        self.texture = None
        self.texture_alpha = 1.0
        self._current_texture_id = None

        self.backface_culling = False

    @classmethod
    def from_camera_and_distance(cls, camera: OpenCVCamera, distance: float, cols: int, rows: int,
                                 texture_paths: List[str]):
        """
        Initialize a Billboard from an OpenCV camera object, a distance from the camera, the size of the image in
        pixels and the set of images.
        """
        assert isinstance(camera, OpenCVCamera), "Camera must be an OpenCVCamera."
        frames = camera.n_frames
        frame_id = camera.current_frame_id

        all_corners = np.zeros((frames, 4, 3))
        for i in range(frames):
            camera.current_frame_id = i
            camera.update_matrices(cols, rows)
            V = camera.get_view_matrix()
            P = camera.get_projection_matrix()
            ndc_from_world = P @ V

            # Comput z coordinate of a point at the given distance
            world_p = camera.position + camera.forward * distance
            ndc_p = ndc_from_world @ np.append(world_p, 1.0)

            # Perspective division
            z = ndc_p[2] / ndc_p[3]

            # NDC of corners at the computed distance
            corners = np.array([
                [ 1,  1, z], 
                [ 1, -1, z], 
                [-1,  1, z], 
                [-1, -1, z],
            ])

            # Transform ndc coordinates to world coordinates
            world_from_ndc = np.linalg.inv(ndc_from_world)
            def transform(x):
                v = world_from_ndc @ np.append(x, 1.0)
                v = v[:3] / v[3]
                return v
            corners = np.apply_along_axis(transform, 1, corners)

            all_corners[i] = corners

        camera.current_frame_id = frame_id

        def img_process_fn(img):
            return cv2.undistort(img, camera.K, camera.dist_coeffs)

        return cls(all_corners, texture_paths, img_process_fn)

    # noinspection PyAttributeOutsideInit
    @Node.once
    def make_renderable(self, ctx):
        self.prog = get_screen_texture_program()
        self.vbo_vertices = ctx.buffer(self.vertices.astype('f4').tobytes())
        self.vbo_uvs = ctx.buffer(self.uvs.astype('f4').tobytes())
        self.vao = ctx.vertex_array(self.prog,
                                    [(self.vbo_vertices, '3f4 /v', 'in_position'),
                                     (self.vbo_uvs, '2f4 /v', 'in_texcoord_0')])
        self.ctx = ctx

    def render(self, camera, **kwargs):
        if self.current_frame_id != self._current_texture_id:
            if self.texture:
                self.texture.release()
                
            path = self.texture_paths[self.current_frame_id]
            if path.endswith((".pickle", "pkl")):
                img = pickle.load(open(path, "rb"))
                img = self.img_process_fn(img)
                self.texture = self.ctx.texture(img.shape[:2], img.shape[2], img.tobytes())
            else:
                img = cv2.cvtColor(cv2.flip(cv2.imread(path), 0), cv2.COLOR_BGR2RGB)
                img = self.img_process_fn(img)
                self.texture = self.ctx.texture((img.shape[1], img.shape[0]), img.shape[2], img.tobytes())
            self._current_texture_id = self.current_frame_id

        self.prog['transparency'] = self.texture_alpha
        self.prog['texture0'].value = 0
        self.texture.use(0)

        self.set_camera_matrices(self.prog, camera, **kwargs)
        
        # Compute the index of the first vertex to use if we have a sequence of vertices of length > 1
        first = 4 * self.current_frame_id if self.vertices.shape[0] > 1 else 0
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, first=first)

    @hooked
    def release(self):
        if self.is_renderable:
            self.vbo_vertices.release()
            self.vbo_uvs.release()
            self.vao.release()
            if self.texture:
                self.texture.release()
    
    def is_transparent(self):
        return self.texture_alpha < 1.0

    def gui_material(self, imgui, show_advanced=True):
        _, self.texture_alpha = imgui.slider_float('Texture alpha##texture_alpha{}'.format(self.unique_name),
                                                    self.texture_alpha, 0.0, 1.0, '%.2f')

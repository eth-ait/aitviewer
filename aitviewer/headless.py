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
import os
from typing import Dict, Tuple

import numpy as np
from PIL.Image import Image

from aitviewer.viewer import Viewer


class HeadlessRenderer(Viewer):
    samples = 4
    window_type = "headless"

    def __init__(self, **kwargs):
        """
        Initializer.
        :param frame_dir: Where to save the frames to.
        :param kwargs: kwargs.
        """
        super().__init__(**kwargs)

    def run(self, frame_dir=None, video_dir=None, output_fps=60):
        """Same as self.save_video, kept for backward compatibility."""
        return self.save_video(frame_dir, video_dir, output_fps)

    def save_video(self, frame_dir=None, video_dir=None, output_fps=60, transparent=False):
        """
        Convenience method to run the headless rendering.
        :param frame_dir: Where to store the individual frames or None if you don't care.
        :param video_dir: If set will automatically generate a video from the images found in `frame_dir`. Must
          be specified if `frame_dir` is None.
        :param output_fps: Fps of the output video, if None uses 60fps as default
        :param transparent: Save video with a transparent background, this is only supported by ".webm" format
            and ignored otherwise.
        """
        self._init_scene()
        self.export_video(
            output_path=video_dir, frame_dir=frame_dir, animation=True, output_fps=output_fps, transparent=transparent
        )

    def save_frame(self, file_path, scale_factor: float = None):
        """
        Run the headless viewer and render a single frame.
        :param file_path: the path where the image is saved.
        :param scale_factor: a scale factor used to scale the image. If None no scale factor is used and
          the image will have the same size as the viewer.
        """
        self._init_scene()
        self.export_frame(file_path, scale_factor)

    def save_depth(self, file_path):
        """
        Render and save the depth buffer, see 'get_depth()' for more information
        about the depth format.
        :param file_path: the path where the image is saved. The file is used by PIL to choose
        the file format, make sure that you use a format that supports 'F' mode PIL Images (e.g. tiff).
        """
        dir = os.path.dirname(file_path)
        if dir:
            os.makedirs(dir, exist_ok=True)
        self.get_depth().save(file_path)

    def save_mask(self, file_path, color_map: Dict[int, Tuple[int, int, int]] = None, id_map: Dict[int, int] = None):
        """
        Render and save a color mask as a 'RGB' PIL image.
        Each object in the mask has a uniform color computed from the Node UID (can be accessed from a node with 'node.uid').

        :param file_path: the path where the image is saved.
        :param color_map:
            if not None specifies the color to use for a given Node UID as a tuple (R, G, B) of integer values from 0 to 255.
            If None the color is computed as an hash of the Node UID instead.
        :param id_map:
            if not None the UIDs in the mask are mapped using this dictionary from Node UID to the specified ID.
            This mapping is applied before the color map (or before hashing if the color map is None).
        """
        dir = os.path.dirname(file_path)
        if dir:
            os.makedirs(dir, exist_ok=True)
        self.get_mask(color_map, id_map).save(file_path)

    def _render_frame(self):
        self._init_scene()

        # Store run_animation old value and set it to false.
        run_animations = self.run_animations
        self.run_animations = False

        # Render frame.
        self.render(0, 0, export=True)

        # Restore run animation and update last frame rendered time.
        self.run_animations = run_animations

    def get_frame(self) -> Image:
        """Render and return a single frame as a 'RGB' PIL image"""
        self._render_frame()
        return self.get_current_frame_as_image()

    def get_depth(self) -> Image:
        """
        Render and return the depth buffer as a 'F' PIL image.
        Depth is stored as the z coordinate in eye (view) space.
        Therefore values in the depth image represent the distance from the pixel to
        the plane passing through the camera and orthogonal to the view direction.
        Values are between the near and far plane distances of the camera used for rendering,
        everything outside this range is clipped by OpenGL.
        """
        self._render_frame()
        return self.get_current_depth_image()

    def get_mask_ids(self, id_map: Dict[int, int] = None) -> np.ndarray:
        """
        Return a mask as a numpy array of shape (height, width) and type np.uint32.
        Each element in the array is the UID of the node covering that pixel (can be accessed from a node with 'node.uid')
        or zero if not covered.

        :param id_map:
            if not None the UIDs in the mask are mapped using this dictionary to the specified ID.
            The final mask only contains the IDs specified in this mapping and zeros everywhere else.
        """
        self._render_frame()
        return self.get_current_mask_ids(id_map)

    def get_mask(self, color_map: Dict[int, Tuple[int, int, int]] = None, id_map: Dict[int, int] = None) -> Image:
        """
        Render and return a color mask as a 'RGB' PIL image.
        Each object in the mask has a uniform color computed from the Node UID (can be accessed from a node with 'node.uid').

        :param color_map:
            if not None specifies the color to use for a given Node UID as a tuple (R, G, B) of integer values from 0 to 255.
            If None the color is computed as an hash of the Node UID instead.
        :param id_map:
            if not None the UIDs in the mask are mapped using this dictionary from Node UID to the specified ID.
            This mapping is applied before the color map (or before hashing if the color map is None).
        """
        self._render_frame()
        return self.get_current_mask_image(color_map, id_map)

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
from aitviewer.viewer import Viewer


class HeadlessRenderer(Viewer):
    gl_version = (4, 0)
    samples = 0  # Headless rendering does not like super sampling.
    window_type = 'headless'

    def __init__(self, **kwargs):
        """
        Initializer.
        :param frame_dir: Where to save the frames to.
        :param kwargs: kwargs.
        """
        super().__init__(**kwargs)

    def run(self, frame_dir=None, video_dir=None, output_fps=60):
        """
        Convenience method to run the headless rendering.
        :param frame_dir: Where to store the individual frames or None if you don't care.
        :param video_dir: If set will automatically generate a video from the images found in `frame_dir`. Must
          be specified if `frame_dir` is None.
        :param output_fps: Fps of the output video, if None uses 60fps as default
        """
        self._init_scene()
        self.export_video(
            output_path=video_dir,
            frame_dir=frame_dir,
            animation=True,
            output_fps=output_fps,
        )

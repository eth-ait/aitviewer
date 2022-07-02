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
import tempfile
import shutil

from aitviewer.scene.camera import PinholeCamera
from aitviewer.utils import images_to_video
from aitviewer.viewer import Viewer
from tqdm import tqdm


class HeadlessRenderer(Viewer):
    gl_version = (3, 3)
    samples = 0  # Headless rendering does not like super sampling.
    window_type = 'headless'

    def __init__(self, **kwargs):
        """
        Initializer.
        :param frame_dir: Where to save the frames to.
        :param kwargs: kwargs.
        """
        super().__init__(**kwargs)

        # Scene setup.
        self.camera = PinholeCamera(45.0)
        self.draw_edges = False

        # Book-keeping for the headless rendering.
        self.n_frames_rendered = 0
        self.frame_dir = None
        self.progress_bar = None

    def run(self, frame_dir=None, video_dir=None, keep_frames=False, log=True, load_cam=False):
        """
        Convenience method to run the headless rendering.
        :param frame_dir: Where to store the individual frames or None if you don't care.
        :param video_dir: If set will automatically generate a video from the images found in `frame_dir`. Must
          be specified if `frame_dir` is None.
        :param keep_frames: Whether to keep the individual frames or automatically delete them. This is ignored if
         `frame_dir` is set, i.e. when `frame_dir` is set, we never delete frames.
        :param log: Log some info.
        """
        if frame_dir is None and video_dir is None:
            raise ValueError("You should either specify a path where to render the images to or where to "
                             "save the video to.")

        if video_dir is not None:
            assert video_dir.endswith(".mp4")

        if frame_dir is None:
            temp_dir = tempfile.TemporaryDirectory()
            frame_dir = tempfile.TemporaryDirectory().name
        else:
            temp_dir = None
            # We want to avoid overriding anything in an existing directory, so add suffixes.
            format_str = "{:0>4}"
            counter = 0
            candidate_dir = os.path.join(frame_dir, format_str.format(counter))
            while os.path.exists(candidate_dir):
                counter += 1
                candidate_dir = os.path.join(frame_dir, format_str.format(counter))
            frame_dir = os.path.abspath(candidate_dir)

        # The frame dir does not yet exist (we've made sure of it).
        os.makedirs(frame_dir)
        self.frame_dir = frame_dir

        self.scene.make_renderable(self.ctx)
        if self.auto_set_floor:
            self.scene.auto_set_floor()

        if self.auto_set_camera_target:
            self.scene.auto_set_camera_target()

        if load_cam:
            self.scene.camera.load_cam()

        self.timer.start()
        while not self.rendering_finished():
            current_time, delta = self.timer.next_frame()
            self.window.clear()
            self.window.render(current_time, delta)
            self.window.swap_buffers()
            self.scene.next_frame()
        _, duration = self.timer.stop()

        if duration > 0 and log:
            print("Duration: {0:.2f}s @ {1:.2f} FPS".format(duration, self.max_frame / duration))

        if video_dir is not None:
            images_to_video(frame_dir, video_dir)

        # Only delete the frames if it was a temporary directory and we don't want to keep them.
        if temp_dir is not None and not keep_frames:
            shutil.rmtree(frame_dir)

    @property
    def max_frame(self):
        return self.scene.n_frames

    def rendering_finished(self):
        return not self.n_frames_rendered < self.max_frame

    def render(self, time, frame_time):
        self.render_shadowmap()
        self.render_prepare()
        self.render_scene()

        self.n_frames_rendered += 1
        if self.progress_bar is None:
            self.progress_bar = tqdm(total=self.max_frame, desc='Rendering frames')

        if self.n_frames_rendered > self.max_frame:
            self.progress_bar.close()
            self.wnd.close()
        else:
            self.progress_bar.update()
            self.save_current_frame_as_image(self.frame_dir, self.n_frames_rendered - 1)


# def _instantiate_and_run_window(window_cls: Viewer, *args, log=True):
#     """
#     Instantiate the window passing the provided args into the constructor and enter a blocking visualization loop.
#     This is built following `moderngl_window.run_window_config`.
#
#     :param window_cls: The window to run.
#     :param args: The arguments passed to `conig_cls` constructor.
#     :param log: Whether to log to the console.
#     """
#     base_window_cls = get_local_window_cls(window_cls.window_type)
#
#     # Calculate window size
#     size = window_cls.window_size
#     size = int(size[0] * window_cls.size_mult), int(size[1] * window_cls.size_mult)
#
#     window = base_window_cls(
#         title=window_cls.title,
#         size=size,
#         fullscreen=window_cls.fullscreen,
#         resizable=window_cls.resizable,
#         gl_version=window_cls.gl_version,
#         aspect_ratio=window_cls.aspect_ratio,
#         vsync=window_cls.vsync,
#         samples=window_cls.samples,
#         cursor=False,
#     )
#     window.print_context_info()
#     activate_context(window=window)
#     timer = Timer()
#     window.config = window_cls(*args, ctx=window.ctx, wnd=window, timer=timer)
#
#     timer.start()
#
#     while not window.is_closing:
#         current_time, delta = timer.next_frame()
#         window.clear()
#         window.render(current_time, delta)
#         window.swap_buffers()
#
#     _, duration = timer.stop()
#     window.destroy()
#     if duration > 0 and log:
#         print("Duration: {0:.2f}s @ {1:.2f} FPS".format(duration, window.frames / duration))

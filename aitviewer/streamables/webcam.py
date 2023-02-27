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
import cv2
import numpy as np
from moderngl_window import geometry

from aitviewer.scene.node import Node
from aitviewer.shaders import get_screen_texture_program
from aitviewer.streamables.streamable import Streamable


class Webcam(Streamable):
    """
    Renders webcam stream to a quad. Quad is positioned in screen coordinates.
    """

    def __init__(
        self,
        src=0,
        size=(2.0, 2.0),
        pos=(0.0, 0.0),
        transparency=1.0,
        icon="\u0088",
        **kwargs,
    ):
        """
        :param src: integer denotes device source id, i.e. webcam 0.
        """

        super(Webcam, self).__init__(icon=icon, **kwargs)

        # Capture source
        self._cap = None

        # Set after cv reads video file or webcam
        self.width = None
        self.height = None
        self.pos = pos
        self.size = size
        self.fps = None
        self.frame_count = None
        self.transparency = transparency
        self.src = src

        # Render into a quad in screen space (z=0)
        self._texture = None

    @Node.once
    def make_renderable(self, ctx):
        self.ctx = ctx
        self.quad = geometry.quad_2d(
            pos=self.pos, size=self.size, normals=False
        )  # (2,2) is Full Screen i.e. -1 to 1 in x/y
        self.prog = get_screen_texture_program()

    def capture(self):
        ret, frame = self._cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._texture.write(frame)

    def render(self, camera, **kwargs):
        self._texture.use(0)
        self.prog["transparency"].value = self.transparency
        self.prog["mvp"].write(np.eye(4).astype("f4").tobytes())
        self.quad.render(self.prog)

    @property
    def enabled(self):
        return self._cap is not None and self._cap.isOpened()

    @enabled.setter
    def enabled(self, enabled):
        # Not running - start
        if enabled and not self.enabled:
            self.start()
        # Running - stop
        elif not enabled and self.enabled:
            self.stop()

    def start(self):
        # Capture from webcam / device source
        self._cap = cv2.VideoCapture(self.src)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)

        # Set W/H Manually
        # self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Initialize texture to
        self._texture = self.ctx.texture((self.width, self.height), 3)

        # Check if the webcam is opened correctly
        if not self._cap.isOpened():
            raise IOError("Cannot open source")

    def stop(self):
        self._cap.release()
        self._texture = None

    def gui(self, imgui):
        if self.enabled:
            imgui.text("Clip Info: {}x{} @ {:.2f} fps".format(self.width, self.height, self.fps))
        _, self.transparency = imgui.slider_float(
            "Opacity##opacity_".format(self.name), self.transparency, 0.0, 1.0, "%.2f"
        )

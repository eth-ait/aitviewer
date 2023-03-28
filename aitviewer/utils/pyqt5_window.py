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

from moderngl_window.context.pyqt5 import Window
from PyQt5 import QtGui, QtOpenGL, QtWidgets

"""
Window class adapted from  moderngl_window.context.pyqt5.Window.
Only slight changes are made to avoid a crash with Python >= 3.10.
"""


class PyQt5Window(Window):
    def __init__(self, **kwargs):
        super(Window, self).__init__(**kwargs)

        # Specify OpenGL context parameters
        gl = QtOpenGL.QGLFormat()
        gl.setVersion(self.gl_version[0], self.gl_version[1])
        gl.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        gl.setDepthBufferSize(24)
        gl.setDoubleBuffer(True)
        gl.setSwapInterval(1 if self._vsync else 0)
        gl.setAlphaBufferSize(8)

        # Configure multisampling if needed
        if self.samples > 1:
            gl.setSampleBuffers(True)
            gl.setSamples(int(self.samples))

        # We need an application object, but we are bypassing the library's
        # internal event loop to avoid unnecessary work
        self._app = QtWidgets.QApplication([])

        # Create the OpenGL widget
        self._widget = QtOpenGL.QGLWidget(gl)
        self.title = self._title

        # If fullscreen we change the window to match the desktop on the primary screen
        if self.fullscreen:
            rect = QtWidgets.QDesktopWidget().screenGeometry()
            self._width = rect.width()
            self._height = rect.height()
            self._buffer_width = rect.width() * self._widget.devicePixelRatio()
            self._buffer_height = rect.height() * self._widget.devicePixelRatio()

        if self.resizable:
            # Ensure a valid resize policy when window is resizable
            size_policy = QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding,
            )
            self._widget.setSizePolicy(size_policy)
            self._widget.resize(self.width, self.height)
        else:
            self._widget.setFixedSize(self.width, self.height)

        # Center the window on the screen if in window mode
        if not self.fullscreen:
            center_window_position = (
                # HACK: Here we changed division to integer division,
                # because move requires its arguments to be integers.
                self.position[0] - self.width // 2,
                self.position[1] - self.height // 2,
            )
            self._widget.move(*center_window_position)

        # Needs to be set before show()
        self._widget.resizeGL = self.resize

        self.cursor = self._cursor

        if self.fullscreen:
            self._widget.showFullScreen()
        else:
            self._widget.show()

        # We want mouse position events
        self._widget.setMouseTracking(True)

        # Override event functions in qt
        self._widget.keyPressEvent = self.key_pressed_event
        self._widget.keyReleaseEvent = self.key_release_event
        self._widget.mouseMoveEvent = self.mouse_move_event
        self._widget.mousePressEvent = self.mouse_press_event
        self._widget.mouseReleaseEvent = self.mouse_release_event
        self._widget.wheelEvent = self.mouse_wheel_event
        self._widget.closeEvent = self.close_event
        self._widget.showEvent = self.show_event
        self._widget.hideEvent = self.hide_event

        # Attach to the context
        self.init_mgl_context()

        # Ensure retina and 4k displays get the right viewport
        self._buffer_width = self._width * self._widget.devicePixelRatio()
        self._buffer_height = self._height * self._widget.devicePixelRatio()

        self.set_default_viewport()

    def _set_icon(self, icon_path: str) -> None:
        self._widget.setWindowIcon(QtGui.QIcon(str(icon_path)))

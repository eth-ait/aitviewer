# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos

from typing import Tuple

from moderngl_window.context.base import BaseKeys, BaseWindow
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtOpenGL import QOpenGLWindow


class Keys(BaseKeys):
    """
    Namespace mapping pyqt specific key constants
    """

    ESCAPE = Qt.Key.Key_Escape
    SPACE = Qt.Key.Key_Space
    ENTER = Qt.Key.Key_Return
    # ENTER = Qt.Key.Key_Enter #This is numpad enter
    PAGE_UP = Qt.Key.Key_PageUp
    PAGE_DOWN = Qt.Key.Key_PageDown
    LEFT = Qt.Key.Key_Left
    RIGHT = Qt.Key.Key_Right
    UP = Qt.Key.Key_Up
    DOWN = Qt.Key.Key_Down

    TAB = Qt.Key.Key_Tab
    COMMA = Qt.Key.Key_Comma
    MINUS = Qt.Key.Key_Minus
    PERIOD = Qt.Key.Key_Period
    SLASH = Qt.Key.Key_Slash
    SEMICOLON = Qt.Key.Key_Semicolon
    EQUAL = Qt.Key.Key_Equal
    LEFT_BRACKET = Qt.Key.Key_BracketLeft
    RIGHT_BRACKET = Qt.Key.Key_BracketRight
    BACKSLASH = Qt.Key.Key_Backslash
    BACKSPACE = Qt.Key.Key_Backspace
    INSERT = Qt.Key.Key_Insert
    DELETE = Qt.Key.Key_Delete
    HOME = Qt.Key.Key_Home
    END = Qt.Key.Key_End
    CAPS_LOCK = Qt.Key.Key_CapsLock

    F1 = Qt.Key.Key_F1
    F2 = Qt.Key.Key_F2
    F3 = Qt.Key.Key_F3
    F4 = Qt.Key.Key_F4
    F5 = Qt.Key.Key_F5
    F6 = Qt.Key.Key_F6
    F7 = Qt.Key.Key_F7
    F8 = Qt.Key.Key_F8
    F9 = Qt.Key.Key_F9
    F10 = Qt.Key.Key_F10
    F11 = Qt.Key.Key_F11
    F12 = Qt.Key.Key_F12

    NUMBER_0 = Qt.Key.Key_0
    NUMBER_1 = Qt.Key.Key_1
    NUMBER_2 = Qt.Key.Key_2
    NUMBER_3 = Qt.Key.Key_3
    NUMBER_4 = Qt.Key.Key_4
    NUMBER_5 = Qt.Key.Key_5
    NUMBER_6 = Qt.Key.Key_6
    NUMBER_7 = Qt.Key.Key_7
    NUMBER_8 = Qt.Key.Key_8
    NUMBER_9 = Qt.Key.Key_9

    # Uses a modifier for numpad. We just repeat the numbers for compatibility
    NUMPAD_0 = Qt.Key.Key_0
    NUMPAD_1 = Qt.Key.Key_1
    NUMPAD_2 = Qt.Key.Key_2
    NUMPAD_3 = Qt.Key.Key_3
    NUMPAD_4 = Qt.Key.Key_4
    NUMPAD_5 = Qt.Key.Key_5
    NUMPAD_6 = Qt.Key.Key_6
    NUMPAD_7 = Qt.Key.Key_7
    NUMPAD_8 = Qt.Key.Key_8
    NUMPAD_9 = Qt.Key.Key_9

    A = Qt.Key.Key_A
    B = Qt.Key.Key_B
    C = Qt.Key.Key_C
    D = Qt.Key.Key_D
    E = Qt.Key.Key_E
    F = Qt.Key.Key_F
    G = Qt.Key.Key_G
    H = Qt.Key.Key_H
    I = Qt.Key.Key_I
    J = Qt.Key.Key_J
    K = Qt.Key.Key_K
    L = Qt.Key.Key_L
    M = Qt.Key.Key_M
    N = Qt.Key.Key_N
    O = Qt.Key.Key_O
    P = Qt.Key.Key_P
    Q = Qt.Key.Key_Q
    R = Qt.Key.Key_R
    S = Qt.Key.Key_S
    T = Qt.Key.Key_T
    U = Qt.Key.Key_U
    V = Qt.Key.Key_V
    W = Qt.Key.Key_W
    X = Qt.Key.Key_X
    Y = Qt.Key.Key_Y
    Z = Qt.Key.Key_Z


class GLWindow(QOpenGLWindow):
    def __init__(self, gl_version, vsync, samples):
        super().__init__()
        fmt = QtGui.QSurfaceFormat()
        fmt.setVersion(gl_version[0], gl_version[1])
        fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setSwapBehavior(QtGui.QSurfaceFormat.SwapBehavior.DoubleBuffer)
        fmt.setSwapInterval(1 if vsync else 0)
        fmt.setAlphaBufferSize(8)
        if samples > 1:
            fmt.setSamples(int(samples))
        self.setFormat(fmt)

    def initializeGL(self):
        pass

    def paintGL(self) -> None:
        pass


class PyQt6Window(BaseWindow):
    """
    A basic window implementation using PyQt5 with the goal of
    creating an OpenGL context and handle keyboard and mouse input.

    This window bypasses Qt's own event loop to make things as flexible as possible.

    If you need to use the event loop and are using other features
    in Qt as well, this example can still be useful as a reference
    when creating your own window.
    """

    #: Name of the window
    name = "pyqt6"
    #: PyQt6 specific key constants
    keys = Keys

    # PyQt supports mode buttons, but we are limited by other libraries
    _mouse_button_map = {
        QtCore.Qt.MouseButton.LeftButton: 1,
        QtCore.Qt.MouseButton.RightButton: 2,
        QtCore.Qt.MouseButton.MiddleButton: 3,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # We need an application object, but we are bypassing the library's
        # internal event loop to avoid unnecessary work
        self._app = QtWidgets.QApplication([])
        # Create the OpenGL window
        self._widget = GLWindow(self.gl_version, self._vsync, self.samples)
        self.title = self._title

        if self.resizable:
            self._widget.resize(self.width, self.height)
        else:
            self._widget.setMaximumSize(QtCore.QSize(self.width, self.height))
            self._widget.setMinimumSize(QtCore.QSize(self.width, self.height))

        # Needs to be set before show()
        self._widget.resizeGL = self.resize

        self.cursor = self._cursor

        if self.fullscreen:
            self._widget.showFullScreen()
        else:
            self._widget.show()

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

        # # Attach to the context
        self.init_mgl_context()

        # Ensure retina and 4k displays get the right viewport
        self._buffer_width = int(self._width * self._widget.devicePixelRatio())
        self._buffer_height = int(self._height * self._widget.devicePixelRatio())

        self.set_default_viewport()

    def _set_icon(self, icon_path: str) -> None:
        self._widget.setIcon(QtGui.QIcon(str(icon_path)))
        pass

    def _set_fullscreen(self, value: bool) -> None:
        if value:
            self._widget.showFullScreen()
        else:
            self._widget.showNormal()

    @property
    def size(self) -> Tuple[int, int]:
        """Tuple[int, int]: current window size.

        This property also support assignment::

            # Resize the window to 1000 x 1000
            window.size = 1000, 1000
        """
        return self._width, self._height

    @size.setter
    def size(self, value: Tuple[int, int]):
        pos = self.position
        self._widget.setGeometry(pos[0], pos[1], value[0], value[1])

    @property
    def position(self) -> Tuple[int, int]:
        """Tuple[int, int]: The current window position.

        This property can also be set to move the window::

            # Move window to 100, 100
            window.position = 100, 100
        """
        geo = self._widget.geometry()
        return geo.x(), geo.y()

    @position.setter
    def position(self, value: Tuple[int, int]):
        self._widget.setGeometry(value[0], value[1], self._width, self._height)

    def swap_buffers(self) -> None:
        """Swap buffers, set viewport, trigger events and increment frame counter"""

        # Trigger a redraw, equivalent to self._widget.swapBuffers for Qt5.
        self._widget.event(QtGui.QPaintEvent(QtGui.QRegion()))
        self.set_default_viewport()
        QtWidgets.QApplication.processEvents()
        self._frames += 1

    @property
    def cursor(self) -> bool:
        """bool: Should the mouse cursor be visible inside the window?

        This property can also be assigned to::

            # Disable cursor
            window.cursor = False
        """
        return self._cursor

    @cursor.setter
    def cursor(self, value: bool):
        if value is True:
            self._widget.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        else:
            self._widget.setCursor(QtCore.Qt.CursorShape.BlankCursor)

        self._cursor = value

    @property
    def title(self) -> str:
        """str: Window title.

        This property can also be set::

            window.title = "New Title"
        """
        return self._title

    @title.setter
    def title(self, value: str):
        self._widget.setTitle(value)
        self._title = value

    def resize(self, width: int, height: int) -> None:
        """Replacement for Qt's ``resizeGL`` method.

        Args:
            width: New window width
            height: New window height
        """
        # NOTE(dario): Do not attempt to resize if the window is closing.
        # This crashes if the OpenGL context has already been destroyed,
        # which seems to be the case when manually closing the window from
        # the cross on windows.
        if self.is_closing:
            return

        self._width = width
        self._height = height
        self._buffer_width = int(width * self._widget.devicePixelRatio())
        self._buffer_height = int(height * self._widget.devicePixelRatio())

        if self._ctx:
            self.set_default_viewport()

        # Make sure we notify the example about the resize
        super().resize(self._buffer_width, self._buffer_height)

    def _handle_modifiers(self, mods) -> None:
        """Update modifiers"""
        self._modifiers.shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        self._modifiers.ctrl = bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier)
        self._modifiers.alt = bool(mods & QtCore.Qt.KeyboardModifier.AltModifier)

    def _set_icon(self, icon_path: str) -> None:
        self._widget.setIcon(QtGui.QIcon(icon_path))

    def key_pressed_event(self, event: QtGui.QKeyEvent) -> None:
        """Process Qt key press events forwarding them to standard methods

        Args:
            event: The qtevent instance
        """
        if self._exit_key is not None and event.key() == self._exit_key:
            self.close()

        if self._fs_key is not None and event.key() == self._fs_key:
            self.fullscreen = not self.fullscreen

        self._handle_modifiers(event.modifiers())
        self._key_pressed_map[event.key()] = True
        self._key_event_func(event.key(), self.keys.ACTION_PRESS, self._modifiers)

        text = event.text()
        if text.strip() or event.key() == self.keys.SPACE:
            self._unicode_char_entered_func(text)

    def key_release_event(self, event: QtGui.QKeyEvent) -> None:
        """Process Qt key release events forwarding them to standard methods

        Args:
            event: The qtevent instance
        """
        self._handle_modifiers(event.modifiers())
        self._key_pressed_map[event.key()] = False
        self._key_event_func(event.key(), self.keys.ACTION_RELEASE, self._modifiers)

    def mouse_move_event(self, event: QtGui.QMouseEvent) -> None:
        """Forward mouse cursor position events to standard methods

        Args:
            event: The qtevent instance
        """
        x, y = event.pos().x(), event.pos().y()
        dx, dy = self._calc_mouse_delta(x, y)

        if self.mouse_states.any:
            self._mouse_drag_event_func(x, y, dx, dy)
        else:
            self._mouse_position_event_func(x, y, dx, dy)

    def mouse_press_event(self, event: QtGui.QMouseEvent) -> None:
        """Forward mouse press events to standard methods

        Args:
            event: The qtevent instance
        """
        self._handle_modifiers(event.modifiers())
        button = self._mouse_button_map.get(event.button())
        if button is None:
            return
        self._handle_mouse_button_state_change(button, True)
        self._mouse_press_event_func(event.pos().x(), event.pos().y(), button)

    def mouse_release_event(self, event: QtGui.QMouseEvent) -> None:
        """Forward mouse release events to standard methods

        Args:
            event: The qtevent instance
        """
        self._handle_modifiers(event.modifiers())
        button = self._mouse_button_map.get(event.button())
        if button is None:
            return

        self._handle_mouse_button_state_change(button, False)
        self._mouse_release_event_func(event.pos().x(), event.pos().y(), button)

    def mouse_wheel_event(self, event: QtGui.QWheelEvent):
        """Forward mouse wheel events to standard metods.

        From Qt docs:

        Returns the distance that the wheel is rotated, in eighths of a degree.
        A positive value indicates that the wheel was rotated forwards away from the user;
        a negative value indicates that the wheel was rotated backwards toward the user.

        Most mouse types work in steps of 15 degrees, in which case the delta value is a
        multiple of 120; i.e., 120 units * 1/8 = 15 degrees.

        However, some mice have finer-resolution wheels and send delta values that are less
        than 120 units (less than 15 degrees). To support this possibility, you can either
        cumulatively add the delta values from events until the value of 120 is reached,
        then scroll the widget, or you can partially scroll the widget in response to each
        wheel event.

        Args:
            event (QWheelEvent): Mouse wheel event
        """
        self._handle_modifiers(event.modifiers())
        point = event.angleDelta()
        self._mouse_scroll_event_func(point.x() / 120.0, point.y() / 120.0)

    def close_event(self, event) -> None:
        """The standard PyQt close events

        Args:
            event: The qtevent instance
        """
        self.close()

    def close(self):
        """Close the window"""
        super().close()
        self._close_func()

    def show_event(self, event):
        """The standard Qt show event"""
        self._iconify_func(False)

    def hide_event(self, event):
        """The standard Qt hide event"""
        self._iconify_func(True)

    def destroy(self) -> None:
        """Quit the Qt application to exit the window gracefully"""
        QtCore.QCoreApplication.instance().quit()

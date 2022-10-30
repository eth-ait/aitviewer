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

from moderngl_window.integrations.imgui import ModernglWindowRenderer

class ImGuiRenderer(ModernglWindowRenderer):
    def __init__(self, window, window_type):
        self.window_type = window_type
        super().__init__(window)

    def key_event(self, key, action, modifiers):
        if self.window_type == 'pyqt5':
            # HACK: we remap Qt.Key_Enter (numpad enter key) to the normal enter key.
            from PyQt5.QtCore import Qt
            if key == Qt.Key_Enter:
                key = self.wnd.keys.ENTER

        super().key_event(key, action, modifiers)

    def render(self, draw_data):
        # HACK: we set the modifiers here every frame because key_event is not called when
        # the window loses and regains focus (e.g. when changing focus with alt+tab)
        self.io.key_alt = self.wnd.modifiers.alt
        self.io.key_ctrl = self.wnd.modifiers.ctrl
        self.io.key_shift = self.wnd.modifiers.shift
        return super().render(draw_data)

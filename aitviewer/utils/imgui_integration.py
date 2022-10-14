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

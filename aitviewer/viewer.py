# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import copy
import os
import sys
from array import array
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Union

import imgui
import moderngl_window
import numpy as np
from moderngl_window import activate_context, get_local_window_cls
from PIL import Image
from tqdm import tqdm

from aitviewer.configuration import CONFIG as C
from aitviewer.remote.message import Message
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.meshes import Meshes, VariableTopologyMeshes
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderer import Renderer, Viewport
from aitviewer.scene.camera import PinholeCamera, ViewerCamera
from aitviewer.scene.node import Node
from aitviewer.scene.scene import Scene
from aitviewer.server import ViewerServer
from aitviewer.shaders import clear_shader_cache
from aitviewer.streamables.streamable import Streamable
from aitviewer.utils import path
from aitviewer.utils.imgui_integration import ImGuiRenderer
from aitviewer.utils.perf_timer import PerfTimer
from aitviewer.utils.utils import get_video_paths, video_to_gif

MeshMouseIntersection = namedtuple(
    "MeshMouseIntersection",
    "node instance_id tri_id vert_id point_world point_local bc_coords",
)

if os.name == "nt":
    import ctypes

    try:
        # On windows we need to modify the current App User Model Id to make the taskbar icon of the viewer
        # the one that we set for the window and not the default python icon.
        #
        # For more details see:
        # https://stackoverflow.com/questions/1551605/how-to-set-applications-taskbar-icon-in-windows-7/1552105#1552105
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ait.viewer.window.1")
    except:
        pass

SHORTCUTS = {
    "SPACE": "Start/stop playing animation.",
    ".": "Go to next frame.",
    ",": "Go to previous frame.",
    "G": "Open a window to change frame by typing the frame number.",
    "X": "Center view on the selected object.",
    "O": "Enable/disable orthographic camera.",
    "T": "Show the camera target in the scene.",
    "B": "Show the camera trackball in the scene.",
    "C": "Save the camera position and orientation to disk.",
    "L": "Load the camera position and orientation from disk.",
    "K": "Lock the selection to the currently selected object.",
    "S": "Show/hide shadows.",
    "D": "Enabled/disable dark mode.",
    "P": "Save a screenshot to the the 'export/screenshots' directory.",
    "I": "Change the viewer mode to 'inspect'",
    "V": "Change the viewer mode to 'view'",
    "E": "If a mesh is selected, show the edges of the mesh.",
    "F": "If a mesh is selected, switch between flat and smooth shading.",
    "Z": "Show a debug visualization of the object IDs.",
    "ESC": "Exit the viewer.",
}


class Viewer(moderngl_window.WindowConfig):
    resource_dir = Path(__file__).parent / "shaders"
    size_mult = 1.0
    samples = 4
    if sys.platform == "darwin":
        gl_version = (4, 0)
    else:
        gl_version = (4, 5)
    window_type = None

    def __init__(
        self,
        title="aitviewer",
        size: Tuple[int, int] = None,
        samples: int = None,
        **kwargs,
    ):
        """
        Initializer.
        :param title: Window title
        :param size: Window size as (width, height) tuple, if None uses the size from the configuration file
        :param kwargs: kwargs.
        """
        # Window Setup (Following `moderngl_window.run_window_config`).
        if self.window_type is None:
            self.window_type = C.window_type

        # HACK: We use our own version of the PyQt5 windows to override
        # part of the initialization that crashes on Python >= 3.10.
        if self.window_type == "pyqt5":
            from aitviewer.utils.pyqt5_window import PyQt5Window

            base_window_cls = PyQt5Window
        elif self.window_type == "pyqt6":
            from aitviewer.utils.pyqt6_window import PyQt6Window

            base_window_cls = PyQt6Window
        else:
            base_window_cls = get_local_window_cls(self.window_type)

        # If no size is provided use the size from the configuration file.
        if size is None:
            size = C.window_width, C.window_height

        # Update nubmer of samples to use if specified as a parameter.
        if samples is not None:
            self.samples = samples

        # Calculate window size
        size = int(size[0] * self.size_mult), int(size[1] * self.size_mult)

        self.window = base_window_cls(
            title=title,
            size=size,
            fullscreen=C.fullscreen,
            resizable=C.resizable,
            gl_version=self.gl_version,
            aspect_ratio=None,  # Have to set this to None otherwise the window will enforce this aspect ratio.
            vsync=C.vsync,  # Set to False for some performance gains.
            samples=self.samples,
            cursor=True,
        )

        self.window_size = size
        self.window.print_context_info()
        activate_context(window=self.window)

        # Try to set the window icon.
        try:
            icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "aitviewer_icon.png")
            self.window.set_icon(icon_path=icon_path)
        except:
            pass

        self.timer = PerfTimer()
        self.ctx = self.window.ctx
        super().__init__(self.ctx, self.window, self.timer)

        # Create renderer
        self.renderer = Renderer(self.ctx, self.wnd, self.window_type)

        # Create GUI context
        self.imgui_ctx = imgui.create_context()
        self.imgui = ImGuiRenderer(self.wnd, self.window_type)
        self.imgui_user_interacting = False

        # Custom UI Font
        self.font_dir = Path(__file__).parent / "resources" / "fonts"
        self.fonts = imgui.get_io().fonts
        self.custom_font = self.fonts.add_font_from_file_ttf(os.path.join(self.font_dir, "Custom.ttf"), 15)
        self.imgui.refresh_font_texture()

        self.modes = {
            "view": {"title": " View", "shortcut": "V"},
            "inspect": {"title": " Inspect", "shortcut": "I"},
        }

        self.gui_controls = {
            "menu": self.gui_menu,
            "scene": self.gui_scene,
            "playback": self.gui_playback,
            "inspect": self.gui_inspect,
            "shortcuts": self.gui_shortcuts,
            "exit": self.gui_exit,
            "go_to_frame": self.gui_go_to_frame,
        }

        # Initialize viewer
        self.scene = None
        self.reset()

        # Key Shortcuts
        self._exit_key = self.wnd.keys.ESCAPE
        self._pause_key = self.wnd.keys.SPACE
        self._next_frame_key = self.wnd.keys.PERIOD
        self._previous_frame_key = self.wnd.keys.COMMA
        self._shadow_key = self.wnd.keys.S
        self._orthographic_camera_key = self.wnd.keys.O
        self._center_view_on_selection_key = self.wnd.keys.X
        self._dark_mode_key = self.wnd.keys.D
        self._screenshot_key = self.wnd.keys.P
        self._middle_mouse_button = 3  # middle
        self._right_mouse_button = 2  # right
        self._left_mouse_button = 1  # left
        self._save_cam_key = self.wnd.keys.C
        self._load_cam_key = self.wnd.keys.L
        self._show_camera_target_key = self.wnd.keys.T
        self._show_camera_trackball_key = self.wnd.keys.B
        self._visualize_key = self.wnd.keys.Z
        self._lock_selection_key = self.wnd.keys.K
        self._mode_inspect_key = self.wnd.keys.I
        self._mode_view_key = self.wnd.keys.V
        self._go_to_frame_key = self.wnd.keys.G
        self._shortcut_names = {
            self.wnd.keys.SPACE: "Space",
            self.wnd.keys.B: "B",
            self.wnd.keys.C: "C",
            self.wnd.keys.D: "D",
            self.wnd.keys.I: "I",
            self.wnd.keys.L: "L",
            self.wnd.keys.K: "K",
            self.wnd.keys.O: "O",
            self.wnd.keys.P: "P",
            self.wnd.keys.S: "S",
            self.wnd.keys.T: "T",
            self.wnd.keys.X: "X",
            self.wnd.keys.G: "G",
            self.wnd.keys.Z: "Z",
        }

        # Disable exit on escape key
        self.window.exit_key = None

        # GUI
        self._render_gui = True
        self._exit_popup_open = False
        self._screenshot_popup_open = False
        self._screenshot_popup_just_opened = False
        self._export_usd_popup_open = False
        self._export_usd_popup_just_opened = False
        self._go_to_frame_popup_open = False
        self._go_to_frame_string = ""
        self._show_shortcuts_window = False
        self._mouse_position = (0, 0)
        self._modal_focus_count = 0

        self.server = None
        if C.server_enabled:
            self.server = ViewerServer(self, C.server_port)

    # noinspection PyAttributeOutsideInit
    def reset(self):
        if self.scene is not None:
            self.scene.release()

        # Setup scene
        self.scene = Scene()
        self.scene.camera = ViewerCamera(45.0)
        self.scene.custom_font = self.custom_font

        # Settings
        self.run_animations = C.run_animations
        self.playback_fps = C.playback_fps
        self.playback_without_skipping = False
        self.shadows_enabled = C.shadows_enabled
        self.auto_set_floor = C.auto_set_floor
        self.auto_set_camera_target = C.auto_set_camera_target
        self.backface_culling = C.backface_culling
        self.scene.light_mode = "dark" if C.dark_mode else "default"
        self.lock_selection = False
        self.visualize = False

        self._pan_camera = False
        self._rotate_camera = False
        self._past_frametimes = np.zeros([60]) - 1.0
        self._last_frame_rendered_at = 0

        # Outline colors
        self.outline_color = (0.3, 0.7, 1.0, 1)
        self.light_outline_color = (0.4, 0.4, 0.4, 1)
        self.selected_outline_color = (1.0, 0.86, 0.35, 1.0)

        # Mouse mesh intersection in inspect mode.
        self.mmi = None

        # Mouse selection
        self._move_threshold = 3
        self._mouse_moved = False
        self._mouse_down_position = np.array([0, 0])

        # Export settings
        self.export_animation = True
        self.export_animation_range = [0, -1]
        self.export_duration = 10
        self.export_format = "mp4"
        self.export_rotate_camera = False
        self.export_rotation_degrees = 360
        self.export_fps = self.playback_fps
        self.export_scale_factor = 1.0
        self.export_transparent = False
        self.export_quality = "medium"

        # Screenshot settings
        self.screenshot_transparent = False
        self.screenshot_name = None

        # Export usd settings
        self.export_usd_name = None
        self.export_usd_directory = False

        # Setup viewports
        self.viewports: List[Viewport] = []
        self._viewport_mode = None
        self.viewport_mode = "single"

        # Set the mode once the viewer has been completely initialized
        self.selected_mode = "view"

    @property
    def render_gui(self):
        return self._render_gui

    @render_gui.setter
    def render_gui(self, value):
        self._render_gui = value

    def _resize_viewports(self):
        """
        Computes the size of the viewports depending on the current mode.
        Uses the window buffer size to update the extents of all existing viewports.
        """
        mode = self.viewport_mode

        w, h = self.wnd.buffer_size
        if mode == "single":
            self.viewports[0].extents = [0, 0, w, h]
        elif mode == "split_v":
            self.viewports[0].extents = [0, 0, w // 2 - 1, h]
            self.viewports[1].extents = [w // 2 + 1, 0, w - w // 2 - 1, h]
        elif mode == "split_h":
            self.viewports[0].extents = [0, h // 2 + 1, w, h - h // 2 - 1]
            self.viewports[1].extents = [0, 0, w, h // 2 - 1]
        elif mode == "split_vh":
            self.viewports[0].extents = [0, h // 2 + 1, w // 2 - 1, h - h // 2 - 1]
            self.viewports[1].extents = [w // 2 + 1, h // 2 + 1, w - w // 2 - 1, h - h // 2 - 1]
            self.viewports[2].extents = [0, 0, w // 2 - 1, h // 2 - 1]
            self.viewports[3].extents = [w // 2 + 1, 0, w - w // 2 - 1, h // 2 - 1]
        else:
            raise ValueError(f"Invalid viewport mode: {mode}")

    def set_ortho_grid_viewports(self, invert=(False, False, False)):
        """
        Sets the viewports in a 2x2 grid with a viewport looking at the scene with an
        orthographic projection from each of the main axes.

        :param invert: a tuple of 3 booleans, for the x, y and z axis respectively.
        If False the camera for the respective axis is placed at the positive side looking
        towards the negative direction, and opposite otherwise.
        """
        self.viewport_mode = "split_vh"

        bounds = self.scene.bounds_without_floor
        center = bounds.mean(-1)
        size = bounds[:, 1] - bounds[:, 0]

        axis = [
            np.array([1, 0, 0]),
            np.array([0, 0, 1]),
            np.array([0, 1, 0]),
        ]
        for i in range(3):
            if invert[i]:
                axis[i] = -axis[i]

        up = [
            np.array([0, 1, 0]),
            np.array([0, 1, 0]),
            np.array([0, 0, 1]),
        ]

        ar = self.window.width / self.window.height

        def compute_ortho_size(w, h):
            if w / h > ar:
                # Limiting size is width, convert to limit on height.
                size = w / ar
            else:
                # Limiting size is height, return directly
                size = h
            return size * 0.55

        ortho_size = [
            compute_ortho_size(size[2], size[1]),
            compute_ortho_size(size[0], size[1]),
            compute_ortho_size(size[0], size[2]),
        ]

        for i, v in enumerate(self.viewports[1:]):
            v.camera = ViewerCamera(orthographic=ortho_size[i])
            v.camera.target = center
            v.camera.position = center + size[i] * axis[i]
            v.camera.up = up[i]

    @property
    def viewport_mode(self):
        """Getter for the current viewport mode"""
        return self._viewport_mode

    @viewport_mode.setter
    def viewport_mode(self, mode):
        """
        Setter for the current viewport mode, valid values are: "single", "split_v",  "split_h" and  "split_hv"
        which split the window in a 1x1, 1x2, 2x1 and 2x2 grid respectively.
        """
        if mode == self._viewport_mode:
            return

        self._viewport_mode = mode
        num_viewports = len(self.viewports)
        if mode == "single":
            new_num_viewports = 1
        elif mode == "split_v":
            new_num_viewports = 2
        elif mode == "split_h":
            new_num_viewports = 2
        elif mode == "split_vh":
            new_num_viewports = 4
        else:
            raise ValueError(f"Invalid viewport mode: {mode}")

        if num_viewports > new_num_viewports:
            # If the camera of one of the viewports that we are removing is selected
            # we just deselct it and remove it from the GUI, since it won't exist anymore.
            for v in self.viewports[new_num_viewports:]:
                if self.scene.selected_object == v.camera and isinstance(v.camera, ViewerCamera):
                    self.scene.select(None)
                    self.scene.gui_selected_object = None
            # Remove extra viewports.
            self.viewports = self.viewports[:new_num_viewports]
        elif num_viewports < new_num_viewports:
            for _ in range(num_viewports, new_num_viewports):
                camera = self.scene.camera
                if isinstance(self.scene.camera, ViewerCamera):
                    camera = camera.copy()
                self.viewports.append(Viewport([], camera))
        self._resize_viewports()

    def _init_scene(self):
        self.scene.make_renderable(self.ctx)
        if self.auto_set_floor:
            self.scene.auto_set_floor()
            self.scene.backface_culling = self.backface_culling
        if self.auto_set_camera_target:
            self.scene.auto_set_camera_target()

    def get_node_by_remote_uid(self, remote_uid: int, client: Tuple[str, str]):
        """
        Returns the Node corresponding to the remote uid and client passed in.

        :param remote_uid: the remote uid to look up.
        :param client: the client that created the node, this is the value of the 'client'
            parameter that was passed to process_message() when the message was received.
        :return: Node corresponding to the remote uid.
        """
        return self.server.get_node_by_remote_uid(remote_uid, client)

    def process_message(self, type: Message, remote_uid: int, args: list, kwargs: dict, client: Tuple[str, str]):
        """
        Default processing of messages received by the viewer.

        This method is called every time a new message has to be processed.
        It can be overriden to intercept messages before they are sent to
        the viewer and to add custom functionality.

        :param type: an integer id that represents the type of the message.
        :param remote_uid: the remote id of the node that this message refers to.
            This is an id generated by the client to reference nodes that
            it created. The viewer keeps track of the mapping from remote_uid
            to the local uid of the node.
            Use 'viewer.get_node_by_remote_uid(remote_uid)' to get the
            Node corresponding to this id.
        :param args: positional arguments received with the message.
        :param kwargs: keyword arguments received with the message.
        :param client: a tuple (ip, port) describing the address of the client
            that sent this message.
        """
        try:
            self.server.process_message(type, remote_uid, args, kwargs, client)
        except Exception as e:
            print(f"Exception while processing mesage: type = {type}, remote_uid = {remote_uid}:\n{e}")

    def send_message(self, msg, client: Tuple[str, str] = None):
        """
        Send a message to a single client or to all connected clients.

        :param msg: a python object that is serialized with pickle and sent to the client.
        :param client: a tuple (host, port) representing the client to which to send the message,
            if None the message is sent to all connected clients.
        """
        if self.server is not None:
            self.server.send_message(msg, client)

    def run(self, *args, log=True):
        """
        Enter a blocking visualization loop. This is built following `moderngl_window.run_window_config`.
        :param args: The arguments passed to `config_cls` constructor.
        :param log: Whether to log to the console.
        """
        self._init_scene()

        self.export_animation_range[-1] = self.scene.n_frames - 1

        self.timer.start()
        self._last_frame_rendered_at = self.timer.time

        while not self.window.is_closing:
            if self.server is not None:
                self.server.process_messages()

            current_time, delta = self.timer.next_frame()

            self.window.clear()
            self.window.render(current_time, delta)
            self.window.swap_buffers()
        _, duration = self.timer.stop()
        self.on_close()
        imgui.destroy_context(self.imgui_ctx)
        # Necessary for pyglet window, otherwise the window is not closed.
        self.window.close()
        self.window.destroy()
        if duration > 0 and log:
            print("Duration: {0:.2f}s @ {1:.2f} FPS".format(duration, self.window.frames / duration))

    def render(self, time, frame_time, export=False, transparent_background=False):
        """The main drawing function."""

        if self.run_animations:
            # Compute number of frames to advance by.
            frames = (int)((time - self._last_frame_rendered_at) * self.playback_fps)
            if frames > 0:
                if self.playback_without_skipping:
                    frames = 1
                self.scene.current_frame_id = (self.scene.current_frame_id + frames) % self.scene.n_frames
                self._last_frame_rendered_at += frames * (1.0 / self.playback_fps)

        # Update camera matrices that will be used for rendering
        if isinstance(self.scene.camera, ViewerCamera):
            self.scene.camera.update_animation(frame_time)

        # Update streamable captures.
        if not export:
            self.streamable_capture()

        # Render.
        self.renderer.render(
            self.scene,
            viewports=self.viewports,
            outline_color=self.outline_color,
            light_outline_color=self.light_outline_color,
            selected_outline_color=self.selected_outline_color,
            export=export,
            transparent_background=transparent_background,
            window_size=self.window_size,
            shadows_enabled=self.shadows_enabled,
            visualize=self.visualize,
        )

        if not export:
            # FPS accounting.
            self._past_frametimes[:-1] = self._past_frametimes[1:]
            self._past_frametimes[-1] = frame_time

            # Render the UI components.
            self.gui()

    def streamable_capture(self):
        # Collect all streamable nodes
        rs = self.scene.collect_nodes(obj_type=Streamable)
        for r in rs:
            r.capture()

    def prevent_background_interactions(self):
        """Prevent background interactions when hovering over any imgui window."""
        self.imgui_user_interacting = self.imgui.io.want_capture_mouse

    def toggle_animation(self, run: bool):
        self.run_animations = run
        if self.run_animations:
            self._last_frame_rendered_at = self.timer.time

    def reset_camera(self, viewport=None):
        if viewport is None:
            viewport = self.viewports[0]
        viewport.reset_camera()
        if viewport == self.viewports[0]:
            self.scene.camera = self.viewports[0].camera

    def set_temp_camera(self, camera, viewport=None):
        if viewport is None:
            viewport = self.viewports[0]
        viewport.set_temp_camera(camera)
        if viewport == self.viewports[0]:
            self.scene.camera = self.viewports[0].camera

    def lock_to_node(self, node: Node, relative_position, smooth_sigma=None, viewport=None):
        """
        Create and return a PinholeCamera that follows a node, the target of the camera is the center
        of the node at each frame and the camera is positioned with a constant offset (relative_position)
        from its target. See aitviewer.utils.path.lock_to_node for more details about parameters.
        The camera is set as the current viewer camera.
        """
        pos, tar = path.lock_to_node(node, relative_position, smooth_sigma=smooth_sigma)
        cam = PinholeCamera(pos, tar, self.window_size[0], self.window_size[1], viewer=self)
        self.scene.add(cam)
        self.set_temp_camera(cam, viewport)
        return cam

    def get_viewport_at_position(self, x: int, y: int) -> Union[Viewport, None]:
        x, y = self._mouse_to_buffer(x, y)
        for v in self.viewports:
            if v.contains(x, y):
                return v
        return None

    def gui(self):
        imgui.new_frame()

        # Add viewport separators to draw list.
        w, h = self.wnd.buffer_size
        draw = imgui.get_background_draw_list()
        c = 36 / 255
        color = imgui.get_color_u32_rgba(c, c, c, 1.0)
        if self.viewport_mode == "split_v" or self.viewport_mode == "split_vh":
            draw.add_rect_filled(w // 2 - 1, 0, w // 2 + 1, h, color)
        if self.viewport_mode == "split_h" or self.viewport_mode == "split_vh":
            draw.add_rect_filled(0, h // 2 - 1, w, h // 2 + 1, color)

        # Create a context menu when right clicking on the background.
        if (
            not any([imgui.is_window_hovered(), imgui.is_any_item_hovered()])
            and imgui.is_mouse_released(button=1)
            and not self._mouse_moved
        ):
            x, y = imgui.get_io().mouse_pos
            # Select the object under the cursor.
            if self.select_object(x, y) or not isinstance(self.scene.selected_object, Node):
                self._context_menu_position = (x, y)
                imgui.open_popup("Context Menu")

        # Draw the context menu for the selected object.
        if imgui.begin_popup("Context Menu"):
            if self.scene.selected_object is None or not isinstance(self.scene.selected_object, Node):
                imgui.close_current_popup()
            else:
                self.scene.selected_object.gui_context_menu(imgui, *self._context_menu_position)
            imgui.end_popup()

        # Reset user interacting state.
        self.imgui_user_interacting = False

        if self.render_gui:
            # Render user controls.
            for gc in self.gui_controls.values():
                gc()
        else:
            # If gui is disabled only render the go to frame window.
            self.gui_go_to_frame()

        # Contains live examples of all possible displays/controls - useful for browsing for new components
        # imgui.show_test_window()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

        self.prevent_background_interactions()

    def gui_scene(self):
        # Render scene GUI
        imgui.set_next_window_position(50, 50, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(self.window_size[0] * 0.25, self.window_size[1] * 0.7, imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("Editor", None)
        if expanded:
            self.scene.gui_editor(imgui, self.viewports, self.viewport_mode)
        imgui.end()

    def gui_menu(self):
        clicked_export = False
        clicked_screenshot = False
        clicked_export_usd = False

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                clicked_quit, selected_quit = imgui.menu_item("Quit", "Cmd+Q", False, True)
                if clicked_quit:
                    exit(1)

                clicked_export, selected_export = imgui.menu_item("Save as video..", None, False, True)

                clicked_screenshot, selected_screenshot = imgui.menu_item(
                    "Screenshot",
                    self._shortcut_names[self._screenshot_key],
                    False,
                    True,
                )

                clicked_export_usd, _ = imgui.menu_item("Save as USD", None, False, True)
                imgui.end_menu()

            if imgui.begin_menu("View", True):
                if imgui.begin_menu("Light modes"):
                    _, default = imgui.menu_item("Default", None, self.scene.light_mode == "default")
                    if default:
                        self.scene.light_mode = "default"

                    _, dark = imgui.menu_item(
                        "Dark",
                        self._shortcut_names[self._dark_mode_key],
                        self.scene.light_mode == "dark",
                    )
                    if dark:
                        self.scene.light_mode = "dark"

                    _, diffuse = imgui.menu_item("Diffuse", None, self.scene.light_mode == "diffuse")
                    if diffuse:
                        self.scene.light_mode = "diffuse"
                    imgui.end_menu()

                _, self.shadows_enabled = imgui.menu_item(
                    "Render Shadows",
                    self._shortcut_names[self._shadow_key],
                    self.shadows_enabled,
                    True,
                )

                _, self.lock_selection = imgui.menu_item(
                    "Lock selection",
                    self._shortcut_names[self._lock_selection_key],
                    self.lock_selection,
                    True,
                )

                if imgui.begin_menu("Viewports"):

                    def menu_entry(text, name):
                        if imgui.menu_item(text, None, self.viewport_mode == name)[1]:
                            self.viewport_mode = name

                    menu_entry("Single", "single")
                    menu_entry("Split vertical", "split_v")
                    menu_entry("Split horizontal", "split_h")
                    menu_entry("Split both", "split_vh")
                    imgui.separator()
                    if imgui.menu_item("Ortho grid +", None)[1]:
                        self.set_ortho_grid_viewports()
                    if imgui.menu_item("Ortho grid -", None)[1]:
                        self.set_ortho_grid_viewports((True, True, True))

                    imgui.end_menu()

                _, self.render_gui = imgui.menu_item("Render GUI", None, self.render_gui, True)

                imgui.end_menu()

            if imgui.begin_menu("Camera", True):
                _, self.scene.camera_target.enabled = imgui.menu_item(
                    "Show camera target",
                    self._shortcut_names[self._show_camera_target_key],
                    self.scene.camera_target.enabled,
                    True,
                )

                _, self.scene.trackball.enabled = imgui.menu_item(
                    "Show camera trackball",
                    self._shortcut_names[self._show_camera_trackball_key],
                    self.scene.trackball.enabled,
                    True,
                )

                clicked, _ = imgui.menu_item(
                    "Center view on selection",
                    self._shortcut_names[self._center_view_on_selection_key],
                    False,
                    isinstance(self.scene.selected_object, Node),
                )
                if clicked:
                    self.center_view_on_selection()

                is_ortho = False if self.viewports[0].using_temp_camera else self.scene.camera.is_ortho
                _, is_ortho = imgui.menu_item(
                    "Orthographic Camera",
                    self._shortcut_names[self._orthographic_camera_key],
                    is_ortho,
                    True,
                )

                if is_ortho and self.viewports[0].using_temp_camera:
                    self.reset_camera()
                self.scene.camera.is_ortho = is_ortho

                if imgui.begin_menu("Control modes", enabled=not self.viewports[0].using_temp_camera):

                    def mode(name, mode):
                        selected = imgui.menu_item(name, None, self.scene.camera.control_mode == mode)[1]
                        if selected:
                            self.reset_camera()
                            self.scene.camera.control_mode = mode

                    mode("Turntable", "turntable")
                    mode("Trackball", "trackball")
                    mode("First Person", "first_person")
                    imgui.end_menu()

                clicked_save_cam, selected_save_cam = imgui.menu_item(
                    "Save Camera", self._shortcut_names[self._save_cam_key], False, True
                )
                if clicked_save_cam:
                    self.reset_camera()
                    self.scene.camera.save_cam()

                clicked_load_cam, selected_load_cam = imgui.menu_item(
                    "Load Camera", self._shortcut_names[self._load_cam_key], False, True
                )
                if clicked_load_cam:
                    self.reset_camera()
                    self.scene.camera.load_cam()

                imgui.end_menu()

            if imgui.begin_menu("Mode", True):
                for id, mode in self.modes.items():
                    mode_clicked, _ = imgui.menu_item(mode["title"], mode["shortcut"], id == self.selected_mode, True)
                    if mode_clicked:
                        self.selected_mode = id

                imgui.end_menu()

            if imgui.begin_menu("Help", True):
                clicked, self._show_shortcuts_window = imgui.menu_item(
                    "Keyboard shortcuts", None, self._show_shortcuts_window
                )
                imgui.end_menu()

            if imgui.begin_menu("Debug", True):
                _, self.visualize = imgui.menu_item(
                    "Visualize debug texture",
                    self._shortcut_names[self._visualize_key],
                    self.visualize,
                    True,
                )

                imgui.end_menu()

            if self.server is not None:
                if imgui.begin_menu("Server", True):
                    imgui.text("Connected clients:")
                    imgui.separator()
                    for c in self.server.connections.keys():
                        imgui.text(f"{c[0]}:{c[1]}")
                    imgui.end_menu()

            imgui.end_main_menu_bar()

        if clicked_export:
            imgui.open_popup("Export Video")
            self.export_fps = self.playback_fps
            self.toggle_animation(False)

        if clicked_screenshot:
            self._screenshot_popup_just_opened = True

        if clicked_export_usd:
            self._export_usd_popup_just_opened = True

        self.gui_export_video()
        self.gui_screenshot()
        self.gui_export_usd()

    def gui_export_video(self):
        imgui.set_next_window_size(570, 0)
        if imgui.begin_popup_modal(
            "Export Video", flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SAVED_SETTINGS
        )[0]:
            if self.scene.n_frames == 1:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.2)
                self.export_animation = False

            if imgui.radio_button("Animation", self.export_animation):
                self.export_animation = True

            if self.scene.n_frames == 1:
                imgui.pop_style_var()
                self.export_animation = False

            imgui.same_line(spacing=15)
            if imgui.radio_button("360 shot", not self.export_animation):
                self.export_animation = False

            if self.export_animation:
                _, animation_range = imgui.drag_int2(
                    "Animation range",
                    *self.export_animation_range,
                    min_value=0,
                    max_value=self.scene.n_frames - 1,
                )
                if animation_range[0] != self.export_animation_range[0]:
                    self.export_animation_range[0] = animation_range[0]
                elif animation_range[-1] != self.export_animation_range[1]:
                    self.export_animation_range[1] = animation_range[1]

                _, self.playback_fps = imgui.drag_float(
                    "Playback fps",
                    self.playback_fps,
                    0.1,
                    min_value=1.0,
                    max_value=120.0,
                    format="%.1f",
                )
                imgui.same_line(spacing=10)
                speedup = self.playback_fps / self.scene.fps
                imgui.text(f"({speedup:.2f}x speed)")
            else:
                _, self.scene.current_frame_id = imgui.slider_int(
                    "Frame",
                    self.scene.current_frame_id,
                    min_value=0,
                    max_value=self.scene.n_frames - 1,
                )
                _, self.export_duration = imgui.drag_float(
                    "Duration (s)",
                    self.export_duration,
                    min_value=0.1,
                    max_value=10000.0,
                    change_speed=0.05,
                    format="%.1f",
                )
                duration = self.export_duration

            if self.export_animation:
                if isinstance(self.scene.camera, ViewerCamera):
                    _, self.export_rotate_camera = imgui.checkbox("Rotate camera", self.export_rotate_camera)
                else:
                    imgui.push_style_var(imgui.STYLE_ALPHA, 0.2)
                    imgui.checkbox("Rotate camera (only available for ViewerCamera)", False)
                    imgui.pop_style_var(1)

            if not self.export_animation or self.export_rotate_camera:
                _, self.export_rotation_degrees = imgui.drag_int(
                    "Rotation angle (degrees)", self.export_rotation_degrees
                )

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Output settings.
            imgui.text("Output")

            imgui.spacing()
            imgui.text("Format:")
            imgui.same_line()
            if imgui.radio_button("MP4", self.export_format == "mp4"):
                self.export_format = "mp4"
            imgui.same_line(spacing=15)
            if imgui.radio_button("WEBM", self.export_format == "webm"):
                self.export_format = "webm"
            imgui.same_line(spacing=15)
            if imgui.radio_button("GIF", self.export_format == "gif"):
                self.export_format = "gif"

            imgui.spacing()
            if self.export_format != "webm":
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.2)
                self.export_transparent = False

            _, self.export_transparent = imgui.checkbox("Transparent background", self.export_transparent)
            if self.export_format != "webm":
                imgui.pop_style_var()
                self.export_transparent = False

            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.text("Available only for WEBM format")
                imgui.end_tooltip()

            if self.export_format == "gif":
                max_output_fps = 30.0
            else:
                max_output_fps = 120.0
            self.export_fps = min(self.export_fps, max_output_fps)
            _, self.export_fps = imgui.drag_float(
                "fps",
                self.export_fps,
                0.1,
                min_value=1.0,
                max_value=max_output_fps,
                format="%.1f",
            )
            imgui.same_line(position=440)
            if imgui.button("1x##fps", width=35):
                self.export_fps = self.playback_fps
            imgui.same_line()
            if imgui.button("1/2x##fps", width=35):
                self.export_fps = self.playback_fps / 2
            imgui.same_line()
            if imgui.button("1/4x##fps", width=35):
                self.export_fps = self.playback_fps / 4

            imgui.spacing()
            imgui.spacing()
            imgui.text(
                f"Resolution: [{int(self.window_size[0] * self.export_scale_factor)}x{int(self.window_size[1] * self.export_scale_factor)}]"
            )
            _, self.export_scale_factor = imgui.drag_float(
                "Scale",
                self.export_scale_factor,
                min_value=0.01,
                max_value=1.0,
                change_speed=0.005,
                format="%.2f",
            )

            imgui.same_line(position=440)
            if imgui.button("1x##scale", width=35):
                self.export_scale_factor = 1.0
            imgui.same_line()
            if imgui.button("1/2x##scale", width=35):
                self.export_scale_factor = 0.5
            imgui.same_line()
            if imgui.button("1/4x##scale", width=35):
                self.export_scale_factor = 0.25

            imgui.spacing()
            if self.export_format == "mp4":
                imgui.text("Quality: ")
                imgui.same_line()
                if imgui.radio_button("high", self.export_quality == "high"):
                    self.export_quality = "high"
                imgui.same_line()
                if imgui.radio_button("medium", self.export_quality == "medium"):
                    self.export_quality = "medium"
                imgui.same_line()
                if imgui.radio_button("low", self.export_quality == "low"):
                    self.export_quality = "low"

            if self.export_animation:
                duration = (animation_range[1] - animation_range[0] + 1) / self.playback_fps
                # Compute exact number of frames if playback fps is an exact multiple of export fps.
                if np.fmod(self.playback_fps, self.export_fps) < 0.1:
                    playback_count = int(np.round(self.playback_fps / self.export_fps))
                    frames = (animation_range[1] - animation_range[0] + 1) // playback_count
                else:
                    frames = int(np.ceil(duration * self.export_fps))
            else:
                frames = int(np.ceil(duration * self.export_fps))

            imgui.spacing()
            if frames > 0:
                imgui.text(f"Duration: {duration:.2f}s ({frames} frames @ {self.export_fps:.2f}fps)")
            else:
                if self.export_animation:
                    imgui.text(f"Error: Animation range is empty")
                else:
                    imgui.text(f"Error: Duration is 0 seconds")
            imgui.spacing()

            # Draw a cancel and exit button on the same line using the available space
            button_width = (imgui.get_content_region_available()[0] - imgui.get_style().item_spacing[0]) * 0.5

            # Style the cancel with a grey color
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.5, 0.5, 0.5, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.6, 0.6, 0.6, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.7, 0.7, 0.7, 1.0)

            if imgui.button("Cancel", width=button_width):
                imgui.close_current_popup()

            imgui.pop_style_color()
            imgui.pop_style_color()
            imgui.pop_style_color()

            imgui.same_line()

            if frames <= 0:
                imgui.push_style_var(imgui.STYLE_ALPHA, 0.2)

            if imgui.button("Export", button_width) and frames >= 0:
                imgui.close_current_popup()
                self.export_video(
                    os.path.join(
                        C.export_dir,
                        "videos",
                        f"{self.window.title}.{self.export_format}",
                    ),
                    animation=self.export_animation,
                    animation_range=self.export_animation_range,
                    duration=self.export_duration,
                    frame=self.scene.current_frame_id,
                    output_fps=self.export_fps,
                    rotate_camera=not self.export_animation or self.export_rotate_camera,
                    rotation_degrees=self.export_rotation_degrees,
                    scale_factor=self.export_scale_factor,
                    transparent=self.export_transparent,
                    quality=self.export_quality,
                )

            if frames <= 0:
                imgui.pop_style_var()

            imgui.end_popup()

    def gui_playback(self):
        """GUI to control playback settings."""
        imgui.set_next_window_position(50, 100 + self.window_size[1] * 0.7, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(self.window_size[0] * 0.4, self.window_size[1] * 0.175, imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("Playback", None)
        if expanded:
            u, run_animations = imgui.checkbox(
                "Run animations [{}]".format(self._shortcut_names[self._pause_key]),
                self.run_animations,
            )
            if u:
                self.toggle_animation(run_animations)

            # Plot FPS
            frametime_avg = np.mean(self._past_frametimes[self._past_frametimes > 0.0])
            fps_avg = 1 / frametime_avg
            ms_avg = frametime_avg * 1000.0
            ms_last = self._past_frametimes[-1] * 1000.0

            imgui.plot_lines(
                "Internal {:.1f} fps @ {:.2f} ms/frame [{:.2f}ms]".format(fps_avg, ms_avg, ms_last),
                array("f", (1.0 / self._past_frametimes).tolist()),
                scale_min=0,
                scale_max=100.0,
                graph_size=(100, 20),
            )

            _, self.playback_fps = imgui.drag_float(
                f"Playback fps",
                self.playback_fps,
                0.1,
                min_value=1.0,
                max_value=120.0,
                format="%.1f",
            )
            imgui.same_line(spacing=10)
            speedup = self.playback_fps / self.scene.fps
            imgui.text(f"({speedup:.2f}x speed)")

            # Sequence Control
            # For simplicity, we allow the global sequence slider to only go as far as the shortest known sequence.
            n_frames = self.scene.n_frames

            _, self.scene.current_frame_id = imgui.slider_int(
                "Frame##r_global_seq_control",
                self.scene.current_frame_id,
                min_value=0,
                max_value=n_frames - 1,
            )
            self.prevent_background_interactions()
        if imgui.collapsing_header("Advanced options")[0]:
            _, self.playback_without_skipping = imgui.checkbox(
                "Playback without skipping", self.playback_without_skipping
            )
        imgui.end()

    def gui_shortcuts(self):
        if self._show_shortcuts_window:
            imgui.set_next_window_position(self.window_size[0] * 0.6, 200, imgui.FIRST_USE_EVER)
            imgui.set_next_window_size(self.window_size[0] * 0.35, 350, imgui.FIRST_USE_EVER)
            expanded, self._show_shortcuts_window = imgui.begin("Keyboard shortcuts", self._show_shortcuts_window)
            if expanded:
                for k, v in SHORTCUTS.items():
                    imgui.bullet_text(f"{k:5} - {v}")
            imgui.end()

    def gui_inspect(self):
        """GUI to control playback settings."""
        if self.selected_mode == "inspect":
            imgui.set_next_window_position(self.window_size[0] * 0.6, 50, imgui.FIRST_USE_EVER)
            imgui.set_next_window_size(self.window_size[0] * 0.35, 140, imgui.FIRST_USE_EVER)
            expanded, _ = imgui.begin("Inspect", None)
            if expanded:
                if self.mmi is not None:
                    for k, v in zip(self.mmi._fields, self.mmi):
                        imgui.text("{}: {}".format(k, v))

                self.prevent_background_interactions()
            imgui.end()

    def gui_go_to_frame(self):
        if self._go_to_frame_popup_open:
            imgui.open_popup("Go to frame##go-to-frame-popup")
            self._go_to_frame_string = str(self.scene.current_frame_id)

        imgui.set_next_window_size(300, 0)
        if imgui.begin_popup_modal(
            "Go to frame##go-to-frame-popup",
            flags=imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_TITLE_BAR
            | imgui.WINDOW_NO_SAVED_SETTINGS,
        )[0]:
            if self._go_to_frame_popup_open:
                imgui.set_keyboard_focus_here()
                u, self._go_to_frame_string = imgui.input_text(
                    "Go to frame",
                    self._go_to_frame_string,
                    64,
                    imgui.INPUT_TEXT_CHARS_DECIMAL
                    | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
                    | imgui.INPUT_TEXT_AUTO_SELECT_ALL,
                )
                if u:
                    try:
                        frame_id = int(self._go_to_frame_string)
                    except:
                        frame_id = -1
                        pass
                    if frame_id >= 0:
                        self.scene.current_frame_id = frame_id
                    self._go_to_frame_popup_open = False
                    imgui.close_current_popup()
            else:
                imgui.close_current_popup()
            imgui.end_popup()

    def gui_screenshot(self):
        if self._screenshot_popup_just_opened:
            self._screenshot_popup_just_opened = False
            self._screenshot_popup_open = True
            self.screenshot_name = None
            self._modal_focus_count = 2
            self.toggle_animation(False)
            imgui.open_popup("Screenshot##screenshot-popup")

        imgui.set_next_window_size(250, 0)
        if imgui.begin_popup_modal(
            "Screenshot##screenshot-popup",
            flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SAVED_SETTINGS,
        )[0]:
            if self._screenshot_popup_open:
                _, self.screenshot_transparent = imgui.checkbox("Transparent background", self.screenshot_transparent)
                if self.screenshot_name is None:
                    self.screenshot_name = "frame_{:0>6}.png".format(self.scene.current_frame_id)

                # HACK: we need to set the focus twice when the modal is first opened for it to take effect
                if self._modal_focus_count > 0:
                    self._modal_focus_count -= 1
                    imgui.set_keyboard_focus_here()
                _, self.screenshot_name = imgui.input_text(
                    "File name",
                    self.screenshot_name,
                    64,
                    imgui.INPUT_TEXT_AUTO_SELECT_ALL,
                )
                imgui.spacing()

                button_width = (imgui.get_content_region_available()[0] - imgui.get_style().item_spacing[0]) * 0.5

                # Style the cancel with a grey color
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.5, 0.5, 0.5, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.6, 0.6, 0.6, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.7, 0.7, 0.7, 1.0)

                if imgui.button("cancel", width=button_width):
                    imgui.close_current_popup()
                    self._screenshot_popup_open = False

                imgui.pop_style_color()
                imgui.pop_style_color()
                imgui.pop_style_color()

                imgui.same_line()
                if imgui.button("save", button_width):
                    if self.screenshot_name:
                        self.take_screenshot(self.screenshot_name, self.screenshot_transparent)
                    imgui.close_current_popup()
                    self._screenshot_popup_open = False

            else:
                imgui.close_current_popup()
            imgui.end_popup()

    def gui_export_usd(self):
        if self._export_usd_popup_just_opened:
            self._export_usd_popup_just_opened = False
            self._export_usd_popup_open = True
            self.export_usd_name = None
            self._modal_focus_count = 2
            self.toggle_animation(False)
            imgui.open_popup("Export USD##export-usd-popup")

        imgui.set_next_window_size(570, 0)
        if imgui.begin_popup_modal(
            "Export USD##export-usd-popup",
            flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SAVED_SETTINGS,
        )[0]:
            if self._export_usd_popup_open:
                # Export selection tree.
                def tree(nodes):
                    # Nodes GUI
                    for r in nodes:
                        # Skip nodes that shouldn't appear in the hierarchy.
                        if not r.show_in_hierarchy:
                            continue

                        # Visibility
                        curr_enabled = r.export_usd_enabled
                        if not curr_enabled:
                            imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 1.0, 0.4)

                        # Title
                        imgui.push_font(self.custom_font)
                        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 2))

                        flags = imgui.TREE_NODE_OPEN_ON_ARROW | imgui.TREE_NODE_FRAME_PADDING
                        if r.export_usd_expanded:
                            flags |= imgui.TREE_NODE_DEFAULT_OPEN
                        if not any(c.show_in_hierarchy for c in r.nodes):
                            flags |= imgui.TREE_NODE_LEAF
                        r.export_usd_expanded = imgui.tree_node(
                            "{} {}##tree_node_{}".format(r.icon, r.name, r.unique_name), flags
                        )

                        imgui.pop_style_var()
                        imgui.pop_font()

                        if r != self:
                            # Aligns checkbox to the right side of the window
                            # https://github.com/ocornut/imgui/issues/196
                            imgui.same_line(position=imgui.get_window_content_region_max().x - 25)
                            eu, enabled = imgui.checkbox("##enabled_r_{}".format(r.unique_name), r.export_usd_enabled)
                            if eu:
                                r.export_usd_enabled = enabled

                        if r.export_usd_expanded:
                            # Recursively render children nodes
                            tree(r.nodes)
                            imgui.tree_pop()

                        if not curr_enabled:
                            imgui.pop_style_color(1)

                imgui.begin_child(
                    "export_select", height=300, border=True, flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR
                )
                tree(self.scene.nodes)
                imgui.end_child()

                if self.export_usd_name is None:
                    self.export_usd_name = "scene"

                # HACK: we need to set the focus twice when the modal is first opened for it to take effect.
                if self._modal_focus_count > 0:
                    self._modal_focus_count -= 1
                    imgui.set_keyboard_focus_here()
                _, self.export_usd_name = imgui.input_text(
                    "File name",
                    self.export_usd_name,
                    64,
                    imgui.INPUT_TEXT_AUTO_SELECT_ALL,
                )

                _, self.export_usd_directory = imgui.checkbox(
                    "Export as directory with textures", self.export_usd_directory
                )

                imgui.spacing()

                button_width = (imgui.get_content_region_available()[0] - imgui.get_style().item_spacing[0]) * 0.5

                # Style the cancel with a grey color
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.5, 0.5, 0.5, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.6, 0.6, 0.6, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.7, 0.7, 0.7, 1.0)

                if imgui.button("cancel", width=button_width):
                    imgui.close_current_popup()
                    self._export_usd_popup_open = False

                imgui.pop_style_color()
                imgui.pop_style_color()
                imgui.pop_style_color()

                imgui.same_line()
                if imgui.button("save", button_width):
                    if self.export_usd_name:
                        self.export_usd(
                            os.path.join(C.export_dir, "usd", f"{self.export_usd_name}"),
                            self.export_usd_directory,
                            False,
                            False,
                        )
                    imgui.close_current_popup()
                    self._export_usd_popup_open = False

            else:
                imgui.close_current_popup()
            imgui.end_popup()

    def gui_exit(self):
        if self._exit_popup_open:
            imgui.open_popup("Exit##exit-popup")

        if imgui.begin_popup_modal(
            "Exit##exit-popup", flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SAVED_SETTINGS
        )[0]:
            if self._exit_popup_open:
                imgui.text("Are you sure you want to exit?")
                imgui.spacing()

                # Draw a cancel and exit button on the same line using the available space
                button_width = (imgui.get_content_region_available()[0] - imgui.get_style().item_spacing[0]) * 0.5

                # Style the cancel with a grey color
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.5, 0.5, 0.5, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.6, 0.6, 0.6, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.7, 0.7, 0.7, 1.0)

                if imgui.button("cancel", width=button_width):
                    imgui.close_current_popup()
                    self._exit_popup_open = False

                imgui.pop_style_color()
                imgui.pop_style_color()
                imgui.pop_style_color()

                imgui.same_line()
                if imgui.button("exit", button_width):
                    self.window.close()
            else:
                imgui.close_current_popup()
            imgui.end_popup()

    def _mouse_to_buffer(self, x: int, y: int):
        return int(x * self.wnd.pixel_ratio), int(self.wnd.buffer_height - (y * self.wnd.pixel_ratio))

    def _mouse_to_viewport(self, x: int, y: int, viewport: Viewport):
        x, y = self._mouse_to_buffer(x, y)
        x = x - viewport.extents[0]
        y = y - viewport.extents[1]
        return x, viewport.extents[3] - y

    def mesh_mouse_intersection(self, x: int, y: int):
        """Given an x/y screen coordinate, get the intersected object, triangle id, and xyz point in camera space"""
        x, y = self._mouse_to_buffer(x, y)
        point_world, obj_id, tri_id, instance_id = self.renderer.read_fragmap_at_pixel(x, y)

        if obj_id >= 0 and tri_id >= 0:
            node = self.scene.get_node_by_uid(obj_id)
            # Camera space to world space
            point_local = (np.linalg.inv(node.model_matrix) @ np.append(point_world, 1.0))[:-1]
            if isinstance(node, Meshes) or isinstance(node, Billboard) or isinstance(node, VariableTopologyMeshes):
                vert_id = node.closest_vertex_in_triangle(tri_id, point_local)
                bc_coords = node.get_bc_coords_from_points(tri_id, [point_local])
            elif isinstance(node, PointClouds):
                vert_id = tri_id
                bc_coords = np.array([1, 0, 0])
            else:
                vert_id = 0
                bc_coords = np.array([0, 0, 0])

            return MeshMouseIntersection(node, instance_id, tri_id, vert_id, point_world, point_local, bc_coords)

        return None

    def select_object(self, x: int, y: int):
        """Selects the object at pixel coordinates x, y, returns True if an object is selected"""
        mmi = self.mesh_mouse_intersection(x, y)
        if mmi is not None:
            node = mmi.node

            # Traverse all parents until one is found that is selectable
            while node.parent is not None:
                if node.is_selectable:
                    # If the selection is locked only allow selecting the locked object
                    if not self.lock_selection or node == self.scene.selected_object:
                        self.scene.select(node, mmi.node, mmi.instance_id, mmi.tri_id)
                        return True
                    else:
                        return False
                node = node.parent

        return False

    def center_view_on_selection(self):
        self.center_view_on_node(self.scene.selected_object, with_animation=True)

    def center_view_on_node(self, node, with_animation=False):
        if isinstance(node, Node):
            self.reset_camera()
            forward = self.scene.camera.forward
            bounds = node.current_bounds
            diag = np.linalg.norm(bounds[:, 0] - bounds[:, 1])
            dist = max(0.01, diag * 1.3)

            target = bounds.mean(-1)
            position = target - forward * dist
            if with_animation:
                self.scene.camera.move_with_animation(position, target, 0.25)
            else:
                self.scene.camera.position = position
                self.scene.camera.target = target

    def resize(self, width: int, height: int):
        self.window_size = (width, height)
        self._resize_viewports()
        self.renderer.resize(width, height)
        self.imgui.resize(width, height)

    def files_dropped_event(self, x: int, y: int, paths):
        for path in paths:
            base, ext = os.path.splitext(path)
            if ext == ".obj" or ext == ".ply":
                import trimesh

                obj = trimesh.load(path)
                obj_mesh = Meshes(obj.vertices, obj.faces, name=os.path.basename(base))
                self.scene.add(obj_mesh)

    def key_event(self, key, action, modifiers):
        self.imgui.key_event(key, action, modifiers)

        # Handle keyboard shortcuts when the exit modal is open
        if action == self.wnd.keys.ACTION_PRESS and self._exit_popup_open:
            if key == self.wnd.keys.ENTER:
                self.window.close()
            elif key == self._exit_key:
                self._exit_popup_open = False
            return

        if action == self.wnd.keys.ACTION_PRESS and (self._screenshot_popup_open or self._export_usd_popup_open):
            if key == self.wnd.keys.ENTER:
                if self.screenshot_name:
                    self.take_screenshot(self.screenshot_name, self.screenshot_transparent)
                    self._screenshot_popup_open = False
                if self.export_usd_name:
                    self.export_usd(
                        os.path.join(C.export_dir, "usd", f"{self.export_usd_name}"), self.export_usd_directory, True
                    )
                    self._export_usd_popup_open = False
            elif key == self._exit_key:
                self._screenshot_popup_open = False
                self._export_usd_popup_open = False
            return

        if action == self.wnd.keys.ACTION_PRESS and self._go_to_frame_popup_open:
            if key == self._exit_key:
                self._go_to_frame_popup_open = False
            return

        if action == self.wnd.keys.ACTION_PRESS and not self.render_gui:
            if key == self._exit_key:
                self.render_gui = True
                return

        if self.imgui.io.want_capture_keyboard:
            return

        if action == self.wnd.keys.ACTION_PRESS:
            if key == self._go_to_frame_key:
                self._go_to_frame_popup_open = True

            if key == self._exit_key:
                self._exit_popup_open = True

            if key == self._pause_key:
                self.toggle_animation(not self.run_animations)

            elif key == self._next_frame_key:
                if not self.run_animations:
                    self.scene.next_frame()

            elif key == self._previous_frame_key:
                if not self.run_animations:
                    self.scene.previous_frame()

            elif key == self._shadow_key:
                self.shadows_enabled = not self.shadows_enabled

            elif key == self._show_camera_target_key:
                self.scene.camera_target.enabled = not self.scene.camera_target.enabled

            elif key == self._show_camera_trackball_key:
                self.scene.trackball.enabled = not self.scene.trackball.enabled

            elif key == self._orthographic_camera_key:
                self.reset_camera()
                self.scene.camera.is_ortho = not self.scene.camera.is_ortho

            elif key == self._center_view_on_selection_key:
                self.center_view_on_selection()

            elif key == self._mode_view_key:
                self.selected_mode = "view"

            elif key == self._mode_inspect_key:
                self.selected_mode = "inspect"

            elif key == self._dark_mode_key:
                self.scene.light_mode = "dark" if self.scene.light_mode != "dark" else "default"

            elif key == self._screenshot_key:
                self._screenshot_popup_just_opened = True

            elif key == self._save_cam_key:
                self.reset_camera()
                self.scene.camera.save_cam()
            elif key == self._load_cam_key:
                self.reset_camera()
                self.scene.camera.load_cam()

            elif key == self._visualize_key:
                self.visualize = not self.visualize

            elif key == self._lock_selection_key:
                self.lock_selection = not self.lock_selection

            # Pass onto selected object
            if isinstance(self.scene.gui_selected_object, Node):
                self.scene.gui_selected_object.key_event(key, self.wnd.keys)

        if action == self.wnd.keys.ACTION_RELEASE:
            pass

    def mouse_position_event(self, x, y, dx, dy):
        self._mouse_position = (x, y)
        self.imgui.mouse_position_event(x, y, dx, dy)

        if self.selected_mode == "inspect":
            self.mmi = self.mesh_mouse_intersection(x, y)

    def mouse_press_event(self, x: int, y: int, button: int):
        self.imgui.mouse_press_event(x, y, button)

        if not self.imgui_user_interacting:
            self._moving_camera_viewport = self.get_viewport_at_position(x, y)
            if self._moving_camera_viewport is None:
                return

            # Pan or rotate camera on middle click.
            if button == self._middle_mouse_button:
                self._pan_camera = self.wnd.modifiers.shift
                self._rotate_camera = not self.wnd.modifiers.shift

            # Rotate camera on left click.
            if button == self._left_mouse_button:
                self.reset_camera(self._moving_camera_viewport)
                self._rotate_camera = True
                self._pan_camera = False
                x, y = self._mouse_to_viewport(x, y, self._moving_camera_viewport)
                self._moving_camera_viewport.camera.rotate_start(x, y, *self._moving_camera_viewport.extents[2:])

            if button == self._right_mouse_button:
                self._pan_camera = True
                self._rotate_camera = False

            self._mouse_down_position = np.array([x, y])
            self._mouse_moved = False

    def mouse_release_event(self, x: int, y: int, button: int):
        self.imgui.mouse_release_event(x, y, button)

        self._pan_camera = False
        self._rotate_camera = False
        self._moving_camera_viewport = None

        if not self.imgui_user_interacting:
            # Select the mesh under the cursor on left click.
            if button == self._left_mouse_button and not self._mouse_moved:
                if not self.select_object(x, y):
                    # If selection is enabled and nothing was selected clear the previous selection
                    if not self.lock_selection:
                        self.scene.select(None)

    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        self.imgui.mouse_drag_event(x, y, dx, dy)

        if not self.imgui_user_interacting:
            if self._moving_camera_viewport is None:
                return

            if self._pan_camera:
                self.reset_camera(self._moving_camera_viewport)
                self._moving_camera_viewport.camera.pan(dx, dy)

            if self._rotate_camera:
                self.reset_camera(self._moving_camera_viewport)
                x, y = self._mouse_to_viewport(x, y, self._moving_camera_viewport)
                self._moving_camera_viewport.camera.rotate(x, y, dx, dy, *self._moving_camera_viewport.extents[2:])

            if (
                not self._mouse_moved
                and np.linalg.norm(np.array([x, y]) - self._mouse_down_position) > self._move_threshold
            ):
                self._mouse_moved = True

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        self.imgui.mouse_scroll_event(x_offset, y_offset)

        if not self.imgui_user_interacting:
            v = self.get_viewport_at_position(*self._mouse_position)
            if v:
                self.reset_camera(v)
                v.camera.dolly_zoom(np.sign(y_offset), self.wnd.modifiers.shift, self.wnd.modifiers.ctrl)

    def unicode_char_entered(self, char):
        self.imgui.unicode_char_entered(char)

    def save_current_frame_as_image(self, path, scale_factor=None, alpha=False):
        """Saves the current frame as an image to disk."""
        image = self.get_current_frame_as_image(alpha)
        if scale_factor is not None and scale_factor != 1.0:
            w = int(image.width * scale_factor)
            h = int(image.height * scale_factor)
            image = image.resize((w, h), Image.LANCZOS)
        image.save(path)

    def get_current_frame_as_image(self, alpha=False):
        return self.renderer.get_current_frame_as_image(alpha)

    def get_current_mask_image(self, color_map: Dict[int, Tuple[int, int, int]] = None, id_map: Dict[int, int] = None):
        return self.renderer.get_current_mask_image(color_map, id_map)

    def get_current_mask_ids(self, id_map: Dict[int, int] = None):
        return self.renderer.get_current_mask_ids(id_map)

    def get_current_depth_image(self):
        return self.renderer.get_current_depth_image(self.scene.camera)

    def on_close(self):
        """
        Clean up before destroying the window
        """
        # Shut down all streams.
        for s in self.scene.collect_nodes(obj_type=Streamable):
            s.stop()

        # Shut down server.
        if self.server is not None:
            self.server.close()

        # Clear the lru_cache on all shaders, we do this so that future instances of the viewer
        # have to recompile shaders with the current moderngl context.
        # See issue #12 https://github.com/eth-ait/aitviewer/issues/12
        clear_shader_cache()

    def take_screenshot(self, file_name=None, transparent_background=False):
        """Save the current frame to an image in the screenshots directory inside the export directory"""
        if file_name is None:
            file_name = "frame_{:0>6}.png".format(self.scene.current_frame_id)
        if not file_name.endswith(".png"):
            file_name += ".png"
        file_path = os.path.join(C.export_dir, "screenshots", file_name)
        self.export_frame(file_path, transparent_background=transparent_background)
        print(f"Screenshot saved to {file_path}")

    def export_frame(self, file_path, scale_factor: float = None, transparent_background=False):
        """Save the current frame to an image.
        :param file_path: the path where the image is saved.
        :param scale_factor: a scale factor used to scale the image. If None no scale factor is used and
          the image will have the same size as the viewer.
        """
        dir = os.path.dirname(file_path)
        if dir:
            os.makedirs(dir, exist_ok=True)

        # Store run_animation old value and set it to false.
        run_animations = self.run_animations
        self.run_animations = False

        # Render and save frame.
        self.render(0, 0, export=True, transparent_background=transparent_background)
        self.save_current_frame_as_image(file_path, scale_factor, transparent_background)

        # Restore run animation and update last frame rendered time.
        self.run_animations = run_animations
        self._last_frame_rendered_at = self.timer.time

    def export_usd(self, path: str, export_as_directory=False, verbose=False, ascii=False):
        from pxr import Usd, UsdGeom

        extension = ".usd" if not ascii else ".usda"
        if export_as_directory:
            if path.endswith(extension):
                directory = path[:-4]
            else:
                directory = path
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, os.path.basename(directory) + extension)
        else:
            directory = None
            if not path.endswith(extension):
                path += extension

        # Create a new file and setup scene parameters.
        stage = Usd.Stage.CreateNew(path)
        stage.SetStartTimeCode(1)
        stage.SetEndTimeCode(self.scene.n_frames)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)

        # Recursively export the scene.
        self.scene.export_usd(stage, "", directory, verbose)

        # Save file.
        stage.Save()

    def export_video(
        self,
        output_path,
        frame_dir=None,
        animation=True,
        animation_range=None,
        duration=10.0,
        frame=None,
        output_fps=60.0,
        rotate_camera=False,
        rotation_degrees=360.0,
        scale_factor=None,
        transparent=False,
        quality="medium",
        ensure_no_overwrite=True,
    ):
        # Load this module to reduce load time.
        import skvideo.io

        if rotate_camera and not isinstance(self.scene.camera, ViewerCamera):
            print("Cannot export a video with camera rotation while using a camera that is not a ViewerCamera")
            return

        if frame_dir is None and output_path is None:
            print("You must either specify a path where to render the images to or where to save the video to")
            return

        if frame_dir is not None:
            # We want to avoid overriding anything in an existing directory, so add suffixes.
            # We always do this, even if `ensure_no_overwrite` is False, because this might lead to unexpected videos
            # if data already exists in this directory.
            format_str = "{:0>4}"
            counter = 0
            candidate_dir = os.path.join(frame_dir, format_str.format(counter))
            while os.path.exists(candidate_dir):
                counter += 1
                candidate_dir = os.path.join(frame_dir, format_str.format(counter))
            frame_dir = os.path.abspath(candidate_dir)

            # The frame dir does not yet exist (we've made sure of it).
            os.makedirs(frame_dir)

        # Store the current camera and create a copy of it if required.
        saved_camera = self.scene.camera
        if rotate_camera:
            self.scene.camera = copy.deepcopy(self.scene.camera)

        # Remember viewer data.
        saved_curr_frame = self.scene.current_frame_id
        saved_run_animations = self.run_animations

        if animation:
            if animation_range is None:
                animation_range = [0, self.scene.n_frames - 1]

            if animation_range[1] < animation_range[0]:
                print("No frames rendered.")
                return

            # Compute duration of the animation at given playback speed
            animation_frames = (animation_range[1] - animation_range[0]) + 1
            duration = animation_frames / self.playback_fps

            # Setup viewer for rendering the animation
            self.run_animations = True
            self.scene.current_frame_id = animation_range[0]
            self._last_frame_rendered_at = 0
        else:
            self.run_animations = False
            self.scene.current_frame_id = frame

        # Compute exact number of frames if we have the playback fps is an exact multiple of the output fps
        if animation and np.fmod(self.playback_fps, output_fps) < 0.1:
            playback_count = int(np.round(self.playback_fps / output_fps))
            frames = (animation_range[1] - animation_range[0] + 1) // playback_count
            self.run_animations = False
            exact_playback = True
        else:
            frames = int(np.ceil(duration * output_fps))
            exact_playback = False

        dt = 1 / output_fps
        time = 0

        # Compute camera speed.
        az_delta = np.radians(rotation_degrees) / frames

        # Initialize video writer.
        if output_path is not None:
            path_video, path_gif, is_gif = get_video_paths(output_path, ensure_no_overwrite)
            pix_fmt = "yuva420p" if transparent else "yuv420p"
            outputdict = {
                "-pix_fmt": pix_fmt,
                "-vf": "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Avoid error when image res is not divisible by 2.
                "-r": str(output_fps),
            }

            if path_video.endswith("mp4"):
                quality_to_crf = {
                    "high": 23,
                    "medium": 28,
                    "low": 33,
                }
                # MP4 specific options
                outputdict.update(
                    {
                        "-c:v": "libx264",
                        "-preset": "slow",
                        "-profile:v": "high",
                        "-level:v": "4.0",
                        "-crf": str(quality_to_crf[quality]),
                    }
                )

            writer = skvideo.io.FFmpegWriter(
                path_video,
                inputdict={
                    "-framerate": str(output_fps),
                },
                outputdict=outputdict,
            )

        for i in tqdm(range(frames), desc="Rendering frames"):
            if rotate_camera:
                self.scene.camera.rotate_azimuth(az_delta)

            self.render(time, time + dt, export=True, transparent_background=transparent)
            img = self.get_current_frame_as_image(alpha=transparent)

            # Scale image by the scale factor.
            if scale_factor is not None and scale_factor != 1.0:
                w = int(img.width * scale_factor)
                h = int(img.height * scale_factor)
                img = img.resize((w, h), Image.LANCZOS)

            # Store the image to disk if a directory for frames was given.
            if frame_dir is not None:
                img_name = os.path.join(frame_dir, "frame_{:0>6}.png".format(i))
                img.save(img_name)

            # Write the frame to the video writer.
            if output_path is not None:
                writer.writeFrame(np.array(img))

            if exact_playback:
                self.scene.current_frame_id = self.scene.current_frame_id + playback_count

            time += dt

        # Save the video.
        if output_path is not None:
            writer.close()
            if is_gif:
                # Convert to gif.
                video_to_gif(path_video, path_gif, remove=True)

                print(f"GIF saved to {os.path.abspath(path_gif)}")
            else:
                print(f"Video saved to {os.path.abspath(output_path)}")
        else:
            print(f"Frames saved to {os.path.abspath(frame_dir)}")

        # Reset viewer data.
        self.scene.camera = saved_camera
        self.scene.current_frame_id = saved_curr_frame
        self.run_animations = saved_run_animations
        self._last_frame_rendered_at = self.timer.time

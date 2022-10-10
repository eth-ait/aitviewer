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
import copy
import imgui
import moderngl
import moderngl_window
import numpy as np
import os
import skvideo.io
import struct
import trimesh

from array import array
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.meshes import Meshes, VariableTopologyMeshes
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.scene.camera import ViewerCamera
from aitviewer.scene.scene import Scene
from aitviewer.scene.node import Node
from aitviewer.shaders import clear_shader_cache
from aitviewer.streamables.streamable import Streamable
from aitviewer.utils import PerfTimer
from aitviewer.utils.utils import get_video_paths, video_to_gif
from collections import namedtuple
from moderngl_window import activate_context
from moderngl_window import geometry
from moderngl_window import get_local_window_cls
from moderngl_window.integrations.imgui import ModernglWindowRenderer
from moderngl_window.opengl.vao import VAO
from omegaconf.dictconfig import DictConfig
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Tuple, Union


MeshMouseIntersection = namedtuple('MeshMouseIntersection', 'node tri_id vert_id point_world point_local bc_coords')

SHORTCUTS = {
    "SPACE": "Start/stop playing animation.",
    ".": "Go to next frame.",
    ",": "Go to previous frame.",
    "X": "Center view on the selected object.",
    "O": "Enable/disable orthographic camera.",
    "T": "Show the camera target in the scene.",
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
    resource_dir = Path(__file__).parent / 'shaders'
    size_mult = 1.0
    samples = 4
    gl_version = (4, 0)
    window_type = C.window_type

    def __init__(self, title="AITViewer", size: Tuple[int, int]=None, config: Union[DictConfig, dict]=None, **kwargs):
        """
        Initializer.
        :param title: Window title
        :param size: Window size as (width, height) tuple, if None uses the size from the configuration file
        :param kwargs: kwargs.
        """

        # Update config with user parameter.
        if config is not None:
            C.update_conf(config)

        # Window Setup (Following `moderngl_window.run_window_config`).
        if self.window_type != 'headless':
            self.window_type = C.window_type
        base_window_cls = get_local_window_cls(self.window_type)

        # If no size is provided use the size from the configuration file
        if size is None:
            size = C.window_width, C.window_height

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

        self.timer = PerfTimer()
        self.ctx = self.window.ctx
        super().__init__(self.ctx, self.window, self.timer)

        # Create GUI context
        self.imgui_ctx = imgui.create_context()
        self.imgui = ModernglWindowRenderer(self.wnd)
        self.imgui_user_interacting = False

        # Shaders for rendering the shadow map
        self.raw_depth_prog = self.load_program('shadow_mapping/raw_depth.glsl')
        self.depth_only_prog = self.load_program('shadow_mapping/depth_only.glsl')

        # Shaders for mesh mouse intersection
        self.frag_map_prog = self.load_program('fragment_picking/frag_map.glsl')
        self.frag_pick_prog = self.load_program('fragment_picking/frag_pick.glsl')
        self.frag_pick_prog['position_texture'].value = 0  # Read from texture channel 0
        self.frag_pick_prog['obj_info_texture'].value = 1  # Read from texture channel 0
        self.picker_output = self.ctx.buffer(reserve=5*4)  # 3 floats, 2 ints
        self.picker_vao = VAO(mode=moderngl.POINTS)

        # Shaders for drawing outlines
        self.outline_prepare_prog = self.load_program('outline/outline_prepare.glsl')
        self.outline_draw_prog = self.load_program('outline/outline_draw.glsl')
        self.outline_quad = geometry.quad_2d(size=(2.0, 2.0), pos=(0.0, 0.0))

        # Create framebuffers
        self.create_framebuffers()

        # Custom UI Font
        self.font_dir = Path(__file__).parent / 'resources' / 'fonts'
        self.fonts = imgui.get_io().fonts
        self.custom_font = self.fonts.add_font_from_file_ttf(os.path.join(self.font_dir, 'Custom.ttf'), 15)
        self.imgui.refresh_font_texture()

        self.modes = {
            'view': {'title': ' View', 'shortcut': 'V'},
            'inspect': {'title': ' Inspect', 'shortcut': 'I'},
        }

        self.gui_controls = {
            'menu': self.gui_menu,
            'scene': self.gui_scene,
            'playback': self.gui_playback,
            'inspect': self.gui_inspect,
            'shortcuts': self.gui_shortcuts,
            'exit': self.gui_exit,
        }

        # Debug
        self.vis_prog =  self.load_program('visualize.glsl')
        self.vis_quad = geometry.quad_2d(size=(0.9, 0.9), pos=(0.5, 0.5))

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
        self._visualize_key = self.wnd.keys.Z
        self._lock_selection_key = self.wnd.keys.K
        self._mode_inspect_key = self.wnd.keys.I
        self._mode_view_key = self.wnd.keys.V
        self._shortcut_names = {self.wnd.keys.SPACE: "Space",
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
                                self.wnd.keys.Z: "Z"}

        # Disable exit on escape key
        self.window.exit_key = None

        # GUI
        self._exit_popup_open = False
        self._show_shortcuts_window = False

    def create_framebuffers(self):
        """
        Create all framebuffers which depend on the window size.
        This is called once at startup and every time the window is resized.
        """
        # Mesh mouse intersection
        self.offscreen_p_depth = self.ctx.depth_texture(self.wnd.buffer_size)
        self.offscreen_p_viewpos = self.ctx.texture(self.wnd.buffer_size, 4, dtype='f4')
        self.offscreen_p_tri_id = self.ctx.texture(self.wnd.buffer_size, 4, dtype='f4')
        self.offscreen_p = self.ctx.framebuffer(
            color_attachments=[
                self.offscreen_p_viewpos,
                self.offscreen_p_tri_id
            ],
            depth_attachment=self.offscreen_p_depth
        )
        self.offscreen_p_tri_id.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Outline rendering
        self.outline_texture = self.ctx.texture(self.wnd.buffer_size, 1, dtype='f4')
        self.outline_framebuffer = self.ctx.framebuffer(color_attachments=[self.outline_texture])

    def reset(self):
        if self.scene is not None:
            self.scene.release()

        # Setup scene
        self.scene = Scene()
        self.scene.camera = ViewerCamera(45.0)
        self.scene.custom_font = self.custom_font

        # Settings
        self.run_animations = C.run_animations
        self.dark_mode = C.dark_mode
        self.playback_fps = C.playback_fps
        self.shadows_enabled = C.shadows_enabled
        self.auto_set_floor = C.auto_set_floor
        self.auto_set_camera_target = C.auto_set_camera_target
        self.backface_culling = C.backface_culling
        self.lock_selection = False
        self.show_camera_target = False
        self.selected_mode = 'view'
        self.visualize = False

        self._pan_camera = False
        self._rotate_camera = False
        self._using_temp_camera = False
        self._past_frametimes = np.zeros([60]) - 1.0
        self._last_frame_rendered_at = 0

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
        self.export_as_gif = False
        self.export_rotate_camera = False
        self.export_seconds_per_rotation = 10
        self.export_fps = self.playback_fps
        self.export_scale_factor = 1.0

    def _init_scene(self):
        self.scene.make_renderable(self.ctx)
        if self.auto_set_floor:
            self.scene.auto_set_floor()
            self.scene.backface_culling = self.backface_culling
        if self.auto_set_camera_target:
            self.scene.auto_set_camera_target()
        self.scene.set_lights(self.dark_mode)

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

    def render(self, time, frame_time, export=False):
        """The main drawing function."""

        if self.run_animations:
            # Compute number of frames to advance by.
            frames = (int)((time - self._last_frame_rendered_at) * self.playback_fps)
            if frames > 0:
                self.scene.current_frame_id = (self.scene.current_frame_id + frames) % self.scene.n_frames
                self._last_frame_rendered_at += frames * (1.0 / self.playback_fps)

        #Update camera matrices that will be used for rendering
        if isinstance(self.scene.camera, ViewerCamera):
            self.scene.camera.update_animation(frame_time)
        self.scene.camera.update_matrices(self.window.size[0], self.window.size[1])

        if not export:
            self.streamable_capture()

        self.render_fragmap()
        self.render_shadowmap()
        self.render_prepare()
        self.render_scene()
        self.render_outline([n for n in self.scene.collect_nodes() if n.draw_outline], (0.3, 0.7, 1, 1))

        if not export:
            # If the selected object is a Node render its outline.
            if isinstance(self.scene.selected_object, Node):
                self.render_outline([self.scene.selected_object], (1.0, 0.86, 0.35, 1.0))

            # If visualize is True draw a texture with the object id to the screen for debugging.
            if self.visualize:
                self.ctx.enable_only(moderngl.NOTHING)
                self.offscreen_p_tri_id.use(location=0)
                self.vis_prog['hash_color'] = True
                self.vis_quad.render(self.vis_prog)

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

    def render_shadowmap(self):
        """A pass to render the shadow map, i.e. render the entire scene once from the view of the light."""
        self.ctx.enable_only(moderngl.DEPTH_TEST)
        if self.shadows_enabled:
            rs = self.scene.collect_nodes()

            for light in self.scene.lights:
                if light.shadow_enabled:
                    light.use(self.ctx)
                    light_matrix = light.mvp()
                    for r in rs:
                        r.render_shadowmap(light_matrix, self.depth_only_prog)

    def render_fragmap(self):
        """A pass to render the fragment picking map, i.e. render the scene with world coords as colors."""
        self.ctx.enable_only(moderngl.DEPTH_TEST)
        self.offscreen_p.clear()
        self.offscreen_p.use()
        rs = self.scene.collect_nodes()
        for r in rs:
            r.render_fragmap(self.ctx, self.scene.camera, self.frag_map_prog)

    def render_outline(self, nodes, color):
        # Prepare the outline buffer, all objects rendered to this buffer will be outlined.
        self.outline_framebuffer.clear()
        self.outline_framebuffer.use()
        # Render outline of the nodes with outlining enabled, this potentially also renders their children.
        for n in nodes:
            n.render_outline(self.ctx, self.scene.camera, self.outline_prepare_prog)

        # Render the outline effect to the window.
        self.wnd.use()
        self.wnd.fbo.depth_mask = False
        self.ctx.enable_only(moderngl.NOTHING)
        self.outline_texture.use(0)
        self.outline_draw_prog['outline'] = 0
        self.outline_draw_prog['outline_color'] = color
        self.outline_quad.render(self.outline_draw_prog)
        self.wnd.fbo.depth_mask = True

    def render_scene(self):
        """Render the current scene to the framebuffer without time accounting and GUI elements."""
        self.scene.render(window_size=self.window.size,
                          lights=self.scene.lights,
                          shadows_enabled=self.shadows_enabled,
                          show_camera_target=self.show_camera_target and not self._using_temp_camera,
                          depth_prepass_prog=self.depth_only_prog)

    def render_prepare(self):
        """Prepare the framebuffer."""
        self.wnd.use()
        # Clear background and make sure only the flags we want are enabled.
        if self.dark_mode:
            self.ctx.clear(0.1, 0.1, 0.1, 1.0)
        else:
            self.ctx.clear(*self.scene.background_color)

        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.CULL_FACE)
        self.ctx.cull_face = 'back'
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    def prevent_background_interactions(self):
        """Prevent background interactions when hovering over any imgui window."""
        self.imgui_user_interacting = self.imgui.io.want_capture_mouse

    def toggle_animation(self, run: bool):
        self.run_animations = run
        if self.run_animations:
            self._last_frame_rendered_at = self.timer.time

    def reset_camera(self):
        if self._using_temp_camera:
            self._using_temp_camera = False

            fwd = self.scene.camera.forward
            pos = self.scene.camera.position

            self.scene.camera = ViewerCamera(45)
            self.scene.camera.position = np.copy(pos)
            self.scene.camera.target = pos + fwd * 3
            self.scene.camera.update_matrices(self.window_size[0], self.window_size[1])

    def set_temp_camera(self, camera):
        self.scene.camera = camera
        self._using_temp_camera = True

    def gui(self):
        imgui.new_frame()

        # Create a context menu when right clicking on the background.
        if (not any([imgui.is_window_hovered(), imgui.is_any_item_hovered()])
            and imgui.is_mouse_released(button=1)
            and not self._mouse_moved):
            # Select the object under the cursor
            if self.select_object(*imgui.get_io().mouse_pos) and hasattr(self.scene.selected_object, 'gui_context_menu'):
                imgui.open_popup("Context Menu")

        # Draw the context menu for the selected object
        if imgui.begin_popup("Context Menu"):
            if self.scene.selected_object is None or not hasattr(self.scene.selected_object, 'gui_context_menu'):
                imgui.close_current_popup()
            else:
                self.scene.selected_object.gui_context_menu(imgui)
            imgui.end_popup()

        # Reset user interacting state
        self.imgui_user_interacting = False

        # Render user controls
        for gc in self.gui_controls.values(): gc()

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
            self.scene.gui_editor(imgui)
        imgui.end()

    def gui_menu(self):
        clicked_export = False

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", 'Cmd+Q', False, True)
                if clicked_quit:
                    exit(1)

                clicked_export, selected_export = imgui.menu_item(
                    "Save as video..", None, False, True)

                clicked_screenshot, selected_screenshot = imgui.menu_item(
                    "Screenshot", self._shortcut_names[self._screenshot_key], False, True)
                if clicked_screenshot:
                    self.take_screenshot()

                imgui.end_menu()

            if imgui.begin_menu("View", True):
                _, self.shadows_enabled = imgui.menu_item("Render Shadows", self._shortcut_names[self._shadow_key],
                                                          self.shadows_enabled, True)
                _, self.dark_mode = imgui.menu_item("Dark Mode", self._shortcut_names[self._dark_mode_key],
                                                    self.dark_mode, True)

                _, self.lock_selection = imgui.menu_item("Lock selection", self._shortcut_names[self._lock_selection_key],
                                                         self.lock_selection, True)

                imgui.end_menu()

            if imgui.begin_menu("Camera", True):
                _, self.show_camera_target = imgui.menu_item("Show Camera Target", self._shortcut_names[self._show_camera_target_key],
                                                    self.show_camera_target, True)

                clicked, _ = imgui.menu_item("Center view on selection", self._shortcut_names[self._center_view_on_selection_key],
                                             False, isinstance(self.scene.selected_object, Node))
                if clicked:
                    self.center_view_on_selection()

                is_ortho = False if self._using_temp_camera else self.scene.camera.is_ortho
                _, is_ortho = imgui.menu_item("Orthographic Camera",
                                                                self._shortcut_names[self._orthographic_camera_key],
                                                is_ortho, True)
                if is_ortho and self._using_temp_camera:
                    self.reset_camera()
                self.scene.camera.is_ortho = is_ortho

                clicked_save_cam, selected_save_cam = imgui.menu_item("Save Camera",
                                                                self._shortcut_names[self._save_cam_key],
                                                                False, True)
                if clicked_save_cam:
                    self.reset_camera()
                    self.scene.camera.save_cam()

                clicked_load_cam, selected_load_cam = imgui.menu_item("Load Camera",
                                                                self._shortcut_names[self._load_cam_key],
                                                                False, True)
                if clicked_load_cam:
                    self.reset_camera()
                    self.scene.camera.load_cam()

                imgui.end_menu()

            if imgui.begin_menu("Mode", True):
                for id, mode in self.modes.items():
                    mode_clicked, _ = imgui.menu_item(mode['title'], mode['shortcut'], id == self.selected_mode, True)
                    if mode_clicked:
                        self.selected_mode = id

                imgui.end_menu()

            if imgui.begin_menu("Help", True):
                clicked, self._show_shortcuts_window = imgui.menu_item("Keyboard shortcuts", None, self._show_shortcuts_window)
                imgui.end_menu()

            if imgui.begin_menu("Debug", True):
                _, self.visualize = imgui.menu_item("Visualize debug texture", self._shortcut_names[self._visualize_key],
                                                    self.visualize, True)

                imgui.end_menu()

            imgui.end_main_menu_bar()

        if clicked_export:
            imgui.open_popup("Export Video")
            self.export_fps = self.playback_fps
            self.toggle_animation(False)

        imgui.set_next_window_size(570,0)
        if imgui.begin_popup_modal("Export Video", flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)[0]:
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
                _, animation_range = imgui.drag_int2('Animation range', *self.export_animation_range, min_value=0, max_value=self.scene.n_frames - 1)
                if animation_range[0] != self.export_animation_range[0]:
                    self.export_animation_range[0] = animation_range[0]
                    self.export_animation_range[1] = max(animation_range[0], animation_range[1])
                elif animation_range[-1] != self.export_animation_range[1]:
                    self.export_animation_range[1] = animation_range[1]
                    self.export_animation_range[0] = min(animation_range[0], animation_range[1])

                _, self.playback_fps = imgui.drag_float('Playback fps', self.playback_fps, 0.1,
                                                         min_value=1.0, max_value=120.0, format='%.1f')
                imgui.same_line(spacing=10)
                speedup = self.playback_fps / self.scene.fps
                imgui.text(f'({speedup:.2f}x speed)')
            else:
                _, self.scene.current_frame_id = imgui.slider_int('Frame', self.scene.current_frame_id,
                                                                   min_value=0, max_value=self.scene.n_frames - 1)
                _, self.export_duration = imgui.drag_float('Duration (s)', self.export_duration, min_value=0.1, max_value=10000.0, change_speed=0.05, format='%.1f')
                duration = self.export_duration

            if self.export_animation:
                if isinstance(self.scene.camera, ViewerCamera):
                    _, self.export_rotate_camera = imgui.checkbox("Rotate camera", self.export_rotate_camera)
                else:
                    imgui.push_style_var(imgui.STYLE_ALPHA, 0.2)
                    imgui.checkbox("Rotate camera (only available for ViewerCamera)", False)
                    imgui.pop_style_var(1)

            if not self.export_animation or self.export_rotate_camera:
                _, self.export_seconds_per_rotation = imgui.drag_float('Rotation time (s)', self.export_seconds_per_rotation, min_value=0.1, max_value=10000.0, change_speed=0.01, format='%.2f')
                imgui.same_line()
                if imgui.button("Once"):
                    if self.export_animation:
                        self.export_seconds_per_rotation = (self.export_animation_range[1] - self.export_animation_range[0] + 1) / self.playback_fps
                    else:
                        self.export_seconds_per_rotation = self.export_duration

            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            # Output settings.
            imgui.text("Output")
            imgui.spacing()
            imgui.text("Format:")
            imgui.same_line()
            if imgui.radio_button("MP4", not self.export_as_gif):
                self.export_as_gif = False
            imgui.same_line(spacing=15)
            if imgui.radio_button("GIF", self.export_as_gif):
                self.export_as_gif = True

            if self.export_as_gif:
                max_output_fps = 30.0
            else:
                max_output_fps = 120.0
            self.export_fps = min(self.export_fps, max_output_fps)
            _, self.export_fps = imgui.drag_float('fps', self.export_fps, 0.1,
                                                  min_value=1.0, max_value=max_output_fps, format='%.1f')
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
            imgui.text(f"Resolution: [{int(self.window_size[0] * self.export_scale_factor)}x{int(self.window_size[1] * self.export_scale_factor)}]")
            _, self.export_scale_factor = imgui.drag_float('Scale', self.export_scale_factor,
                                                               min_value=0.01, max_value=1.0,
                                                               change_speed=0.005, format='%.2f')

            imgui.same_line(position=440)
            if imgui.button("1x##scale", width=35):
                self.export_scale_factor = 1.0
            imgui.same_line()
            if imgui.button("1/2x##scale", width=35):
                self.export_scale_factor = 0.5
            imgui.same_line()
            if imgui.button("1/4x##scale", width=35):
                self.export_scale_factor = 0.25

            if self.export_animation:
                duration =  (animation_range[1] - animation_range[0] + 1) / self.playback_fps
                # Compute exact number of frames if playback fps is an exact multiple of export fps.
                if np.fmod(self.playback_fps, self.export_fps) < 0.1:
                    playback_count = int(np.round(self.playback_fps / self.export_fps))
                    frames = (animation_range[1] - animation_range[0] + 1) // playback_count
                else:
                    frames = int(np.ceil(duration * self.export_fps))
            else:
                frames = int(np.ceil(duration * self.export_fps))

            imgui.spacing()
            imgui.text(f"Duration: {duration:.2f}s ({frames} frames @ {self.export_fps:.2f}fps)")
            imgui.spacing()

            # Draw a cancel and exit button on the same line using the available space
            button_width = (imgui.get_content_region_available()[0] - imgui.get_style().item_spacing[0]) * 0.5

            # Style the cancel with a grey color
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.5, 0.5, 0.5, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE,  0.6, 0.6, 0.6, 1.0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.7, 0.7, 0.7, 1.0)

            if imgui.button("Cancel", width=button_width):
                imgui.close_current_popup()

            imgui.pop_style_color()
            imgui.pop_style_color()
            imgui.pop_style_color()

            imgui.same_line()
            if imgui.button("Export", button_width):
                imgui.close_current_popup()
                self.export_video(
                    os.path.join(C.export_dir, 'videos', self.window.title + ('.gif' if self.export_as_gif else '.mp4')),
                    animation=self.export_animation,
                    animation_range=self.export_animation_range,
                    duration=self.export_duration,
                    frame=self.scene.current_frame_id,
                    output_fps=self.export_fps,
                    rotate_camera=not self.export_animation or self.export_rotate_camera,
                    seconds_per_rotation=self.export_seconds_per_rotation,
                    scale_factor=self.export_scale_factor
                )

            imgui.end_popup()

    def gui_playback(self):
        """GUI to control playback settings."""
        imgui.set_next_window_position(50, 100 + self.window_size[1] * 0.7, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(self.window_size[0] * 0.4, self.window_size[1] * 0.15, imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("Playback", None)
        if expanded:
            u, run_animations = imgui.checkbox("Run animations [{}]".format(self._shortcut_names[self._pause_key]),
                                                    self.run_animations)
            if u:
                self.toggle_animation(run_animations)

            # Plot FPS
            frametime_avg = np.mean(self._past_frametimes[self._past_frametimes > 0.0])
            fps_avg = 1 / frametime_avg
            ms_avg = frametime_avg * 1000.0
            ms_last = self._past_frametimes[-1] * 1000.0

            imgui.plot_lines("Internal {:.1f} fps @ {:.2f} ms/frame [{:.2f}ms]".format(fps_avg, ms_avg, ms_last),
                            array('f', (1.0 / self._past_frametimes).tolist()),
                            scale_min=0, scale_max=100.0, graph_size=(100, 20))

            _, self.playback_fps = imgui.drag_float(f'Playback fps', self.playback_fps, 0.1,
                                                    min_value=1.0, max_value=120.0, format='%.1f')
            imgui.same_line(spacing=10)
            speedup = self.playback_fps / self.scene.fps
            imgui.text(f'({speedup:.2f}x speed)')

            # Sequence Control
            # For simplicity, we allow the global sequence slider to only go as far as the shortest known sequence.
            n_frames = self.scene.n_frames

            _, self.scene.current_frame_id = imgui.slider_int('Frame##r_global_seq_control', self.scene.current_frame_id,
                                                            min_value=0, max_value=n_frames - 1)
            self.prevent_background_interactions()
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
        if self.selected_mode == 'inspect':
            imgui.set_next_window_position(self.window_size[0] * 0.6, 50, imgui.FIRST_USE_EVER)
            imgui.set_next_window_size(self.window_size[0] * 0.35, 140, imgui.FIRST_USE_EVER)
            expanded, _ = imgui.begin("Inspect", None)
            if expanded:
                if self.mmi is not None:
                    for k, v in zip(self.mmi._fields, self.mmi):
                        imgui.text("{}: {}".format(k, v))

                self.prevent_background_interactions()
            imgui.end()

    def gui_exit(self):
        if self._exit_popup_open:
            imgui.open_popup("Exit##exit-popup")

        if imgui.begin_popup_modal("Exit##exit-popup", flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)[0]:
            if self._exit_popup_open:
                imgui.text("Are you sure you want to exit?")
                imgui.spacing()

                # Draw a cancel and exit button on the same line using the available space
                button_width = (imgui.get_content_region_available()[0] - imgui.get_style().item_spacing[0]) * 0.5

                # Style the cancel with a grey color
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.5, 0.5, 0.5, 1.0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE,  0.6, 0.6, 0.6, 1.0)
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

    def mesh_mouse_intersection(self, x: int, y: int):
        """Given an x/y screen coordinate, get the intersected object, triangle id, and xyz point in camera space"""

        # Texture is y=0 at bottom, so we flip y coords
        pos = int(x * self.wnd.pixel_ratio), int(self.wnd.buffer_height - (y * self.wnd.pixel_ratio))

        # Fragment picker uses already encoded position/object/triangle in the frag_pos program textures
        self.frag_pick_prog['texel_pos'].value = pos
        self.offscreen_p_viewpos.use(location=0)
        self.offscreen_p_tri_id.use(location=1)
        self.picker_vao.transform(self.frag_pick_prog, self.picker_output, vertices=1)
        x, y, z, obj_id, tri_id = struct.unpack('3f2i', self.picker_output.read())

        if obj_id >= 0 and tri_id >= 0:
            node = self.scene.get_node_by_uid(obj_id)
            # Camera space to world space
            point_world = np.array(np.linalg.inv(self.scene.camera.get_view_matrix()) @ np.array((x, y, z, 1.0)))[:-1]
            point_local = (np.linalg.inv(node.model_matrix) @ np.append(point_world, 1.0))[:-1]
            if isinstance(node, Meshes) or isinstance(node, Billboard) or isinstance(node, VariableTopologyMeshes):
                vert_id = node.closest_vertex_in_triangle(tri_id, point_local)
                bc_coords = node.get_bc_coords_from_points(tri_id, [point_local])
            elif isinstance(node, PointClouds):
                vert_id = tri_id
                bc_coords = np.array([1, 0, 0])
            else:
                vert_id = 0
                bc_coords = np.array(0, 0, 0)

            return MeshMouseIntersection(node, tri_id, vert_id, point_world, point_local, bc_coords)

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
                        self.scene.select(node, mmi.node, mmi.tri_id)
                        return True
                    else:
                        return False
                node = node.parent

        return False

    def center_view_on_selection(self):
        if isinstance(self.scene.selected_object, Node):
            if self._using_temp_camera:
                self.reset_camera()
            forward = self.scene.camera.forward
            bounds = self.scene.selected_object.current_bounds
            diag = np.linalg.norm(bounds[:, 0] - bounds[:, 1])
            dist = max(0.01, diag * 1.3)

            center = bounds.mean(-1)
            self.scene.camera.move_with_animation(center - forward * dist, center)

    def resize(self, width: int, height: int):
        self.window_size = (width, height)
        self.imgui.resize(width, height)
        self.create_framebuffers()

    def files_dropped_event(self, x: int, y: int, paths):
        for path in paths:
            base, ext = os.path.splitext(path)
            if ext == ".obj" or ext == ".ply":
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

        if self.imgui.io.want_capture_keyboard:
            return

        if action == self.wnd.keys.ACTION_PRESS:
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
                self.show_camera_target = not self.show_camera_target

            elif key == self._orthographic_camera_key:
                if self._using_temp_camera:
                    self.reset_camera()
                self.scene.camera.is_ortho = not self.scene.camera.is_ortho

            elif key == self._center_view_on_selection_key:
                self.center_view_on_selection()

            elif key == self._mode_view_key:
                self.selected_mode = 'view'

            elif key == self._mode_inspect_key:
                self.selected_mode = 'inspect'

            elif key == self._dark_mode_key:
                self.dark_mode = not self.dark_mode
                self.scene.set_lights(self.dark_mode)

            elif key == self._screenshot_key:
                self.take_screenshot()
            elif key == self._save_cam_key:
                if self._using_temp_camera:
                    self.reset_camera()
                self.scene.camera.save_cam()
            elif key == self._load_cam_key:
                if self._using_temp_camera:
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
        self.imgui.mouse_position_event(x, y, dx, dy)

        if self.selected_mode == 'inspect':
            self.mmi = self.mesh_mouse_intersection(x, y)

    def mouse_press_event(self, x: int, y: int, button: int):
        self.imgui.mouse_press_event(x, y, button)

        if not self.imgui_user_interacting:
            # Pan or rotate camera on middle click.
            if button == self._middle_mouse_button:
                self._pan_camera = self.wnd.modifiers.shift
                self._rotate_camera = not self.wnd.modifiers.shift

            # Rotate camera on left click.
            if button == self._left_mouse_button:
                self._rotate_camera = True
                self._pan_camera = False

            if button == self._right_mouse_button:
                self._pan_camera = True
                self._rotate_camera = False

            self._mouse_down_position = np.array([x, y])
            self._mouse_moved = False

    def mouse_release_event(self, x: int, y: int, button: int):
        self.imgui.mouse_release_event(x, y, button)

        self._pan_camera = False
        self._rotate_camera = False

        if not self.imgui_user_interacting:
            # Select the mesh under the cursor on left click.
            if button == self._left_mouse_button and not self._mouse_moved:
                if not self.select_object(x, y):
                    # If selection is enabled and nothing was selected clear the previous selection
                    if not self.lock_selection:
                        self.scene.select(None)

    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        self.imgui.mouse_drag_event(x, y, dx, dy)

        if not self.imgui_user_interacting :
            if self._pan_camera:
                if self._using_temp_camera:
                    self.reset_camera()
                self.scene.camera.pan(dx, dy)

            if self._rotate_camera:
                if self._using_temp_camera:
                    self.reset_camera()
                self.scene.camera.rotate_azimuth_elevation(dx, dy)

            if not self._mouse_moved and np.linalg.norm(np.array([x, y]) - self._mouse_down_position) > self._move_threshold:
                self._mouse_moved = True

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        self.imgui.mouse_scroll_event(x_offset, y_offset)

        if not self.imgui_user_interacting:
            if self._using_temp_camera:
                self.reset_camera()
            self.scene.camera.dolly_zoom(np.sign(y_offset), self.wnd.modifiers.shift)

    def unicode_char_entered(self, char):
        self.imgui.unicode_char_entered(char)

    def save_current_frame_as_image(self, path, scale_factor=None):
        """Saves the current frame as an image to disk."""
        image = self.get_current_frame_as_image()
        if scale_factor is not None and scale_factor != 1.0:
            w = int(image.width * scale_factor)
            h = int(image.height * scale_factor)
            image = image.resize((w, h), Image.LANCZOS)
        image.save(path)

    def get_current_frame_as_image(self):
        """Return the FBO content as a PIL image."""
        image = Image.frombytes('RGB',
                                (self.wnd.fbo.viewport[2] - self.wnd.fbo.viewport[0],
                                 self.wnd.fbo.viewport[3] - self.wnd.fbo.viewport[1]),
                                self.wnd.fbo.read(viewport=self.wnd.fbo.viewport, alignment=1))
        return image.transpose(Image.FLIP_TOP_BOTTOM)

    def get_current_depth_image(self):
        """
        Return the depth buffer as a 'F' PIL image.
        Depth is stored as the z coordinate in eye (view) space.
        Therefore values in the depth image represent the distance from the pixel to
        the plane passing through the camera and orthogonal to the view direction.
        Values are between the near and far plane distances of the camera used for rendering,
        everything outside this range is clipped by OpenGL.
        """
        # Get depth image from depth buffer.
        depth = Image.frombytes('F',
                                (self.wnd.fbo.viewport[2] - self.wnd.fbo.viewport[0],
                                 self.wnd.fbo.viewport[3] - self.wnd.fbo.viewport[1]),
                                self.wnd.fbo.read(viewport=self.wnd.fbo.viewport, alignment=1, attachment=-1, dtype='f4'))

        # Convert from [0, 1] range to [-1, 1] range.
        # This is necessary because our projection matrix computes NDC
        # from [-1, 1], but depth is then stored normalized from 0 to 1.
        depth = np.array(depth) * 2.0 - 1.0

        # Extract projection matrix parameters used for mapping Z coordinates.
        P = self.scene.camera.get_projection_matrix()
        a, b = P[2, 2], P[2, 3]

        # Linearize depth values. This converts from [-1, 1] range to the
        # view space Z coordinate value, with positive z in front of the camera.
        z = b / (a + depth)
        return Image.fromarray(z, mode='F').transpose(Image.FLIP_TOP_BOTTOM)

    def get_current_mask_image(self):
        """
        Render and return a color mask as a 'RGB' PIL image. Each object in the mask
        has a uniform color computed as an hash of the Node uid.
        """
        # Get object ids as floating point numbers from the first channel of
        # the first attachment of the picking framebuffer.
        id = Image.frombytes('F',
                             (self.wnd.fbo.viewport[2] - self.wnd.fbo.viewport[0],
                              self.wnd.fbo.viewport[3] - self.wnd.fbo.viewport[1]),
                             self.offscreen_p.read(viewport=self.wnd.fbo.viewport, alignment=1, components=1, attachment=1, dtype='f4'))

        # Convert the id to integer values.
        id_int = np.asarray(id).astype(dtype=np.int32)

        # Hash the ids.
        def hash(h):
            h ^= h >> 16
            h *= 0x85ebca6b
            h ^= h >> 13
            h *= 0xc2b2ae35
            h ^= h >> 16
            return h
        hashed = hash(id_int.copy())

        # Convert the hashed ids to an RGBA image and then throw away the alpha channel.
        img = Image.frombytes('RGBA', id.size, hashed.tobytes()).convert('RGB')
        return img.transpose(Image.FLIP_TOP_BOTTOM)

    def on_close(self):
        """
        Clean up before destroying the window
        """
        # Shut down all streams
        for s in self.scene.collect_nodes(obj_type=Streamable):
            s.stop()

        # Clear the lru_cache on all shaders, we do this so that future instances of the viewer
        # have to recompile shaders with the current moderngl context.
        # See issue #12 https://github.com/eth-ait/aitviewer/issues/12
        clear_shader_cache()

    def take_screenshot(self):
        """Save the current frame to an image in the screenshots directory inside the export directory"""
        file_path = os.path.join(C.export_dir, 'screenshots', 'frame_{:0>6}.png'.format(self.scene.current_frame_id))
        self.export_frame(file_path)
        print(f"Screenshot saved to {file_path}")

    def export_frame(self, file_path, scale_factor:float=None):
        """Save the current frame to an image.
        :param file_path: the path where the image is saved.
        :param scale_factor: a scale factor used to scale the image. If None no scale factor is used and
          the image will have the same size as the viewer.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Store run_animation old value and set it to false.
        run_animations = self.run_animations
        self.run_animations = False

        # Render and save frame.
        self.render(0, 0, export=True)
        self.save_current_frame_as_image(file_path, scale_factor)

        # Restore run animation and update last frame rendered time.
        self.run_animations = run_animations
        self._last_frame_rendered_at = self.timer.time

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
        seconds_per_rotation=10.0,
        scale_factor=None,
        ):
        if rotate_camera and not isinstance(self.scene.camera, ViewerCamera):
            print("Cannot export a video with camera rotation while using a camera that is not a ViewerCamera")
            return

        if frame_dir is None and output_path is None:
            print("You must either specify a path where to render the images to or where to save the video to")
            return

        if frame_dir is not None:
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
        az_delta = 2 * np.pi / seconds_per_rotation * (duration / frames)

        # Initialize video writer.
        if output_path is not None:
            path_mp4, path_gif, is_gif = get_video_paths(output_path)
            writer = skvideo.io.FFmpegWriter(
                path_mp4,
                inputdict = {
                    '-framerate': str(output_fps),  # must be this early in the command, otherwise it is not applied.
                },
                outputdict = {
                    '-c:v': 'libx264',
                    '-preset': 'slow',
                    '-profile:v': 'high',
                    '-level:v': '4.0',
                    '-pix_fmt': 'yuv420p',
                    '-vf': 'pad=ceil(iw/2)*2:ceil(ih/2)*2',  # Avoid error when image res is not divisible by 2.
                    '-r': str(output_fps),
                })

        for i in tqdm(range(frames), desc='Rendering frames'):
            if rotate_camera:
                self.scene.camera.rotate_azimuth(az_delta)

            self.render(time, time + dt, export=True)
            img = self.get_current_frame_as_image()

            # Scale image by the scale factor.
            if scale_factor is not None and scale_factor != 1.0:
                w = int(img.width * scale_factor)
                h = int(img.height * scale_factor)
                img = img.resize((w, h), Image.LANCZOS)

            # Store the image to disk if a directory for frames was given.
            if frame_dir is not None:
                img_name = os.path.join(frame_dir, 'frame_{:0>6}.png'.format(i))
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
                video_to_gif(path_mp4, path_gif, remove=True)

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


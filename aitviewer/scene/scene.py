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
import moderngl
import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.coordinate_system import CoordinateSystem
from aitviewer.renderables.lines import Lines
from aitviewer.renderables.plane import ChessboardPlane
from aitviewer.scene.camera import ViewerCamera
from aitviewer.scene.light import Light
from aitviewer.scene.node import Node
from aitviewer.utils.utils import (
    compute_union_of_bounds,
    compute_union_of_current_bounds,
)


class Scene(Node):
    """Generic scene node"""

    def __init__(self, **kwargs):
        """Create a scene with a name."""
        kwargs["gui_material"] = False
        super(Scene, self).__init__(**kwargs)

        # References resources in the scene
        self.lights = []
        self.camera = None

        # Scene has a reference to ctx
        self.ctx = None

        self.backface_culling = True
        self.fps = C.scene_fps
        self.background_color = C.background_color

        # Default Setup
        # If you update the number of lights, make sure to change the respective `define` statement in
        # directional_lights.glsl as well!
        # Influence of diffuse lighting is controlled globally for now, but should eventually be a material property.
        self.lights.append(
            Light.facing_origin(
                light_color=(1.0, 1.0, 1.0),
                name="Back Light",
                position=(0.0, 10.0, -15.0),
                shadow_enabled=False,
            )
        )
        self.lights.append(
            Light.facing_origin(
                light_color=(1.0, 1.0, 1.0),
                name="Front Light",
                position=(0.0, 10.0, 15.0),
            )
        )
        self.add(*self.lights)

        self.ambient_strength = 2.0

        # Scene items
        self.origin = CoordinateSystem(name="Origin", length=0.1, gui_affine=False, gui_material=False)
        self.add(self.origin)

        self.floor = ChessboardPlane(100.0, 200, (0.9, 0.9, 0.9, 1.0), (0.82, 0.82, 0.82, 1.0), name="Floor")
        self.floor.material.diffuse = 0.1
        self.add(self.floor)

        # Camera cursor rendered at the camera target when moving the camera
        self.camera_target = Lines(
            np.array(
                [
                    [-1, 0, 0],
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0],
                    [0, 0, -1],
                    [0, 0, 1],
                ]
            )
            * 0.05,
            r_base=0.002,
            color=(0.2, 0.2, 0.2, 1),
            mode="lines",
            cast_shadow=False,
        )
        self.add(self.camera_target, show_in_hierarchy=False, enabled=False)

        self.custom_font = None
        self.properties_icon = "\u0094"

        # Currently selected object, None if no object is selected
        self.selected_object = None
        # Object shown in the property panel
        self.gui_selected_object = None

        # The scene node in the GUI is expanded at the start.
        self.expanded = True

    def render(self, **kwargs):
        # As per https://learnopengl.com/Advanced-OpenGL/Blending

        # Setup the camera target cursor for rendering.
        if self.camera_target.enabled and isinstance(self.camera, ViewerCamera):
            self.camera_target.position = self.camera.target

        # Collect all renderable nodes
        rs = self.collect_nodes()

        transparent = []

        # Draw all opaque objects first
        for r in rs:
            if not r.is_transparent():
                # Turn off backface culling if enabled for the scene
                # and requested by the current object
                if self.backface_culling and r.backface_culling:
                    self.ctx.enable(moderngl.CULL_FACE)
                else:
                    self.ctx.disable(moderngl.CULL_FACE)

                self.safe_render(r, **kwargs)
            else:
                # Otherwise append to transparent list
                transparent.append(r)

        fbo = kwargs["fbo"]

        # Draw back to front by sorting transparent objects based on distance to camera.
        # As an approximation we only sort using the origin of the object
        # which may be incorrect for large objects or objects significantly
        # offset from their origin, but works in many cases.
        for r in sorted(
            transparent,
            key=lambda x: np.linalg.norm(x.position - self.camera.position),
            reverse=True,
        ):
            # Render to depth buffer only
            self.ctx.depth_func = "<"

            fbo.color_mask = (False, False, False, False)
            self.safe_render_depth_prepass(r, **kwargs)
            fbo.color_mask = (True, True, True, True)

            # Turn off backface culling if enabled for the scene
            # and requested by the current object
            if self.backface_culling and r.backface_culling:
                self.ctx.enable(moderngl.CULL_FACE)
            else:
                self.ctx.disable(moderngl.CULL_FACE)

            # Render normally with less equal depth comparison function,
            # drawing only the pixels closer to the camera to avoid
            # order dependent blending artifacts
            self.ctx.depth_func = "<="
            self.safe_render(r, **kwargs)

        # Restore the default depth comparison function
        self.ctx.depth_func = "<"

    def safe_render_depth_prepass(self, r, **kwargs):
        if not r.is_renderable:
            r.make_renderable(self.ctx)
        r.render_depth_prepass(self.camera, **kwargs)

    def safe_render(self, r, **kwargs):
        if not r.is_renderable:
            r.make_renderable(self.ctx)
        r.render(self.camera, **kwargs)

    def make_renderable(self, ctx):
        self.ctx = ctx
        rs = self.collect_nodes(req_enabled=False)
        for r in rs:
            r.make_renderable(self.ctx)

    @property
    def bounds(self):
        return compute_union_of_bounds([n for n in self.nodes if n not in self.lights])

    @property
    def current_bounds(self):
        return compute_union_of_current_bounds([n for n in self.nodes if n not in self.lights])

    def auto_set_floor(self):
        """Finds the minimum lower bound in the y coordinate from all the children bounds and uses that as the floor"""
        if self.floor is not None and len(self.nodes) > 0:
            self.floor.position[1] = self.current_bounds[1, 0]
            self.floor.update_transform()

    def auto_set_camera_target(self):
        """Sets the camera target to the average of the center of all objects in the scene"""
        centers = []
        for n in self.nodes:
            if n not in self.lights:
                centers.append(n.current_center)

        if isinstance(self.camera, ViewerCamera) and len(centers) > 0:
            self.camera.target = np.array(centers).mean(0)

    @property
    def light_mode(self):
        return self._light_mode

    @light_mode.setter
    def light_mode(self, mode):
        if mode == "default":
            self._light_mode = mode
            self.ambient_strength = 2.0
            for l in self.lights:
                l.strength = 1.0
        elif mode == "dark":
            self._light_mode = mode
            self.ambient_strength = 0.4
            for l in self.lights:
                l.strength = 1.0
        elif mode == "diffuse":
            self._light_mode = mode
            self.ambient_strength = 1.0
            for l in self.lights:
                l.strength = 2.0
        else:
            raise ValueError(f"Invalid light mode: {mode}")

    def collect_nodes(self, req_enabled=True, obj_type=Node):
        nodes = []

        # Use head recursion in order to collect the most deep nodes first
        # These are likely to be nodes which should be rendered first (i.e. transparent parent nodes should be last)
        def rec_collect_nodes(nn):
            if not req_enabled or nn.enabled:
                if isinstance(nn, obj_type):
                    nodes.append(nn)
                for n_child in nn.nodes:
                    rec_collect_nodes(n_child)

        for n in self.nodes:
            rec_collect_nodes(n)
        return nodes

    def get_node_by_name(self, name):
        assert name != ""
        ns = self.collect_nodes()
        for n in ns:
            if n.name == name:
                return n
        return None

    def get_node_by_uid(self, uid):
        ns = self.collect_nodes()
        for n in ns:
            if n.uid == uid:
                return n
        return None

    def select(self, obj, selected_node=None, selected_instance=None, selected_tri_id=None):
        """Set 'obj' as the selected object"""
        self.selected_object = obj
        if isinstance(obj, Node):
            self.selected_object.on_selection(selected_node, selected_instance, selected_tri_id)
        # Always keep the last selected object in the property panel
        if obj is not None:
            self.gui_selected_object = obj

    def is_selected(self, obj):
        """Returns true if obj is currently selected"""
        return obj == self.selected_object

    def gui_selected(self, imgui):
        """GUI to edit the selected node"""
        if self.gui_selected_object:
            s = self.gui_selected_object

            # Custom GUI Elements
            imgui.indent(22)
            imgui.push_font(self.custom_font)
            imgui.text(f"{s.icon} {s.name}")
            imgui.pop_font()

            # Modes
            if hasattr(s, "gui_modes") and len(s.gui_modes) > 1:
                imgui.push_font(self.custom_font)
                imgui.spacing()
                for i, (gm_key, gm_val) in enumerate(s.gui_modes.items()):
                    if s.selected_mode == gm_key:
                        imgui.push_style_color(imgui.COLOR_BUTTON, 0.26, 0.59, 0.98, 1.0)
                    mode_clicked = imgui.button(f" {gm_val['icon']}{gm_val['title']} ")
                    if s.selected_mode == gm_key:
                        imgui.pop_style_color()
                    if mode_clicked:
                        s.selected_mode = gm_key
                    if i != len(s.gui_modes) - 1:
                        imgui.same_line()
                imgui.pop_font()

                # Mode specific GUI
                imgui.spacing()
                if "fn" in s.gui_modes[s.selected_mode]:
                    s.gui_modes[s.selected_mode]["fn"](imgui)

            # Custom GUI (i.e. Camera specific params)
            s.gui(imgui)
            imgui.unindent()

            # General GUI elements
            if hasattr(s, "gui_controls"):
                imgui.spacing()
                imgui.spacing()
                imgui.spacing()
                for i, (gc_key, gc_val) in enumerate(s.gui_controls.items()):
                    if not gc_val["is_visible"]:
                        continue
                    imgui.begin_group()
                    imgui.push_font(self.custom_font)
                    imgui.text(f"{gc_val['icon']}")
                    imgui.pop_font()
                    imgui.end_group()
                    imgui.same_line(spacing=8)
                    imgui.begin_group()
                    gc_val["fn"](imgui)
                    imgui.end_group()
                    imgui.spacing()
                    imgui.spacing()
                    imgui.spacing()

    def gui(self, imgui):
        imgui.text(f"FPS: {self.fps:.1f}")
        # Background color
        uc, color = imgui.color_edit4("Background", *self.background_color, show_alpha=True)
        if uc:
            self.background_color = color

        _, self.ambient_strength = imgui.drag_float(
            "Ambient strength",
            self.ambient_strength,
            0.01,
            min_value=0.0,
            max_value=10.0,
            format="%.2f",
        )

    def gui_editor(self, imgui):
        """GUI to control scene settings."""
        # Also include the camera GUI in the scene node.
        self.gui_camera(imgui)
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        self.gui_hierarchy(imgui, [self])
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        imgui.spacing()
        self.gui_selected(imgui)

    def gui_camera(self, imgui):
        # Camera GUI
        imgui.push_font(self.custom_font)
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 2))

        flags = imgui.TREE_NODE_LEAF | imgui.TREE_NODE_FRAME_PADDING
        if self.is_selected(self.camera):
            flags |= imgui.TREE_NODE_SELECTED

        if isinstance(self.camera, ViewerCamera):
            name = self.camera.name
        else:
            name = f"Camera: {self.camera.name}"
        camera_expanded = imgui.tree_node(f"{self.camera.icon} {name}##tree_node_r_camera", flags)
        if imgui.is_item_clicked():
            self.select(self.camera)

        imgui.pop_style_var()
        imgui.pop_font()
        if camera_expanded:
            imgui.tree_pop()

    def gui_lights(self, imgui):
        # Lights GUI
        for _, light in enumerate(self.lights):
            imgui.push_font(self.custom_font)
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 2))

            flags = imgui.TREE_NODE_LEAF | imgui.TREE_NODE_FRAME_PADDING
            if self.is_selected(light):
                flags |= imgui.TREE_NODE_SELECTED
            light_expanded = imgui.tree_node(f"{light.icon} {light.name}##tree_node_r", flags)
            if imgui.is_item_clicked():
                self.select(light)

            imgui.pop_style_var()
            imgui.pop_font()
            if light_expanded:
                imgui.tree_pop()

    def gui_hierarchy(self, imgui, rs):
        # Nodes GUI
        for r in rs:
            # Skip nodes that shouldn't appear in the hierarchy.
            if not r.show_in_hierarchy:
                continue

            # Visibility
            curr_enabled = r.enabled
            if not curr_enabled:
                imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 1.0, 0.4)

            # Title
            imgui.push_font(self.custom_font)
            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 2))

            flags = imgui.TREE_NODE_OPEN_ON_ARROW | imgui.TREE_NODE_FRAME_PADDING
            if r.expanded:
                flags |= imgui.TREE_NODE_DEFAULT_OPEN
            if self.is_selected(r):
                flags |= imgui.TREE_NODE_SELECTED
            if not any(c.show_in_hierarchy for c in r.nodes):
                flags |= imgui.TREE_NODE_LEAF
            r.expanded = imgui.tree_node("{} {}##tree_node_{}".format(r.icon, r.name, r.unique_name), flags)
            if imgui.is_item_clicked():
                self.select(r)

            imgui.pop_style_var()
            imgui.pop_font()

            if r != self:
                # Aligns checkbox to the right side of the window
                # https://github.com/ocornut/imgui/issues/196
                imgui.same_line(position=imgui.get_window_content_region_max().x - 25)
                eu, enabled = imgui.checkbox("##enabled_r_{}".format(r.unique_name), r.enabled)
                if eu:
                    r.enabled = enabled

            if r.expanded:
                # Recursively render children nodes
                self.gui_hierarchy(imgui, r.nodes)
                imgui.tree_pop()

            if not curr_enabled:
                imgui.pop_style_color(1)

    def add_light(self, light):
        self.lights.append(light)

    @property
    def n_lights(self):
        return len(self.lights)

    @property
    def n_frames(self):
        n_frames = 1
        ns = self.collect_nodes(req_enabled=False)
        for n in ns:
            if n._enabled_frames is None:
                n_frames = max(n_frames, n.n_frames)
            else:
                n_frames = max(n_frames, n._enabled_frames.shape[0])
        return n_frames

    def render_outline(self, *args, **kwargs):
        # No outline when the scene node is selected
        return

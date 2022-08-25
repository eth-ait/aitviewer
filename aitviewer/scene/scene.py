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
import moderngl
import numpy as np

from aitviewer.renderables.coordinate_system import CoordinateSystem
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.plane import ChessboardPlane
from aitviewer.scene.light import Light
from aitviewer.scene.node import Node
from aitviewer.renderables.lines import Lines


class Scene(Node):
    """Generic scene node"""

    def __init__(self, **kwargs):
        """Create a scene with a name."""
        super(Scene, self).__init__(**kwargs)

        # References resources in the scene
        self.lights = []
        self.camera = None

        # Scene has a reference to ctx
        self.ctx = None

        self.backface_culling = True

        # Default Setup
        # If you update the number of lights, make sure to change the respective `define` statement in
        # directional_lights.glsl as well!
        # Influence of diffuse lighting is controlled globally for now, but should eventually be a material property.
        self.lights.append(Light(name='Back Light',  position=(0.0, 10.0, 15.0),  color=(1.0, 1.0, 1.0, 1.0)))
        self.lights.append(Light(name='Front Light', position=(0.0, 10.0, -15.0), color=(1.0, 1.0, 1.0, 1.0), shadow_enabled=False))
        self.add(*self.lights, show_in_hierarchy=False)

        # Scene items
        self.origin = CoordinateSystem(name="Origin", length=0.1)
        self.add(self.origin, has_gui=False)
        
        self.floor = ChessboardPlane(100.0, 200, (0.9, 0.9, 0.9, 1.0),  (0.82, 0.82, 0.82, 1.0), name="Floor")
        self.floor.material.diffuse = 0.1
        self.add(self.floor)

        # Camera cursor rendered at the camera target when moving the camera
        self.camera_target = Lines(np.array([
            [-1, 0, 0], [1, 0, 0],
            [0, -1, 0], [0, 1, 0],
            [0, 0, -1], [0, 0, 1],
        ]) * 0.05, r_base=0.002, color=(0.2, 0.2, 0.2, 1), mode='lines', cast_shadow=False)
        self.add(self.camera_target, show_in_hierarchy=False)

        self.custom_font = None

        # Currently selected object, None if no object is selected
        self.selected_object = None
        
    def render(self, **kwargs):
        # As per https://learnopengl.com/Advanced-OpenGL/Blending

        # Setup the camera target cursor for rendering
        self.camera_target.enabled = kwargs['show_camera_target']
        if self.camera_target.enabled:
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
        
        # Draw back to front by sorting transparent objects based on distance to camera.
        # As an approximation we only sort using the origin of the object 
        # which may be incorrect for large objects or objects significantly 
        # offset from their origin, but works in many cases.
        for r in sorted(transparent, key=lambda r: np.linalg.norm(r.position - self.camera.position), reverse=True):
            # Render to depth buffer only
            self.ctx.depth_func = '<'
            self.safe_render_depth_prepass(r, **kwargs)

            # Turn off backface culling if enabled for the scene
            # and requested by the current object
            if self.backface_culling and r.backface_culling:
                self.ctx.enable(moderngl.CULL_FACE)
            else:
                self.ctx.disable(moderngl.CULL_FACE)

            # Render normally with less equal depth comparison function,
            # drawing only the pixels closer to the camera to avoid
            # order dependent blending artifacts
            self.ctx.depth_func = '<='
            self.safe_render(r, **kwargs)

        # Restore the default depth comparison function
        self.ctx.depth_func = '<'
    
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

    def auto_set_floor(self):
        """Finds the minimum lower bound in the y coordinate from all the children bounds and uses that as the floor"""
        rs = self.collect_nodes()
        collected_bounds = []
        for r in rs:
            if r.bounds is not None:
                collected_bounds.append(r.bounds)

        if len(collected_bounds) > 0:
            self.floor.position[1] = np.array(collected_bounds)[:, 1, 0].min()

    def auto_set_camera_target(self):
        """Sets the camera target to the average of the center of all objects in the scene"""
        rs = self.collect_nodes()
        centers = []
        for r in rs:
            if r.bounds is not None:
                center = np.matmul(r.rotation, r.bounds.mean(-1)) + r.position
                if center.sum() != 0.0:
                    centers.append(center)

        if len(centers) > 0:
            self.camera.target = np.array(centers).mean(0)

    def set_lights(self, is_dark_mode=False):
        if is_dark_mode:
            for l in self.lights:
                l.intensity_ambient = 0.2
        else:
            for l in self.lights:
                l.intensity_ambient = 1.0

    def collect_nodes(self, req_enabled=True, obj_type=Node):
        nodes = []

        # Use head recursion in order to collect the most deep nodes first
        # These are likely to be nodes which should be rendered first (i.e. transparent parent nodes should be last)
        def rec_collect_nodes(n):
            if (not req_enabled or n.enabled):
                if isinstance(n, obj_type):
                    nodes.append(n)
                for n_child in n.nodes:
                    rec_collect_nodes(n_child)

        for n in self.nodes:
            rec_collect_nodes(n)
        return nodes

    def get_node_by_name(self, name):
        assert name != ''
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
    
    def select(self, obj):
        """Set 'obj' as the selected object"""
        self.selected_object = obj

    def is_selected(self, obj):
        """Returns true if obj is currently selected"""
        return obj == self.selected_object
    
    def gui_selected(self, imgui):
        """GUI to edit the selected node"""
        if self.selected_object:
            s = self.selected_object

            # Draw object icon and name
            imgui.push_font(self.custom_font)
            imgui.text(f"{s.icon} {s.name}")
            imgui.pop_font()

            # Draw gui
            s.gui(imgui)

    def gui(self, imgui):
        """GUI to control scene settings."""
        self.gui_camera(imgui)
        self.gui_lights(imgui)
        self.gui_renderables(imgui, self.nodes)
        imgui.spacing()
        imgui.separator()
        imgui.spacing()
        self.gui_selected(imgui)
        
    def gui_camera(self, imgui):
        # Camera GUI
        imgui.push_font(self.custom_font)
        imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 2))

        flags = imgui.TREE_NODE_LEAF | imgui.TREE_NODE_FRAME_PADDING 
        if self.is_selected(self.camera):
            flags |= imgui.TREE_NODE_SELECTED
        camera_expanded = imgui.tree_node(f"{self.camera.icon} {self.camera.name}##tree_node_r_camera", flags)
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

    def gui_renderables(self, imgui, rs):
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
            if not any(c.show_in_hierarchy for c in r.nodes) or not r.has_gui:
                flags |= imgui.TREE_NODE_LEAF
            r.expanded = imgui.tree_node("{} {}##tree_node_{}".format(r.icon, r.name, r.unique_name), flags)
            if imgui.is_item_clicked():
                self.select(r)

            imgui.pop_style_var()
            imgui.pop_font()

            # Aligns checkbox to the right side of the window
            # https://github.com/ocornut/imgui/issues/196
            imgui.same_line(position=imgui.get_window_content_region_max().x - 25)
            eu, enabled = imgui.checkbox('##enabled_r_{}'.format(r.unique_name), r.enabled)
            if eu:
                r.enabled = enabled

            if r.expanded:
                if r.has_gui:
                    # Recursively render children nodes
                    self.gui_renderables(imgui, r.nodes)
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
            n_frames = max(n_frames, n.n_frames)
        return n_frames

    def capture_selection(self, node):
        # The scene is the common ancestors of all nodes, therefore it should 
        # never be selected when a descendant is clicked.
        return False

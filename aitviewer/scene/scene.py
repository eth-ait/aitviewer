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
from aitviewer.renderables.plane import Chessboard
from aitviewer.scene.light import Light
from aitviewer.scene.node import Node


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
        self.lights.append(Light(name='Back Light', position=(0.0, 10.0, 15.0), color=(1.0, 1.0, 1.0, 1.0)))
        self.lights.append(Light(name='Front Light', position=(0.0, 10.0, -15.0), color=(1.0, 1.0, 1.0, 1.0)))

        # Scene items
        self.origin = CoordinateSystem(name="Origin", length=0.1)
        self.floor = Chessboard(100.0, 200, name="Floor")
        self.floor.mesh.material.diffuse = 0.1
        self.floor.mesh.material._show_edges = False
        self.floor.c1 = (0.9, 0.9, 0.9, 1.0)
        self.floor.c2 = (0.82, 0.82, 0.82, 1.0)
        self.floor._update_colors()

        self.add(self.origin, has_gui=False)
        self.add(self.floor)

        self.custom_font = None

    def render(self, **kwargs):
        # As per https://learnopengl.com/Advanced-OpenGL/Blending

        # Collect all renderable nodes
        rs = self.collect_nodes()

        # Turning off backface culling is available for opaque objects only
        if not self.backface_culling:
            self.ctx.disable(moderngl.CULL_FACE)

        # Draw all opaque objects first
        for r in rs:
            if r.color[3] == 1.0:
                self.safe_render(r, **kwargs)
        if not self.backface_culling:
            self.ctx.enable(moderngl.CULL_FACE)

        # Sort transparent objects by distance to camera (Not done here)
        # Draw all transparent objects
        for r in rs:
            if r.color[3] < 1.0:
                self.safe_render(r, **kwargs)

    def safe_render(self, r, **kwargs):
        if not r.is_renderable:
            r.make_renderable(self.ctx)
        r.render(self.camera, lights=self.lights, **kwargs)

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

    def gui(self, imgui):
        """GUI to control scene settings."""
        self.gui_camera(imgui)
        self.gui_lights(imgui)
        self.gui_renderables(imgui, self.nodes)

    def gui_camera(self, imgui):
        # Camera GUI
        imgui.push_font(self.custom_font)
        camera_expanded = imgui.tree_node("\u0084 Camera##tree_node_r_camera")
        imgui.pop_font()
        if camera_expanded:
            self.camera.gui(imgui)
            imgui.tree_pop()
        imgui.spacing()

    def gui_lights(self, imgui):
        # Lights GUI
        for _, light in enumerate(self.lights):
            imgui.push_font(self.custom_font)
            light_expanded = imgui.tree_node("\u0085 {}##tree_node_r".format(light.name))
            imgui.pop_font()
            if light_expanded:
                light.gui(imgui)
                imgui.tree_pop()
                imgui.spacing()

    def gui_renderables(self, imgui, rs):
        # Nodes GUI
        for i, r in enumerate(rs):
            if r.show_in_hierarchy:
                # Visibility
                curr_enabled = r.enabled
                if not curr_enabled:
                    imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 1.0, 1.0, 0.4)

                # Title
                imgui.push_font(self.custom_font)
                if r.has_gui:
                    r.expanded = imgui.tree_node("{} {}##tree_node_{}".format(r.icon, r.name, r.unique_name),
                                                 r.expanded and imgui.TREE_NODE_DEFAULT_OPEN)
                else:
                    imgui.spacing()
                    imgui.same_line(spacing=24)
                    imgui.text("{} {}".format(r.icon, r.name))
                imgui.pop_font()

                # Aligns checkbox to the right side of the window
                # https://github.com/ocornut/imgui/issues/196
                imgui.same_line(position=imgui.get_window_content_region_max().x - 25)
                eu, enabled = imgui.checkbox('##enabled_r_{}'.format(r.unique_name), r.enabled)
                if eu:
                    r.enabled = enabled

                if r.has_gui and r.expanded:
                    # Render this elements GUI
                    r.gui(imgui)
                    # Recursively render the GUI of this elements children renderables
                    self.gui_renderables(imgui, r.nodes)

                    imgui.spacing()
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

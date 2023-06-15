import struct
from typing import Dict, List, Tuple

import moderngl
import numpy as np
from moderngl_window import BaseWindow, geometry
from moderngl_window.opengl.vao import VAO
from PIL import Image

from aitviewer.scene.camera import CameraInterface, ViewerCamera
from aitviewer.scene.node import Node
from aitviewer.scene.scene import Scene
from aitviewer.shaders import load_program


class Viewport:
    def __init__(self, extents: Tuple[int, int, int, int], camera: CameraInterface):
        self.extents = extents
        self._camera = camera
        self._using_temp_camera = not isinstance(camera, ViewerCamera)

    @property
    def camera(self):
        return self._camera

    @camera.setter
    def camera(self, camera: CameraInterface):
        self._camera = camera
        self._using_temp_camera = not isinstance(camera, ViewerCamera)

    def contains(self, x: int, y: int):
        e = self.extents
        return x >= e[0] and x < e[0] + e[2] and y >= e[1] and y < e[1] + e[3]

    def set_temp_camera(self, camera: CameraInterface):
        self.camera = camera
        self._using_temp_camera = True

    def reset_camera(self):
        if self._using_temp_camera:
            self._using_temp_camera = False

            fwd = self.camera.forward
            pos = self.camera.position

            self.camera = ViewerCamera(45)
            self.camera.position = np.copy(pos)
            self.camera.target = pos + fwd * 3
            self.camera.update_matrices(*self.extents[2:])


class Renderer:
    def __init__(self, ctx: moderngl.Context, wnd: BaseWindow, window_type: str):
        self.ctx = ctx
        self.wnd = wnd
        self.window_type = window_type

        # Shaders for mesh mouse intersection.
        self.frag_pick_prog = load_program("fragment_picking/frag_pick.glsl")
        self.frag_pick_prog["position_texture"].value = 0  # Read from texture channel 0
        self.frag_pick_prog["obj_info_texture"].value = 1  # Read from texture channel 0
        self.picker_output = self.ctx.buffer(reserve=6 * 4)  # 3 floats, 3 ints
        self.picker_vao = VAO(mode=moderngl.POINTS)

        # Shaders for drawing outlines.
        self.outline_draw_prog = load_program("outline/outline_draw.glsl")
        self.outline_quad = geometry.quad_2d(size=(2.0, 2.0), pos=(0.0, 0.0))

        # Create framebuffers
        self.offscreen_p_depth = None
        self.offscreen_p_viewpos = None
        self.offscreen_p_tri_id = None
        self.offscreen_p = None
        self.outline_texture = None
        self.outline_framebuffer = None
        self.headless_fbo_color = None
        self.headless_fbo_depth = None
        self.headless_fbo = None
        self.create_framebuffers()

        # Debug
        self.vis_prog = load_program("visualize.glsl")
        self.vis_quad = geometry.quad_2d(size=(0.9, 0.9), pos=(0.5, 0.5))

    def clear(self, scene: Scene, **kwargs):
        """Clear the window framebuffer and the fragmap framebuffer."""
        # Clear picking buffer.
        self.offscreen_p.clear()

        # Clear background and make sure only the flags we want are enabled.
        if kwargs["transparent_background"]:
            self.wnd.clear(0, 0, 0, 0)
        else:
            if scene.light_mode == "dark":
                self.wnd.clear(0.1, 0.1, 0.1, 1.0)
            else:
                self.wnd.clear(*scene.background_color)

    def create_framebuffers(self):
        """
        Create all framebuffers which depend on the window size.
        This is called once at startup and every time the window is resized.
        """

        # Release framebuffers if they already exist.
        def safe_release(b):
            if b is not None:
                b.release()

        safe_release(self.offscreen_p_depth)
        safe_release(self.offscreen_p_viewpos)
        safe_release(self.offscreen_p_tri_id)
        safe_release(self.offscreen_p)
        safe_release(self.outline_texture)
        safe_release(self.outline_framebuffer)

        # Mesh mouse intersection
        self.offscreen_p_depth = self.ctx.depth_texture(self.wnd.buffer_size)
        self.offscreen_p_viewpos = self.ctx.texture(self.wnd.buffer_size, 4, dtype="f4")
        self.offscreen_p_tri_id = self.ctx.texture(self.wnd.buffer_size, 4, dtype="f4")
        self.offscreen_p = self.ctx.framebuffer(
            color_attachments=[self.offscreen_p_viewpos, self.offscreen_p_tri_id],
            depth_attachment=self.offscreen_p_depth,
        )
        self.offscreen_p_tri_id.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Outline rendering
        self.outline_texture = self.ctx.texture(self.wnd.buffer_size, 1, dtype="f4")
        self.outline_framebuffer = self.ctx.framebuffer(color_attachments=[self.outline_texture])

        # If in headlesss mode we create a framebuffer without multisampling that we can use
        # to resolve the default framebuffer before reading.
        if self.window_type == "headless":
            safe_release(self.headless_fbo_color)
            safe_release(self.headless_fbo_depth)
            safe_release(self.headless_fbo)
            self.headless_fbo_color = self.ctx.texture(self.wnd.buffer_size, 4)
            self.headless_fbo_depth = self.ctx.depth_texture(self.wnd.buffer_size)
            self.headless_fbo = self.ctx.framebuffer(self.headless_fbo_color, self.headless_fbo_depth)

    def render_shadowmap(self, scene: Scene, **kwargs):
        """A pass to render the shadow map, i.e. render the entire scene once from the view of the light."""
        if kwargs["shadows_enabled"]:
            self.ctx.enable_only(moderngl.DEPTH_TEST)
            rs = scene.collect_nodes()

            for light in scene.lights:
                if light.shadow_enabled:
                    light.use(self.ctx)
                    light_matrix = light.mvp()
                    for r in rs:
                        r.render_shadowmap(light_matrix)

    def render_fragmap(self, scene: Scene, camera: CameraInterface, viewport):
        """A pass to render the fragment picking map, i.e. render the scene with world coords as colors."""
        self.ctx.enable_only(moderngl.DEPTH_TEST)
        self.offscreen_p.use()
        self.offscreen_p.viewport = viewport

        rs = scene.collect_nodes()
        for r in rs:
            r.render_fragmap(self.ctx, camera)

    def render_scene(self, scene: Scene, camera: CameraInterface, viewport, **kwargs):
        """Render the current scene to the framebuffer without time accounting and GUI elements."""
        # Bind framebuffer.
        self.wnd.use()

        # Setup viewport.
        self.wnd.fbo.viewport = viewport

        # Configure OpenGL context.
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.CULL_FACE)
        self.ctx.cull_face = "back"
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
            moderngl.ONE,
            moderngl.ONE,
        )

        # Render scene.
        scene.render(
            camera=camera,
            window_size=kwargs["window_size"],
            lights=scene.lights,
            shadows_enabled=kwargs["shadows_enabled"],
            ambient_strength=scene.ambient_strength,  # ???
            fbo=self.wnd.fbo,
        )

    def render_outline(self, nodes, color, camera: CameraInterface, viewport):
        """A pass to render outlines of an object."""
        # Prepare the outline buffer, all objects rendered to this buffer will be outlined.
        self.outline_framebuffer.clear()
        self.outline_framebuffer.use()
        self.outline_framebuffer.viewport = viewport

        # Render outline of the nodes with outlining enabled, this potentially also renders their children.
        for n in nodes:
            n.render_outline(self.ctx, camera)

        # Render the outline effect to the window.
        self.wnd.use()
        self.wnd.fbo.depth_mask = False
        self.wnd.fbo.viewport = viewport
        self.ctx.enable_only(moderngl.NOTHING)
        self.outline_texture.use(0)
        self.outline_draw_prog["outline"] = 0
        self.outline_draw_prog["outline_color"] = color
        self.outline_quad.render(self.outline_draw_prog)
        self.wnd.fbo.depth_mask = True

    def render_viewport(self, scene: Scene, camera: CameraInterface, viewport: Viewport, **kwargs):
        """Render the scene for a viewport with the given camera."""
        # Parameters.
        export = kwargs["export"]
        outline_color = kwargs["outline_color"]
        light_outline_color = kwargs["light_outline_color"]
        selected_outline_color = kwargs["selected_outline_color"]

        # Update camera to viewport.
        width, height = viewport[2:]
        camera.update_matrices(width, height)

        # Disable some renderables for exporting.
        renderables_to_enable: List[Node] = []
        if export:

            def disable_for_export(r: Node):
                if r.enabled:
                    r.enabled = False
                    renderables_to_enable.append(r)

            # Disable lights.
            for l in scene.lights:
                disable_for_export(l.mesh)

            # Disable camera target.
            disable_for_export(scene.camera_target)

        self.render_fragmap(scene, camera, viewport)
        self.render_scene(scene, camera, viewport, **kwargs)
        self.render_outline(
            [n for n in scene.collect_nodes() if n.draw_outline],
            outline_color,
            camera,
            viewport,
        )

        # Re-enable renderables that were disabled for exporting.
        if export:
            for r in renderables_to_enable:
                r.enabled = True

        if not export:
            self.render_outline([l for l in scene.lights if l.enabled], light_outline_color, camera, viewport)

            # If the selected object is a Node render its outline.
            if isinstance(scene.selected_object, Node):
                self.render_outline([scene.selected_object], selected_outline_color, camera, viewport)

    def resize(self, _width: int, _height: int):
        """Resize rendering resources on window resize."""
        self.create_framebuffers()

    def render(self, scene: Scene, **kwargs):
        """Main rendering function."""
        viewports: List[Viewport] = kwargs["viewports"]

        # Viewport 0 always matches the scene camera for backwards compatitibility.
        viewports[0].camera = scene.camera

        # Render to shadowmap, this is can be done once for all viewports.
        self.render_shadowmap(scene, **kwargs)
        self.clear(scene, **kwargs)

        for v in viewports:
            self.render_viewport(scene, v.camera, v.extents, **kwargs)
        self.wnd.fbo.viewport = [0, 0, self.wnd.buffer_size[0], self.wnd.buffer_size[1]]

        if not kwargs["export"]:
            # If visualize is True draw a texture with the object id to the screen for debugging.
            if kwargs["visualize"]:
                self.ctx.enable_only(moderngl.NOTHING)
                self.offscreen_p_tri_id.use(location=0)
                self.vis_prog["hash_color"] = True
                self.vis_quad.render(self.vis_prog)

    def read_fragmap_at_pixel(self, x: int, y: int) -> Tuple[np.ndarray, int, int, int]:
        """Given an x/y screen coordinate, get the intersected object, triangle id, and xyz point in camera space."""

        # Fragment picker uses already encoded position/object/triangle in the frag_pos program textures
        self.frag_pick_prog["texel_pos"].value = (x, y)
        self.offscreen_p_viewpos.use(location=0)
        self.offscreen_p_tri_id.use(location=1)
        self.picker_vao.transform(self.frag_pick_prog, self.picker_output, vertices=1)
        x, y, z, obj_id, tri_id, instance_id = struct.unpack("3f3i", self.picker_output.read())
        return np.array((x, y, z)), obj_id, tri_id, instance_id

    def get_current_frame_as_image(self, alpha=False):
        """Return the FBO content as a PIL image."""
        if alpha:
            fmt = "RGBA"
            components = 4
        else:
            fmt = "RGB"
            components = 3

        # If in headless mode we first resolve the multisampled framebuffer into
        # a non multisampled one and read from that instead.
        if self.window_type == "headless":
            self.ctx.copy_framebuffer(self.headless_fbo, self.wnd.fbo)
            fbo = self.headless_fbo
        else:
            fbo = self.wnd.fbo

        width = self.wnd.fbo.viewport[2] - self.wnd.fbo.viewport[0]
        height = self.wnd.fbo.viewport[3] - self.wnd.fbo.viewport[1]
        image = Image.frombytes(
            fmt,
            (width, height),
            fbo.read(viewport=self.wnd.fbo.viewport, alignment=1, components=components),
        )
        if width != self.wnd.size[0] or height != self.wnd.size[1]:
            image = image.resize(self.wnd.size, Image.NEAREST)

        return image.transpose(Image.FLIP_TOP_BOTTOM)

    def get_current_depth_image(self, camera: CameraInterface):
        """
        Return the depth buffer as a 'F' PIL image.
        Depth is stored as the z coordinate in eye (view) space.
        Therefore values in the depth image represent the distance from the pixel to
        the plane passing through the camera and orthogonal to the view direction.
        Values are between the near and far plane distances of the camera used for rendering,
        everything outside this range is clipped by OpenGL.
        """

        # If in headless mode we first resolve the multisampled framebuffer into
        # a non multisampled one and read from that instead.
        if self.window_type == "headless":
            self.ctx.copy_framebuffer(self.headless_fbo, self.wnd.fbo)
            fbo = self.headless_fbo
        else:
            fbo = self.wnd.fbo

        width = self.wnd.fbo.viewport[2] - self.wnd.fbo.viewport[0]
        height = self.wnd.fbo.viewport[3] - self.wnd.fbo.viewport[1]

        # Get depth image from depth buffer.
        depth = Image.frombytes(
            "F",
            (width, height),
            fbo.read(viewport=self.wnd.fbo.viewport, alignment=1, attachment=-1, dtype="f4"),
        )

        if width != self.wnd.size[0] or height != self.wnd.size[1]:
            depth = depth.resize(self.wnd.size, Image.NEAREST)

        # Convert from [0, 1] range to [-1, 1] range.
        # This is necessary because our projection matrix computes NDC
        # from [-1, 1], but depth is then stored normalized from 0 to 1.
        depth = np.array(depth) * 2.0 - 1.0

        # Extract projection matrix parameters used for mapping Z coordinates.
        P = camera.get_projection_matrix()
        a, b = P[2, 2], P[2, 3]

        # Linearize depth values. This converts from [-1, 1] range to the
        # view space Z coordinate value, with positive z in front of the camera.
        z = b / (a + depth)
        return Image.fromarray(z, mode="F").transpose(Image.FLIP_TOP_BOTTOM)

    def get_current_mask_ids(self, id_map: Dict[int, int] = None):
        """
        Return a mask as a numpy array of shape (height, width) and type np.uint32.
        Each element in the array is the UID of the node covering that pixel (can be accessed from a node with 'node.uid')
        or zero if not covered.

        :param id_map:
            if not None the UIDs in the mask are mapped using this dictionary to the specified ID.
            The final mask only contains the IDs specified in this mapping and zeros everywhere else.
        """
        width = self.wnd.fbo.viewport[2] - self.wnd.fbo.viewport[0]
        height = self.wnd.fbo.viewport[3] - self.wnd.fbo.viewport[1]

        # Get object ids as floating point numbers from the first channel of
        # the first attachment of the picking framebuffer.
        id = Image.frombytes(
            "F",
            (width, height),
            self.offscreen_p.read(
                viewport=self.wnd.fbo.viewport,
                alignment=1,
                components=1,
                attachment=1,
                dtype="f4",
            ),
        )

        if width != self.wnd.size[0] or height != self.wnd.size[1]:
            id = id.resize(self.wnd.size, Image.NEAREST)

        # Convert the id to integer values.
        id_int = np.asarray(id).astype(dtype=np.uint32)

        # If an id_map is given use this to map the ids.
        if id_map is not None:
            output = np.zeros(id_int.shape, dtype=np.uint32)
            for k, v in id_map.items():
                output[id_int == k] = v
            return output
        else:
            # Copy here because the array constructed from a PIL image is read-only.
            return id_int.copy()

    def get_current_mask_image(self, color_map: Dict[int, Tuple[int, int, int]] = None, id_map: Dict[int, int] = None):
        """
        Return a color mask as a 'RGB' PIL image.
        Each object in the mask has a uniform color computed from the Node UID (can be accessed from a node with 'node.uid').

        :param color_map:
            if not None specifies the color to use for a given Node UID as a tuple (R, G, B) of integer values from 0 to 255.
            If None the color is computed as an hash of the Node UID instead.
        :param id_map:
            if not None the UIDs in the mask are mapped using this dictionary from Node UID to the specified ID.
            This mapping is applied before the color map (or before hashing if the color map is None).
        """

        if color_map is None:
            # If no colormap is given hash the ids mapped with id_map.
            ids = self.get_current_mask_ids(id_map)

            # Hash the ids.
            def hash(h):
                h ^= h >> 16
                h *= 0x85EBCA6B
                h ^= h >> 13
                h *= 0xC2B2AE35
                h ^= h >> 16
                return h

            output = hash(ids)
        else:
            # If a colormap is given use it to map from UID to colors.
            ids = self.get_current_mask_ids()

            def rgb_to_uint(r, g, b):
                return ((b & 0xFF) << 16) | ((g & 0xFF) << 8) | (r & 0xFF)

            output = np.zeros(ids.shape, dtype=np.uint32)
            if id_map is not None:
                # If a id_map is given use it before indexing into the color map
                for k, v in id_map.items():
                    output[ids == k] = rgb_to_uint(*color_map[v])
            else:
                # If no id_map is given use the color_map directly.
                for k, v in color_map.items():
                    output[ids == k] = rgb_to_uint(*v)

        # Convert the hashed ids to an RGBA image and then throw away the alpha channel.
        img = Image.frombytes("RGBA", (ids.shape[1], ids.shape[0]), output.tobytes()).convert("RGB")
        return img.transpose(Image.FLIP_TOP_BOTTOM)

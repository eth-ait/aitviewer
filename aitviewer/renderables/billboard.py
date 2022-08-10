import moderngl
import numpy as np

from aitviewer.scene.node import Node
from aitviewer.shaders import get_screen_texture_program, get_smooth_lit_with_edges_program, get_chessboard_program
from aitviewer.utils import set_lights_in_program
from aitviewer.utils import set_material_properties

from PIL import Image
import pickle

from aitviewer.utils.decorators import hooked

class Billboard(Node):
    """ A billboard for displaying a sequence of images as an object in the world"""

    def __init__(self,
                 vertices,
                 texture_paths,
                 **kwargs):
        """ Initializer.
        :param vertices:
            A np array of 4 billboard vertices in world space coordinates of shape (4, 3)
            or an array of shape (N, 4, 3) containing 4 vertices for each frame of the sequence
        :param texture_paths: A list of length N containing paths to the textures as image files.
        """
        super(Billboard, self).__init__(n_frames=len(texture_paths), **kwargs)
        
        if len(vertices.shape) == 2:
            vertices = vertices[np.newaxis]
        else:
            assert vertices.shape[0] == len(texture_paths), "the length of the sequence of vertices must be 1 or match the number of textures"

        self.vertices = vertices

        # Tile the uv buffer to match the size of the vertices buffer,
        # we do this so that we can use the same vertex array for all draws
        self.uvs = np.repeat(np.array([[
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]], np.float32), self.vertices.shape[0], axis=0)

        self.texture_paths = texture_paths

        self.texture = None
        self._current_texture_id = None

        self.backface_culling = False

    @classmethod
    def from_opencv_camera_and_distance(cls, camera, distance, cols, rows, texture_paths):
        """
        Initialize from an OpenCV camera object, a distance from the camera 
        and the size of the image in pixels as expected by the camera intrinsics matrix
        """

        V, P = camera.compute_opengl_view_projection(cols, rows)
        ndc_from_world = P @ V
        
        # Comput z coordinate of a point at the given distance
        world_p = camera.position + camera.forward * distance
        ndc_p = (ndc_from_world @ np.concatenate([world_p, np.array([1])]))

        # Compute z after perspective division
        z = ndc_p[2] / ndc_p[3]

        # Compute world coordinates of NDC corners at the computed distance
        corners = np.array([
            [ 1,  1, z], 
            [ 1, -1, z], 
            [-1,  1, z], 
            [-1, -1, z],
        ])

        world_from_ndc = np.linalg.inv(ndc_from_world)
        def transform(x):
            v = world_from_ndc @ np.append(x, 1.0)
            v = v[:3] / v[3]
            return v
        corners = np.apply_along_axis(transform, 1, corners)

        return cls(corners, texture_paths)

    # noinspection PyAttributeOutsideInit
    @Node.once
    def make_renderable(self, ctx):
        self.prog = get_screen_texture_program()
        self.vbo_vertices = ctx.buffer(self.vertices.astype('f4').tobytes())
        self.vbo_uvs = ctx.buffer(self.uvs.astype('f4').tobytes())
        self.vao = ctx.vertex_array(self.prog,
                                    [(self.vbo_vertices, '3f4 /v', 'in_position'),
                                     (self.vbo_uvs, '2f4 /v', 'in_texcoord_0')])
        self.ctx = ctx

    def render(self, camera, **kwargs):
        if self.current_frame_id != self._current_texture_id:
            if self.texture:
                self.texture.release()
                
            path = self.texture_paths[self.current_frame_id]
            if path.endswith((".pickle", "pkl")):
                img = pickle.load(open(path, "rb"))
                self.texture = self.ctx.texture(img.shape[:2], img.shape[2], img.tobytes())
            else:
                img = Image.open(path).transpose(method=Image.FLIP_TOP_BOTTOM).convert("RGB")
                self.texture = self.ctx.texture(img.size, 3, img.tobytes())
            self._current_texture_id = self.current_frame_id

        self.prog['transparency'] = 1.0
        self.prog['texture0'].value = 0
        self.texture.use(0)

        self.set_camera_matrices(self.prog, camera, **kwargs)
        
        # Compute the index of the first vertex to use if we have a sequence of vertices of length > 1
        first = 4 * self.current_frame_id if self.vertices.shape[0] > 1 else 0
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, first=first)

    @hooked
    def release(self):
        if self.is_renderable:
            self.vbo_vertices.release()
            self.vbo_uvs.release()
            self.vao.release()
            if self.texture:
                self.texture.release()
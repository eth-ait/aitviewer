import moderngl
import numpy as np
import trimesh
from moderngl_window.opengl.vao import VAO

from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.node import Node
from aitviewer.shaders import (
    get_marching_cubes_shader,
    get_smooth_lit_with_edges_program,
)
from aitviewer.utils.decorators import hooked
from aitviewer.utils.marching_cubes_table import TRIS_TABLE
from aitviewer.utils.utils import compute_vertex_and_face_normals_sparse


class Volume(Meshes):
    """
    A signed distance function volume that is meshed with marching cubes directly on the GPU.
    """

    # Constants for marching cubes shader. The product of BX, BY and BZ must be smaller than the maximum supported
    # group size in the opengl compute shader. This is usually at least 1024 on most implementations.
    BX = 8
    BY = 8
    BZ = 8
    COMPACT_GROUP_SIZE = 128

    def __init__(
        self, volume, size=(1, 1, 1), level=0.0, max_triangles=None, max_vertices=None, invert_normals=False, **kwargs
    ):
        """
        Initializer.

        volume:
        :param volume: np array of shape (Z, Y, X) of signed distance values.
        :param size: size of the volume in local units.
        :param level: the level set that is meshed using marching cubes.
        :param max_triangles: the maximum number of triangles allocated for the mesh. If none a default value of 15M is used.
        :param max_vertices: the maximum number of vertices allocated for the mesh. If none a default value of 15M is used.
        :param invert_normals: if true flips the normals of the output mesh.
        :param **kwargs: arguments forwarded to the Meshes constructor.
        """
        # Initialize empty mesh.
        super().__init__(np.zeros((1, 0, 3)), np.zeros((0, 3)), **kwargs)

        # Disable backface culling for this mesh.
        self.backface_culling = False

        # Compute constants for marching cubes.
        self.NZ = volume.shape[0]
        self.NY = volume.shape[1]
        self.NX = volume.shape[2]

        self.BLOCKS_X = (self.NX - 1 + (self.BX - 2)) // (self.BX - 1)
        self.BLOCKS_Y = (self.NY - 1 + (self.BY - 2)) // (self.BY - 1)
        self.BLOCKS_Z = (self.NZ - 1 + (self.BZ - 2)) // (self.BZ - 1)
        self.TOTAL_BLOCKS = self.BLOCKS_X * self.BLOCKS_Y * self.BLOCKS_Z

        if max_triangles is None:
            max_triangles = 1000 * 1000 * 15
        if max_vertices is None:
            max_vertices = 1000 * 1000 * 15
        self.MAX_VERTICES = max_vertices
        self.MAX_TRIANGLES = max_triangles

        # Store parameters.
        self._volume = volume
        self.level = level
        self.size = size
        self.spacing = size / (np.array(volume.shape[::-1]) - 1.0)
        self.invert_normals = invert_normals

        # Internal variables.
        self._need_update = True
        self._need_readback = True
        self._debug_gui = False

    @property
    def vertices(self):
        return self.current_vertices[np.newaxis]

    @vertices.setter
    def vertices(self, _):
        raise Exception(f"vertices cannot be set on {self.__class__.__name__} object")

    @property
    def current_vertices(self):
        self._readback_data()
        return self._vertices_cpu

    @current_vertices.setter
    def current_vertices(self, _):
        raise Exception(f"vertices cannot be set on {self.__class__.__name__} object")

    @property
    def vertex_normals(self):
        return self.current_vertex_normals[np.newaxis]

    @vertex_normals.setter
    def vertex_normals(self, _):
        raise Exception(f"vertex normals cannot be set on {self.__class__.__name__} object")

    @property
    def current_vertex_normals(self):
        self._readback_data()
        return self._normals_cpu

    @current_vertex_normals.setter
    def current_vertex_normals(self, _):
        raise Exception(f"vertex normals cannot be set on {self.__class__.__name__} object")

    @property
    def faces(self):
        self._readback_data()
        return self._faces_cpu

    @faces.setter
    def faces(self, _):
        raise Exception(f"faces cannot be set on {self.__class__.__name__} object")

    @property
    def face_normals(self):
        _vertex_faces_sparse = trimesh.geometry.index_sparse(self.vertices.shape[1], self.faces)
        return compute_vertex_and_face_normals_sparse(self.vertices, self.faces, _vertex_faces_sparse, normalize=True)[
            1
        ]

    @property
    def current_face_normals(self):
        _vertex_faces_sparse = trimesh.geometry.index_sparse(self.vertices.shape[1], self.faces)
        return compute_vertex_and_face_normals_sparse(
            self.current_vertices, self.faces, _vertex_faces_sparse, normalize=True
        )[1]

    @property
    def bounds(self):
        return self.current_bounds

    @property
    def current_bounds(self):
        min = np.array([0, 0, 0])
        max = self.size
        return self.get_bounds(np.vstack((min, max)))

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, volume):
        assert (
            self._volume.shape == volume.shape
        ), f"New volume shape {volume.shape} must match old volume shape {self._volume.shape}"
        self._volume = volume
        if self.is_renderable:
            self._texture3d.write(self._volume.astype(np.float32).tobytes())
        self.redraw()

    @hooked
    def redraw(self, **kwargs):
        self._need_update = True

    def _readback_data(self):
        if self._need_update:
            self._marching_cubes()

        if self._need_readback:
            num_verts = np.frombuffer(self._num_vertices.read(), dtype=np.uint32)[0]
            if num_verts > self.MAX_VERTICES:
                raise Exception(
                    f"Too many vertices (Got: {num_verts}, max is: {self.MAX_VERTICES}). Consider using a higher max_vertices parameter"
                )

            num_indices = np.frombuffer(self._draw_args.read(4), dtype=np.uint32)[0]
            if num_indices // 3 > self.MAX_TRIANGLES:
                raise Exception(
                    f"Too many triangles (Got: {num_indices // 3}, max is: {self.MAX_TRIANGLES}. Consider using a higher max_triangles parameter"
                )

            # This ensures that buffers used by the GPU are never mapped in host memory.
            # If we don't do this the buffers stay mapped on the host and GPU operations on
            # these buffer become significantly slower.
            def readback(buffer, size):
                readback_buffer = self.ctx.buffer(reserve=size)
                self.ctx.copy_buffer(readback_buffer, buffer, size)
                data = readback_buffer.read()
                readback_buffer.release()
                return data

            vert_bytes = readback(self._vertices, num_verts * 3 * 4)
            norm_bytes = readback(self._normals, num_verts * 3 * 4)
            tri_bytes = readback(self._triangles, num_indices * 4)

            self._vertices_cpu = np.frombuffer(vert_bytes, dtype=np.float32).reshape((num_verts, 3))
            self._normals_cpu = np.frombuffer(norm_bytes, dtype=np.float32).reshape((num_verts, 3))
            self._faces_cpu = np.frombuffer(tri_bytes, dtype=np.uint32).reshape((num_indices // 3, 3))

            self._need_readback = False

    def _marching_cubes(self):
        # Check surface blocks.
        with self.time_queries["check_surface"]:
            self._texture3d.bind_to_image(0, read=True, write=False)

            self._all_blocks.bind_to_storage_buffer(0)

            self.prog_check_surface["u_level"] = self.level
            self.prog_check_surface.run(group_x=self.BLOCKS_X, group_y=self.BLOCKS_Y, group_z=self.BLOCKS_Z)
            self.ctx.memory_barrier()

        # Compact surface blocks.
        with self.time_queries["compact"]:
            # Initialize number of groups in x, y, and z for dispatch.
            self._out_blocks.write(np.array([0, 1, 1], dtype=np.uint32))

            self._all_blocks.bind_to_storage_buffer(0)
            self._out_blocks.bind_to_storage_buffer(1)

            self.prog_compact["u_num_blocks"] = self.TOTAL_BLOCKS
            self.prog_compact.run(group_x=(self.TOTAL_BLOCKS + self.COMPACT_GROUP_SIZE - 1) // self.COMPACT_GROUP_SIZE)
            self.ctx.memory_barrier()

        # Compute marching cubes
        with self.time_queries["mc"]:
            # Initialize draw arguments: count, instance_count, first_index, base_vertex, base_instances.
            self._draw_args.write(np.array([0, 1, 0, 0, 0], dtype=np.uint32))
            self._num_vertices.write(b"\x00\x00\x00\x00")

            self._texture3d.bind_to_image(0, read=True, write=False)

            self._out_blocks.bind_to_storage_buffer(0)
            self._tris_table.bind_to_storage_buffer(1)
            self._vertices.bind_to_storage_buffer(2)
            self._normals.bind_to_storage_buffer(3)
            self._triangles.bind_to_storage_buffer(4)
            self._num_vertices.bind_to_storage_buffer(5)
            self._draw_args.bind_to_storage_buffer(6)

            self.prog_mc["u_level"] = self.level
            self.prog_mc["u_blocks"] = (self.BLOCKS_X, self.BLOCKS_Y)
            self.prog_mc["u_volume_spacing"] = tuple(self.spacing)
            self.prog_mc["u_max_indices"] = self.MAX_TRIANGLES * 3
            self.prog_mc["u_max_vertices"] = self.MAX_VERTICES
            self.prog_mc["u_invert_normals"] = self.invert_normals

            self.prog_mc.run_indirect(buffer=self._out_blocks)
            self.ctx.memory_barrier()

        self._need_update = False
        self._need_readback = True

    @Node.once
    def make_renderable(self, ctx: moderngl.Context):
        try:
            # Load shaders for drawing.
            vs = "lit_with_edges.glsl"
            positions_vs = "mesh_positions.vs.glsl"
            self._load_programs(vs, positions_vs)

            # Load compute shaders for marching cubes.
            BX = self.BX
            BY = self.BY
            BZ = self.BZ
            CS = self.COMPACT_GROUP_SIZE
            self.prog_check_surface = get_marching_cubes_shader("check_surface.cs.glsl", BX, BY, BZ, CS)
            self.prog_compact = get_marching_cubes_shader("compact.cs.glsl", BX, BY, BZ, CS)
            self.prog_mc = get_marching_cubes_shader("marching_cubes.cs.glsl", BX, BY, BZ, CS)

            # Upload lookup table.
            self._tris_table = ctx.buffer(TRIS_TABLE.tobytes())

            # Create compute shader buffers.
            self._texture3d = ctx.texture3d(
                (self.NX, self.NY, self.NZ), 1, self.volume.astype(np.float32).tobytes(), dtype="f4"
            )
            self._all_blocks = ctx.buffer(reserve=self.TOTAL_BLOCKS * 4, dynamic=True)
            self._out_blocks = ctx.buffer(reserve=self.TOTAL_BLOCKS * 4 + 12, dynamic=True)
            self._num_vertices = ctx.buffer(reserve=4, dynamic=True)
            self._draw_args = ctx.buffer(reserve=20, dynamic=True)

            self._triangles = ctx.buffer(reserve=self.MAX_TRIANGLES * 12, dynamic=True)
            self._vertices = ctx.buffer(reserve=self.MAX_VERTICES * 12, dynamic=True)
            self._normals = ctx.buffer(reserve=self.MAX_VERTICES * 12, dynamic=True)
            self._colors = ctx.buffer(reserve=16, dynamic=True)

            self.prog = get_smooth_lit_with_edges_program("lit_with_edges.glsl", 0)

            self.vao = VAO()
            self.vao.buffer(self._vertices, "3f4", "in_position")
            self.vao.buffer(self._normals, "3f4", "in_normal")
            self.vao.buffer(self._colors, "4f4", "in_color")
            self.vao.index_buffer(self._triangles)

            self.ctx = ctx

            self.time_queries = {
                "check_surface": ctx.query(time=True),
                "compact": ctx.query(time=True),
                "mc": ctx.query(time=True),
                "render": ctx.query(time=True),
            }
        except Exception as e:
            raise Exception(
                f"Failed to initialize Volume renderable. This renderable requires support for OpenGL 4.5 (not available on macOS and old GPUs).\nError:\n{e}"
            )

    def render(self, camera, **kwargs):
        if self._need_update:
            self._marching_cubes()
        prog = self._use_program(camera, **kwargs)
        with self.time_queries["render"]:
            self.vao.render_indirect(prog, self._draw_args, mode=moderngl.TRIANGLES, count=1)

    def render_positions(self, prog):
        if self._need_update:
            self._marching_cubes()
        self.vao.render_indirect(prog, self._draw_args, mode=moderngl.TRIANGLES, count=1)

    @hooked
    def release(self):
        # vao and vbos are released by Meshes release method.
        if self.is_renderable:
            self._tris_table.release()
            self._texture3d.release()
            self._all_blocks.release()
            self._out_blocks.release()
            self._num_vertices.release()
            self._draw_args.release()

    def gui(self, imgui):
        u, self.level = imgui.drag_float("Level", self.level, 2e-3, min_value=-1.0, max_value=1.0)
        if u:
            self.redraw()

        if self._debug_gui:
            # Draw debugging stats about marching cubes and rendering time.
            total = 0
            for k, v in self.time_queries.items():
                imgui.text(f"{k}: {v.elapsed * 1e-6:5.3f}ms")
                total += v.elapsed * 1e-6
            imgui.text(f"Total: {total: 5.3f}ms")

        super().gui(imgui)

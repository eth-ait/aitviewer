import moderngl
import numpy as np
from moderngl_window.opengl.vao import VAO

from aitviewer.scene.camera import CameraInterface
from aitviewer.scene.node import Node
from aitviewer.shaders import (
    get_gaussian_splat_draw_program,
    get_gaussian_splat_prepare_program,
)
from aitviewer.utils.decorators import hooked
from aitviewer.utils.gpu_sort import GpuSort


class GaussianSplats(Node):
    PREPARE_GROUP_SIZE = 128

    def __init__(self, splat_positions, splat_shs, splat_opacities, splat_scales, splat_rotations, **kwargs):
        super().__init__(**kwargs)

        self.num_splats = splat_positions.shape[0]
        self.splat_positions: np.ndarray = splat_positions
        self.splat_shs: np.ndarray = splat_shs
        self.splat_opacities: np.ndarray = splat_opacities
        self.splat_scales: np.ndarray = splat_scales
        self.splat_rotations: np.ndarray = splat_rotations

        self.splat_opacity_scale = 1.0
        self.splat_size_scale = 1.0

        self.backface_culling = False

        self._debug_gui = False

    def is_transparent(self):
        return True

    @property
    def bounds(self):
        return self.current_bounds

    @property
    def current_bounds(self):
        return self.get_bounds(self.splat_positions)

    @Node.once
    def make_renderable(self, ctx: moderngl.Context):
        self.prog_prepare = get_gaussian_splat_prepare_program(self.PREPARE_GROUP_SIZE)
        self.prog_draw = get_gaussian_splat_draw_program()

        # Buffer for splat positions.
        self.splat_positions_buf = ctx.buffer(self.splat_positions.astype(np.float32).tobytes())

        # Buffer for other splat data: SHs, opacity, scale, rotation.
        #
        # TODO: In theory we could pre-process rotations and scales and store them
        # as a 6 element covariance matrix directly here.
        #
        # TODO: Currently this only renders with base colors (first spherical harmonics coefficient)
        # We need to unswizzle the other coefficients and evaluate them for rendering here.
        splat_data = np.hstack(
            (
                self.splat_shs[:, :3] * 0.2820948 + 0.5,
                self.splat_opacities[:, np.newaxis],
                self.splat_scales,
                np.zeros((self.num_splats, 1), np.float32),
                self.splat_rotations,
            )
        )
        self.splat_data_buf = ctx.buffer(splat_data.astype(np.float32).tobytes())

        # Buffer for splat views.
        self.splat_views_buf = ctx.buffer(None, reserve=self.num_splats * 48)

        # Buffers for distances and sorted indices.
        self.splat_distances_buf = ctx.buffer(None, reserve=self.num_splats * 4)
        self.splat_sorted_indices_buf = ctx.buffer(None, reserve=self.num_splats * 4)

        # Create a vao for rendering a single quad.
        indices = np.array((0, 1, 2, 1, 3, 2), np.uint32)
        self.vbo_indices = ctx.buffer(indices.tobytes())
        self.vao = VAO()
        self.vao.index_buffer(self.vbo_indices)

        self.gpu_sort = GpuSort(ctx, self.num_splats)

        # Time queries for profiling.
        self.time_queries = {
            "prepare": ctx.query(time=True),
            "sort": ctx.query(time=True),
            "draw": ctx.query(time=True),
        }
        self.ctx = ctx

    def render(self, camera: CameraInterface, **kwargs):
        # Convert gaussians from 3D to 2D quads.
        with self.time_queries["prepare"]:
            self.splat_positions_buf.bind_to_storage_buffer(0)
            self.splat_data_buf.bind_to_storage_buffer(1)
            self.splat_views_buf.bind_to_storage_buffer(2)
            self.splat_distances_buf.bind_to_storage_buffer(3)
            self.splat_sorted_indices_buf.bind_to_storage_buffer(4)

            self.prog_prepare["u_opacity_scale"] = self.splat_opacity_scale
            self.prog_prepare["u_scale2"] = np.square(self.splat_size_scale)

            self.prog_prepare["u_num_splats"] = self.num_splats
            V = camera.get_view_matrix()
            P = camera.get_projection_matrix()
            self.prog_prepare["u_limit"] = 1.3 / P[0, 0]
            self.prog_prepare["u_focal"] = kwargs["window_size"][0] * P[0, 0] * 0.5

            self.prog_prepare["u_world_from_object"].write(self.model_matrix.T.astype("f4").tobytes())
            self.prog_prepare["u_view_from_world"].write(V.T.astype("f4").tobytes())
            self.prog_prepare["u_clip_from_world"].write((P @ V).astype("f4").T.tobytes())

            num_groups = (self.num_splats + self.PREPARE_GROUP_SIZE - 1) // self.PREPARE_GROUP_SIZE
            self.prog_prepare.run(num_groups, 1, 1)
            self.ctx.memory_barrier()

        # Sort splats based on distance to camera plane.
        with self.time_queries["sort"]:
            self.gpu_sort.run(self.ctx, self.splat_distances_buf, self.splat_sorted_indices_buf)

        # Render each splat as a 2D quad with instancing.
        with self.time_queries["draw"]:
            self.splat_views_buf.bind_to_storage_buffer(2)
            self.splat_sorted_indices_buf.bind_to_storage_buffer(3)

            self.prog_draw["u_screen_size"].write(np.array(kwargs["window_size"], np.float32).tobytes())

            kwargs["fbo"].depth_mask = False
            self.vao.render(self.prog_draw, moderngl.TRIANGLES, 6, 0, self.num_splats)
            kwargs["fbo"].depth_mask = True

    def gui(self, imgui):
        if self._debug_gui:
            # Draw debugging stats about marching cubes and rendering time.
            total = 0
            for k, v in self.time_queries.items():
                imgui.text(f"{k}: {v.elapsed * 1e-6:5.3f}ms")
                total += v.elapsed * 1e-6
            imgui.text(f"Total: {total: 5.3f}ms")

        _, self.splat_size_scale = imgui.drag_float(
            "Size",
            self.splat_size_scale,
            1e-2,
            min_value=0.001,
            max_value=10.0,
            format="%.3f",
        )

        _, self.splat_opacity_scale = imgui.drag_float(
            "Opacity",
            self.splat_opacity_scale,
            1e-2,
            min_value=0.001,
            max_value=10.0,
            format="%.3f",
        )

        super().gui(imgui)

    @classmethod
    def from_ply(cls, path, sh_degree=3, **kwargs):
        with open(path, "rb") as f:
            # Read header.
            head = f.readline().decode("utf-8").strip().lower()
            if head != "ply":
                print(head)
                raise ValueError(f"Not a ply file: {head}")

            encoding = f.readline().decode("utf-8").strip().lower()
            if "binary_little_endian" not in encoding:
                raise ValueError(f"Invalid encoding: {encoding}")

            elements = f.readline().decode("utf-8").strip().lower()
            count = int(elements.split()[2])

            # Read until end of header.
            while f.readline().decode("utf-8").strip().lower() != "end_header":
                pass

            # Number of 32 bit floats used to encode Spherical Harmonics coefficients.
            # The last multiplication by 3 is because we have 3 components (RGB) for each coefficient.
            sh_coeffs = (sh_degree + 1) * (sh_degree + 1) * 3

            # Position (vec3), normal (vec3), spherical harmonics (sh_coeffs), opacity (float),
            # scale (vec3) and rotation (quaternion). All values are float32 (4 bytes).
            size = count * (3 + 3 + sh_coeffs + 1 + 3 + 4) * 4

            data = f.read(size)
            arr = np.frombuffer(data, dtype=np.float32).reshape((count, -1))

            # Positions.
            position = arr[:, :3].copy()

            # Currently we don't need normals for rendering.
            # normal = arr[:, 3:6].copy()

            # Spherical harmonic coefficients.
            sh = arr[:, 6 : 6 + sh_coeffs].copy()

            # Activate alpha: sigmoid(alpha).
            opacity = 1.0 / (1.0 + np.exp(-arr[:, 6 + sh_coeffs]))

            # Exponentiate scale.
            scale = np.exp(arr[:, 7 + sh_coeffs : 10 + sh_coeffs])

            # Normalize quaternions.
            rotation = arr[:, 10 + sh_coeffs : 14 + sh_coeffs].copy()
            rotation /= np.linalg.norm(rotation, ord=2, axis=1)[..., np.newaxis]

            # Convert from wxyz to xyzw.
            rotation = np.roll(rotation, -1, axis=1)

            return cls(position, sh, opacity, scale, rotation, **kwargs)

    @hooked
    def release(self):
        # vao and vbos are released by Meshes release method.
        if self.is_renderable:
            self.splat_positions_buf.release()
            self.splat_data_buf.release()
            self.splat_views_buf.release()
            self.splat_distances_buf.release()
            self.splat_sorted_indices_buf.release()
            self.vao.release()
            self.gpu_sort.release()
        self.splat_positions = None
        self.splat_shs = None
        self.splat_opacities = None
        self.splat_scales = None
        self.splat_rotations = None

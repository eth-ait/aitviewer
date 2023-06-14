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
import trimesh
from moderngl_window.opengl.vao import VAO

from aitviewer.scene.material import Material
from aitviewer.scene.node import Node
from aitviewer.shaders import (
    get_depth_only_program,
    get_fragmap_program,
    get_lines_instanced_program,
    get_outline_program,
    get_simple_unlit_program,
)
from aitviewer.utils import (
    compute_vertex_and_face_normals,
    set_lights_in_program,
    set_material_properties,
)
from aitviewer.utils.decorators import hooked
from aitviewer.utils.so3 import aa2rot_numpy as aa2rot

_CYLINDER_SECTORS = 8


def _create_disk(n_disks=1, radius=1.0, sectors=None, plane="xz"):
    """
    Create `n_disks` many disks centered at the origin with the given radius1.
    :param n_disks: How many disks to create.
    :param radius: Radius of the disks.
    :param sectors: How many lines to use to approximate the disk. Increasing this will lead to more vertex data.
    :param plane: In which plane to create the disk.
    :return: Vertices as a np array of shape (N, V, 3), and face data as a np array of shape (F, 3).
    """
    assert plane in ["xz", "xy", "yz"]
    sectors = sectors or _CYLINDER_SECTORS
    angle = 2 * np.pi / sectors

    c1 = "xyz".index(plane[0])
    c2 = "xyz".index(plane[1])
    c3 = "xyz".index("xyz".replace(plane[0], "").replace(plane[1], ""))

    # Vertex Data.
    vertices = np.zeros((n_disks, sectors + 1, 3))
    x = radius * np.cos(np.arange(sectors) * angle)
    y = radius * np.sin(np.arange(sectors) * angle)
    vertices[:, 1:, c1] = x
    vertices[:, 1:, c2] = y

    # Faces.
    faces = np.zeros((sectors, 3), dtype=np.int32)
    idxs = np.array(range(1, sectors + 1), dtype=np.int32)
    faces[:, 2] = idxs
    faces[:-1, 1] = idxs[1:]
    faces[-1, 1] = 1

    return {"vertices": vertices, "faces": faces}


def _create_cylinder_from_to(v1, v2, radius1=1.0, radius2=1.0, sectors=None):
    """
    Create cylinders from points v1 to v2.
    :param v1: A np array of shape (N, 3).
    :param v2: A np array of shape (N, 3).
    :param radius1: The radius at the bottom of the cylinder.
    :param radius2: The radius at the top of the cylinder.
    :param sectors: how many lines to use to approximate the bottom disk.
    :return: Vertices and normals as a np array of shape (N, V, 3) and face data in shape (F, 3), i.e. only one
      face array is created for all cylinders.
    """
    sectors = sectors or _CYLINDER_SECTORS
    n_cylinders = v1.shape[0]

    # Create bottom lid.
    bottom = _create_disk(n_disks=n_cylinders, radius=radius1, sectors=sectors)

    # We must also change the winding of the bottom triangles because we have backface culling enabled and
    # otherwise we wouldn't see the bottom lid even if the normals are correct.
    fs_bottom = bottom["faces"]
    fs_bottom[:, 1], fs_bottom[:, 2] = fs_bottom[:, 2], fs_bottom[:, 1].copy()

    # Create top lid.
    top = _create_disk(n_disks=n_cylinders, radius=radius2, sectors=sectors)
    p2 = np.zeros((n_cylinders, 3))
    p2[:, 1] = np.linalg.norm(v2 - v1, axis=-1)
    top["vertices"] = top["vertices"] + p2[:, np.newaxis]

    # Shift indices of top faces by how many vertices the bottom lid has.
    n_vertices = bottom["vertices"].shape[1]
    fs_top = top["faces"] + n_vertices

    # Create the faces that make up the coat between bottom and top lid.
    idxs_bot = np.array(range(1, sectors + 1), dtype=np.int32)
    idxs_top = idxs_bot + n_vertices
    fs_coat1 = np.zeros((sectors, 3), dtype=np.int32)
    fs_coat1[:, 0] = idxs_top
    fs_coat1[:-1, 1] = idxs_top[1:]
    fs_coat1[-1, 1] = idxs_top[0]
    fs_coat1[:, 2] = idxs_bot

    fs_coat2 = np.zeros((sectors, 3), dtype=np.int32)
    fs_coat2[:, 0] = fs_coat1[:, 1]
    fs_coat2[:-1, 1] = idxs_bot[1:]
    fs_coat2[-1, 1] = idxs_bot[0]
    fs_coat2[:, 2] = idxs_bot

    # Concatenate everything to create a single mesh.
    vs = np.concatenate([bottom["vertices"], top["vertices"]], axis=1)
    fs = np.concatenate([fs_bottom, fs_top, fs_coat1, fs_coat2], axis=0)

    # Compute smooth normals.
    vertex_faces = trimesh.Trimesh(vs[0], fs, process=False).vertex_faces
    ns, _ = compute_vertex_and_face_normals(vs[0:1], fs, vertex_faces, normalize=True)
    ns = np.repeat(ns, n_cylinders, axis=0)

    # Rotate cylinders to align the the given data.
    vs, ns = _rotate_cylinder_to(v2 - v1, vs, ns)

    # Translate cylinders to the given positions
    vs += v1[:, np.newaxis]

    return {"vertices": vs, "normals": ns, "faces": fs}


def _create_cone_from_to(v1, v2, radius=1.0, sectors=None):
    """
    Create a cone from points v1 to v2.
    :param v1: A np array of shape (N, 3).
    :param v2: A np array of shape (N, 3).
    :param radius: The radius for the disk of the cone.
    :param sectors: how many lines to use to approximate the bottom disk.
    :return: Vertices and normals as a np array of shape (N, V, 3) and face data in shape (F, 3), i.e. only one
      face array is created for all cones.
    """
    sectors = sectors or _CYLINDER_SECTORS
    n_cylinders = v1.shape[0]

    # Create bottom lid.
    bottom = _create_disk(n_cylinders, radius, sectors)

    # We must also change the winding of the bottom triangles because we have backface culling enabled and
    # otherwise we wouldn't see the bottom lid even if the normals are correct.
    fs_bottom = bottom["faces"]
    fs_bottom[:, 1], fs_bottom[:, 2] = fs_bottom[:, 2], fs_bottom[:, 1].copy()

    # Add the top position as a new vertex.
    p2 = np.zeros((n_cylinders, 3))
    p2[:, 1] = np.linalg.norm(v2 - v1, axis=-1)
    vs = np.concatenate([bottom["vertices"], p2[:, np.newaxis]], axis=1)
    n_vertices = vs.shape[1]
    idx_top = n_vertices - 1

    # Create the faces from the bottom lid to the top point.
    idxs_bot = np.array(range(1, sectors + 1), dtype=np.int32)
    fs_coat = np.zeros((sectors, 3), dtype=np.int32)
    fs_coat[:, 0] = idx_top
    fs_coat[:-1, 1] = idxs_bot[1:]
    fs_coat[-1, 1] = idxs_bot[0]
    fs_coat[:, 2] = idxs_bot

    # Concatenate everything to create a single mesh.
    fs = np.concatenate([fs_bottom, fs_coat], axis=0)

    # Compute smooth normals.
    vertex_faces = trimesh.Trimesh(vs[0], fs, process=False).vertex_faces
    ns, _ = compute_vertex_and_face_normals(vs[0:1], fs, vertex_faces, normalize=True)
    ns = np.repeat(ns, n_cylinders, axis=0)

    # Rotate cones to align the the given data.
    vs, ns = _rotate_cylinder_to(v2 - v1, vs, ns)

    # Translate cones to the given positions
    vs += v1[:, np.newaxis]

    return {"vertices": vs, "normals": ns, "faces": fs}


def _rotate_cylinder_to(target, vs, ns):
    """
    Rotate vertices and normals such that the main axis of the cylinder aligns with `target`.
    :param target: A np array of shape (N, 3) specifying the target direction for each cylinder given in `vs`.
    :param vs: A np array of shape (N, V, 3), i.e. the vertex data for each cylinder.
    :param ns: A np array of shape (N, V, 3), i.e. the normal data for each cylinder.
    :return: The rotated vertices and normals.
    """
    n_cylinders = vs.shape[0]
    s = np.array([[0.0, 1.0, 0.0]]).repeat(n_cylinders, axis=0)
    v = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-16)
    a = np.cross(s, v, axis=1)
    dot = np.sum(s * v, axis=1)
    theta = np.arccos(dot)
    is_col = np.linalg.norm(a, axis=1) < 10e-6
    a[is_col] = np.array([1.0, 0.0, 0.0])
    theta[is_col] = 0.0
    theta[np.logical_and(dot < 0.0, is_col)] = np.pi
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    rot = aa2rot(a * theta[..., np.newaxis])
    vs = np.matmul(rot[:, np.newaxis], vs[..., np.newaxis]).squeeze(-1)
    ns = np.matmul(rot[:, np.newaxis], ns[..., np.newaxis]).squeeze(-1)
    return vs, ns


class Lines(Node):
    """Render lines as cylinders or cones. Can render approx. 600k lines at 40 fps."""

    def __init__(
        self,
        lines,
        r_base=0.01,
        r_tip=None,
        color=(0.0, 0.0, 1.0, 1.0),
        mode="line_strip",
        cast_shadow=True,
        **kwargs,
    ):
        """
        Initializer.
        :param lines: Set of 3D coordinates as a np array of shape (F, L, 3) or (L, 3).
        :param r_base: Thickness of the line.
        :param r_tip: If set, the thickness of the line will taper from r_base to r_tip. If set to 0.0 it will create
          a proper cone.
        :param color: Color of the line (4-tuple) or array of color (N_LINES, 4), one for each line.
        :param mode: 'lines' or 'line_strip'.
            'lines': a line is drawn from point 0 to 1, from 2 to 3, and so on, number of lines is L / 2.
            'line_strip': a line is drawn between all adjacent points, 0 to 1, 1 to 2 and so on, number of lines is L - 1.
        :param cast_shadow: If True the mesh casts a shadow on other objects.
        """
        if len(lines.shape) == 2:
            lines = lines[np.newaxis]
        assert len(lines.shape) == 3
        assert mode == "lines" or mode == "line_strip"
        if mode == "lines":
            assert lines.shape[1] % 2 == 0

        self._lines = lines
        self.mode = mode
        self.r_base = r_base
        self.r_tip = r_tip if r_tip is not None else r_base

        self.vertices, self.faces = self.get_mesh()
        self.n_lines = self.lines.shape[1] // 2 if mode == "lines" else self.lines.shape[1] - 1

        # Define a default material in case there is None.
        if isinstance(color, tuple) or len(color.shape) == 1:
            kwargs["material"] = kwargs.get("material", Material(color=color, ambient=0.2))
            self.line_colors = kwargs["material"].color
        else:
            assert (
                color.shape[1] == 4 and color.shape[0] == self.n_lines
            ), "Color must be a tuple of 4 values or a numpy array of shape (N_LINES, 4)"
            self.line_colors = color

        super(Lines, self).__init__(n_frames=self.lines.shape[0], **kwargs)

        self._need_upload = True
        self.draw_edges = False

        # Render passes.
        self.outline = True
        self.fragmap = True
        self.depth_prepass = True
        self.cast_shadow = cast_shadow

    @property
    def bounds(self):
        bounds = self.get_bounds(self.lines)
        r = max(self.r_base, self.r_tip)
        bounds[:, 0] -= r
        bounds[:, 1] += r
        return bounds

    @property
    def current_bounds(self):
        bounds = self.get_bounds(self.current_lines)
        r = max(self.r_base, self.r_tip)
        bounds[:, 0] -= r
        bounds[:, 1] += r
        return bounds

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, value):
        self._lines = value if len(value.shape) == 3 else value[np.newaxis]
        self.n_frames = self.lines.shape[0]
        self.redraw()

    @property
    def current_lines(self):
        idx = self.current_frame_id if self._lines.shape[0] > 1 else 0
        return self._lines[idx]

    @current_lines.setter
    def current_lines(self, lines):
        assert len(lines.shape) == 2
        idx = self.current_frame_id if self._lines.shape[0] > 1 else 0
        self._lines[idx] = lines
        self.redraw()

    @Node.color.setter
    def color(self, color):
        self.material.color = color
        self.line_colors = color
        self.redraw()

    @property
    def line_colors(self):
        if len(self._line_colors.shape) == 1:
            t = np.tile(np.array(self._line_colors), (self.n_lines, 1))
            return t
        else:
            return self._line_colors

    @line_colors.setter
    def line_colors(self, color):
        if isinstance(color, tuple):
            color = np.array(color)
        self._line_colors = color
        self.redraw()

    def on_frame_update(self):
        self.redraw()

    def redraw(self, **kwargs):
        self._need_upload = True

    @Node.once
    def make_renderable(self, ctx: moderngl.Context):
        self.prog = get_lines_instanced_program()

        vs_path = "lines_instanced_positions.vs.glsl"
        self.outline_program = get_outline_program(vs_path)
        self.depth_only_program = get_depth_only_program(vs_path)
        self.fragmap_program = get_fragmap_program(vs_path)

        self.vbo_vertices = ctx.buffer(self.vertices.astype("f4").tobytes())
        self.vbo_indices = ctx.buffer(self.faces.astype("i4").tobytes())
        self.vbo_instance_base = ctx.buffer(reserve=self.n_lines * 12)
        self.vbo_instance_tip = ctx.buffer(reserve=self.n_lines * 12)
        self.vbo_instance_color = ctx.buffer(reserve=self.n_lines * 16)

        self.vao = VAO()
        self.vao.buffer(self.vbo_vertices, "3f4", "in_position")
        self.vao.buffer(self.vbo_instance_base, "3f4/i", "instance_base")
        self.vao.buffer(self.vbo_instance_tip, "3f4/i", "instance_tip")
        self.vao.buffer(self.vbo_instance_color, "4f4/i", "instance_color")
        self.vao.index_buffer(self.vbo_indices)

    def _upload_buffers(self):
        if not self.is_renderable or not self._need_upload:
            return
        self._need_upload = False

        lines = self.current_lines
        if self.mode == "lines":
            v0s = lines[::2]
            v1s = lines[1::2]
        else:
            v0s = lines[:-1]
            v1s = lines[1:]

        self.vbo_instance_base.write(v0s.astype("f4").tobytes())
        self.vbo_instance_tip.write(v1s.astype("f4").tobytes())

        if len(self._line_colors.shape) > 1:
            self.vbo_instance_color.write(self._line_colors.astype("f4").tobytes())

    def render(self, camera, **kwargs):
        self._upload_buffers()

        prog = self.prog
        prog["r_base"] = self.r_base
        prog["r_tip"] = self.r_tip
        if len(self._line_colors.shape) == 1:
            prog["use_uniform_color"] = True
            prog["uniform_color"] = tuple(self.color)
        else:
            prog["use_uniform_color"] = False
        prog["draw_edges"].value = 1.0 if self.draw_edges else 0.0
        prog["win_size"].value = kwargs["window_size"]
        prog["clip_control"].value = (0, 0, 0)

        self.set_camera_matrices(prog, camera, **kwargs)
        set_lights_in_program(
            prog,
            kwargs["lights"],
            kwargs["shadows_enabled"],
            kwargs["ambient_strength"],
        )
        set_material_properties(prog, self.material)
        self.receive_shadow(prog, **kwargs)
        self.vao.render(prog, moderngl.TRIANGLES, instances=self.n_lines)

    def render_positions(self, prog):
        if self.is_renderable:
            self._upload_buffers()
            prog["r_base"] = self.r_base
            prog["r_tip"] = self.r_tip
            self.vao.render(prog, moderngl.TRIANGLES, instances=self.n_lines)

    def get_mesh(self):
        v0s = np.array([[0, 0, 0]], np.float32)
        v1s = np.array([[0, 0, 1]], np.float32)

        # If r_tip is below a certain threshold, we create a proper cone, i.e. with just a single vertex at the top.
        if self.r_tip < 1e-5:
            data = _create_cone_from_to(v0s, v1s, radius=1.0)
        else:
            data = _create_cylinder_from_to(v0s, v1s, radius1=1.0, radius2=1.0)

        return data["vertices"][0], data["faces"]

    @hooked
    def release(self):
        if self.is_renderable:
            self.vao.release()

    def update_frames(self, lines, frames):
        self.lines[frames] = lines
        self.redraw()

    def add_frames(self, lines):
        if len(lines.shape) == 2:
            lines = lines[np.newaxis]
        self.lines = np.append(self.lines, lines, axis=0)

    def remove_frames(self, frames):
        self.lines = np.delete(self.lines, frames, axis=0)
        self.redraw()


class Lines2D(Node):
    """Render 2D lines."""

    def __init__(
        self,
        lines,
        color=(0.0, 0.0, 1.0, 1.0),
        mode="line_strip",
        **kwargs,
    ):
        """
        Initializer.
        :param lines: Set of 3D coordinates as a np array of shape (F, L, 3) or (L, 3).
        :param color: Color of the line (4-tuple) or array of color (N_LINES, 4), one for each line.
        :param mode: 'lines' or 'line_strip'.
            'lines': a line is drawn from point 0 to 1, from 2 to 3, and so on, number of lines is L / 2.
            'line_strip': a line is drawn between all adjacent points, 0 to 1, 1 to 2 and so on, number of lines is L - 1.
        """
        if len(lines.shape) == 2:
            lines = lines[np.newaxis]
        assert len(lines.shape) == 3
        assert mode == "lines" or mode == "line_strip"
        if mode == "lines":
            assert lines.shape[1] % 2 == 0

        self._lines = lines
        self.mode = mode
        self.n_lines = self.lines.shape[1] // 2 if mode == "lines" else self.lines.shape[1] - 1
        self.is_renderable = False

        # Define a default material in case there is None.
        if isinstance(color, tuple) or len(color.shape) == 1:
            kwargs["material"] = kwargs.get("material", Material(color=color, ambient=0.2))
            self.line_colors = kwargs["material"].color
        else:
            assert (
                color.shape[1] == 4 and color.shape[0] == self.n_lines
            ), "Color must be a tuple of 4 values or a numpy array of shape (N_LINES, 4)"
            self.line_colors = color

        super(Lines2D, self).__init__(n_frames=self.lines.shape[0], **kwargs)

    @property
    def bounds(self):
        bounds = self.get_bounds(self.lines)
        return bounds

    @property
    def current_bounds(self):
        bounds = self.get_bounds(self.current_lines)
        return bounds

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, value):
        self._lines = value if len(value.shape) == 3 else value[np.newaxis]
        self.n_frames = self.lines.shape[0]
        self.redraw()

    @property
    def current_lines(self):
        idx = self.current_frame_id if self._lines.shape[0] > 1 else 0
        return self._lines[idx]

    @current_lines.setter
    def current_lines(self, lines):
        assert len(lines.shape) == 2
        idx = self.current_frame_id if self._lines.shape[0] > 1 else 0
        self._lines[idx] = lines
        self.redraw()

    @Node.color.setter
    def color(self, color):
        self.material.color = color
        self.line_colors = color
        self.redraw()

    @property
    def line_colors(self):
        if len(self._line_colors.shape) == 1:
            t = np.tile(np.array(self._line_colors), (self.n_lines, 1))
            return t
        else:
            return self._line_colors

    @line_colors.setter
    def line_colors(self, color):
        if isinstance(color, tuple):
            color = np.array(color)
        self._line_colors = color
        self.redraw()

    def on_frame_update(self):
        super().on_frame_update()
        self.redraw()

    def _get_vertices(self):
        vertices = self.current_lines
        if self.mode == "line_strip":
            expanded = np.zeros((self.n_lines * 2, 3))
            expanded[::2] = vertices[:-1]
            expanded[1::2] = vertices[1:]
            vertices = expanded
        return vertices

    def _get_colors(self):
        cols = self.line_colors
        doubled = np.zeros((self.n_lines * 2, 4))
        doubled[::2] = cols
        doubled[1::2] = cols
        return doubled

    def redraw(self, **kwargs):
        """Upload the current frame data to the GPU for rendering."""
        if not self.is_renderable:
            return

        self.vbo_vertices.write(self._get_vertices().astype("f4").tobytes())
        self.vbo_colors.write(self._get_colors().astype("f4").tobytes())

    @Node.once
    def make_renderable(self, ctx: moderngl.Context):
        self.prog = get_simple_unlit_program()

        self.vbo_vertices = ctx.buffer(self._get_vertices().astype("f4").tobytes(), dynamic=True)
        self.vbo_colors = ctx.buffer(self._get_colors().astype("f4").tobytes(), dynamic=True)
        self.vao = VAO("lines", mode=moderngl.LINES)
        self.vao.buffer(self.vbo_vertices, "3f", ["in_position"])
        self.vao.buffer(self.vbo_colors, "4f", ["in_color"])

    def render(self, camera, **kwargs):
        self.set_camera_matrices(self.prog, camera, **kwargs)
        self.vao.render(self.prog, vertices=self.n_lines * 2)

    @hooked
    def release(self):
        if self.is_renderable:
            self.vao.release()

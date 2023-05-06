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
import numpy as np
from skimage import measure

from aitviewer.renderables.bounding_boxes import BoundingBoxes
from aitviewer.renderables.lines import Lines
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.node import Node
from aitviewer.utils.decorators import hooked


class SDF(Node):
    """
    Renderable that can be used to draw level sets of a dense SDF volume meshed using marching cubes.

    This renderable internally uses the marching cubes algorithm from skimage.
    For a faster marching cubes implementation see the Volume renderable.
    """

    def __init__(
        self,
        volume,
        size=(1, 1, 1),
        level=0.0,
        color=(0.7, 0.7, 0.7, 1.0),
        level_sets=None,
        level_set_colors=None,
        mc_step_size=1,
        **kwargs,
    ):
        """Initializer.
        :param volume: np array of shape (X, Y, Z) of signed distance values
        :param size: size of the volume in local units.
        :param level: the level set used for the main mesh.
        :param color: color of the main mesh.
        :param level_sets: a list or array of additional level set values to display.
        :param level_set_colors: a list or array of shape (L, 4) of the same length as
            the level_set parameter with colors to use for the additional level sets.
        :param mc_step_size: step size used for marching cubes.
        :param **kwargs: arguments forwarded to the Node constructor.
        """
        assert len(volume.shape) == 3 and len(size) == 3
        kwargs["gui_material"] = False
        super().__init__(**kwargs)

        self.volume = volume
        self.size = np.array((size), np.float32)

        # Mesh.
        verts, faces, normals, _ = measure.marching_cubes(
            volume, level, spacing=self.size / (np.array(self.volume.shape) - 1.0), step_size=mc_step_size
        )
        self.mesh = Meshes(verts, faces, vertex_normals=-normals, color=color, name="Mesh")

        # Level sets.
        self.level_sets: list[Meshes] = []
        if level_sets is not None:
            if level_set_colors is not None:
                assert len(level_sets) == len(level_set_colors)

            for i, s in enumerate(level_sets):
                verts, faces, normals, _ = measure.marching_cubes(
                    volume, s, spacing=self.size / (np.array(self.volume.shape) - 1.0), step_size=mc_step_size
                )
                shell = Meshes(verts, faces, vertex_normals=-normals, name=f"Level {s:.03f}", cast_shadow=False)
                if level_set_colors is not None:
                    shell.color = tuple(level_set_colors[i])

                shell.clip_control = np.array((1, 1, 1))
                shell.clip_value = self.size.copy()
                shell.backface_culling = False

                self.level_sets.append(shell)

        # Bounding box.
        self.bounding_box = BoundingBoxes.from_min_max_diagonal(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([self.size], dtype=np.float32),
            color=(0, 0, 0, 1),
            name="Bounding Box",
            gui_affine=False,
        )

        # Clip plane lines.
        self.clip_lines = []
        for i, axis in enumerate(["X", "Y", "Z"]):
            s0 = self.size[(i + 0) % 3]
            s1 = self.size[(i + 1) % 3]
            s2 = self.size[(i + 2) % 3]
            lines = np.array(
                (
                    [s0, 0, 0],
                    [s0, s1, 0],
                    [s0, s1, s2],
                    [s0, 0, s2],
                    [s0, 0, 0],
                ),
                dtype=np.float32,
            )
            lines = np.roll(lines, axis=1, shift=(0, i))
            color = np.array([0, 0, 0, 1])
            color[i] = 1
            self.clip_lines.append(Lines(lines, cast_shadow=False, color=color, name=f"Clip {axis}", gui_affine=False))

        # Group clip planes under a common parent node.
        self.clip_planes_node = Node("Clip planes", gui_affine=False, gui_material=False)
        self.clip_planes_node.add(*self.clip_lines)

        # Group level sets under a common parent node.
        self.level_sets_node = Node("Level sets", gui_material=False)
        self.level_sets_node.add(*self.level_sets)

        # Add all children nodes.
        self.add(self.mesh, self.level_sets_node, self.bounding_box, self.clip_planes_node)

        # Initialze clip extents and reverse booleans.
        self._clip_extents = np.array([1.0, 1.0, 1.0])
        self._clip_reversed = np.array([False, False, False])

    @classmethod
    def with_level_sets(
        cls,
        volume,
        inside_levels=None,
        inside_color_start=(0.6, 0.6, 0.7, 1.0),
        inside_color_end=(0.2, 0.2, 0.7, 1.0),
        outside_levels=None,
        outside_color_start=(0.7, 0.6, 0.6, 1.0),
        outside_color_end=(0.7, 0.2, 0.2, 1.0),
        **kwargs,
    ) -> "SDF":
        """
        Crates an SDF object with additional level sets inside and/or outside the mesh.
        Additionally a starting and end color can be specified for inside and outside level sets,
        level sets will be colored interpolating between these two values.

        :param volume: np array of shape (X, Y, Z) of signed distance values.
        :param inside_levels: list or array of level set values for the inside of the mesh.
        :param inside_color_start: starting color for the inside level sets.
        :param inside_color_end: end color for the inside level sets.
        :param outside_levels: list or array of level set values for the outside of the mesh.
        :param outside_color_start: starting color for the outside level sets.
        :param outside_color_end: end color for the outside level sets.
        :param **kwargs: forwarded to the SDF constructor.

        :return: SDF renderable.
        """
        level_sets = []
        level_set_colors = []
        if inside_levels is not None:
            inside_levels = np.array(inside_levels)
            inside_color_start = np.array(inside_color_start, dtype=np.float32)
            inside_color_end = np.array(inside_color_end, dtype=np.float32)
            inside_colors = np.linspace(inside_color_start, inside_color_end, inside_levels.shape[0])
            level_sets.append(inside_levels)
            level_set_colors.append(inside_colors)

        if outside_levels is not None:
            outside_levels = np.array(outside_levels)
            outside_color_start = np.array(outside_color_start, dtype=np.float32)
            outside_color_end = np.array(outside_color_end, dtype=np.float32)
            outside_colors = np.linspace(outside_color_start, outside_color_end, outside_levels.shape[0])
            level_sets.append(outside_levels)
            level_set_colors.append(outside_colors)

        level_sets = np.hstack(level_sets)
        level_set_colors = np.vstack(level_set_colors)
        return cls(volume, level_sets=level_sets, level_set_colors=level_set_colors, **kwargs)

    @property
    def bounds(self):
        return self.current_bounds

    @property
    def current_bounds(self):
        min = np.array([0, 0, 0])
        max = self.size
        return self.get_bounds(np.vstack((min, max)))

    @property
    def clip_extents(self):
        """Property controlling the clip extents used to clip the additional level sets."""
        return self._clip_extents

    @clip_extents.setter
    def clip_extents(self, v):
        """Setter of property controlling the clip extents used to clip the additional level sets."""
        assert len(v) == 3
        for i in range(3):
            if self.clip_extents[i] == v[i]:
                continue
            self._clip_extents[i] = v[i]
            self.clip_lines[i].lines[:, :, i] = v[i] * self.size[i]
            self.clip_lines[i].redraw()
            for s in self.level_sets:
                s.clip_value[i] = v[i] * self.size[i]

    @property
    def clip_reversed(self):
        """Property controlling if the clip extents used to clip the additional level sets are reversed."""
        return self._clip_reversed

    @clip_reversed.setter
    def clip_reversed(self, v):
        """Setter of property controlling if the clip extents used to clip the additional level sets are reversed."""
        assert len(v) == 3
        self._clip_reversed = np.array(v)
        for i in range(3):
            for s in self.shells:
                s.clip_control[i] = -1 if self._clip_reversed[i] else 1

    @hooked
    def gui(self, imgui):
        u, val = imgui.drag_float3(
            f"Clip extents##{self.uid}", *self.clip_extents, 2e-3, format="%.3f", min_value=0.0, max_value=1.0
        )
        if u:
            self.clip_extents = val
        imgui.spacing()
        imgui.text("Reverse axis: ")
        imgui.same_line()
        for i, axis in enumerate(["X", "Y", "Z"]):
            _, self.clip_reversed[i] = imgui.checkbox(f"{axis} ", self.clip_reversed[i])
            if i != 2:
                imgui.same_line()
            for s in self.level_sets:
                s.clip_control[i] = -1 if self._clip_reversed[i] else 1

    def render_outline(self, *args, **kwargs):
        # Disable outline when the node itself is selected.
        pass

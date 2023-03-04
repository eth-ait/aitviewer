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
    def __init__(self, volume, size=(1, 1, 1), level=0.0, shells=None, shell_colors=None, mc_step_size=1, **kwargs):
        assert len(volume.shape) == 3 and len(size) == 3
        super().__init__(**kwargs)

        self.volume = volume
        self.size = np.array((size), np.float32)

        # Mesh.
        verts, faces, normals, _ = measure.marching_cubes(
            volume, level, spacing=self.size / (np.array(self.volume.shape) - 1.0), step_size=mc_step_size
        )
        self.mesh = Meshes(verts, faces, vertex_normals=-normals, name="Mesh")

        # Shells.
        self.shells: list[Meshes] = []
        if shells is not None:
            if shell_colors is not None:
                assert len(shells) == len(shell_colors)

            for i, s in enumerate(shells):
                verts, faces, normals, _ = measure.marching_cubes(
                    volume, s, spacing=self.size / (np.array(self.volume.shape) - 1.0), step_size=mc_step_size
                )
                shell = Meshes(verts, faces, vertex_normals=-normals, name=f"Level {s:.03f}", cast_shadow=False)
                if shell_colors is not None:
                    shell.color = tuple(shell_colors[i])

                shell.clip_control = np.array((1, 1, 1))
                shell.clip_value = self.size.copy()
                shell.backface_culling = False

                self.shells.append(shell)

        # Bounding box.
        self.bounding_box = BoundingBoxes.from_min_max_diagonal(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([self.size], dtype=np.float32),
            color=(0, 0, 0, 1),
            name="Bounding Box",
        )

        # Clip lines.
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
            self.clip_lines.append(Lines(lines, cast_shadow=False, color=color, name=f"Clip {axis}"))

        self.add(self.mesh, self.bounding_box, *self.clip_lines, *self.shells)

        self._clip_extents = np.array([1.0, 1.0, 1.0])
        self._clip_reversed = np.array([False, False, False])

    @property
    def clip_extents(self):
        return self._clip_extents

    @clip_extents.setter
    def clip_extents(self, v):
        assert len(v) == 3
        for i in range(3):
            if self.clip_extents[i] == v[i]:
                continue
            self._clip_extents[i] = v[i]
            self.clip_lines[i].lines[:, :, i] = v[i] * self.size[i]
            self.clip_lines[i].redraw()
            for s in self.shells:
                s.clip_value[i] = v[i] * self.size[i]

    @property
    def clip_reversed(self):
        return self._clip_reversed

    @clip_reversed.setter
    def clip_reversed(self, v):
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
        for i, axis in enumerate(["X", "Y", "Z"]):
            _, self.clip_reversed[i] = imgui.checkbox(f"Reverse {axis}", self.clip_reversed[i])
            for s in self.shells:
                s.clip_control[i] = -1 if self._clip_reversed[i] else 1

    def render_outline(self, *args, **kwargs):
        pass

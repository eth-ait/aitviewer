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
import trimesh

from aitviewer.renderables.meshes import Meshes
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # Number of frames.
    N = 200

    # Load a simple untextured cube.
    cube = trimesh.load("resources/cube.obj")

    # Create sequences of position, scale and rotation values.
    p1 = np.linspace(np.array([0, 0, 0]), np.array([5, 0, 0]), num=N)
    p2 = np.linspace(np.array([0, 0, 0]), np.array([0, 0, 5]), num=N)
    p3 = np.linspace(np.array([0, 0, 0]), np.array([0, 5, 0]), num=N)
    r2 = aa2rot_numpy(np.linspace([0, 0, 0], np.array([0, np.pi, 0]), num=N))
    s3 = np.linspace(1.0, 2.0, num=N)

    # Create 3 cubes, specifying sequences for their position, rotation and/or scale.
    # Each property can be either a single value (e.g. a single position of shape (3)) or an array
    # of values (e.g. an array of positions with shape (N, 3) where N is the number of frames).
    # Array properties will be animated when playing the sequence (you can start playing pressing the spacebar).
    c1 = Meshes(
        cube.vertices,
        cube.faces,
        name="C1",
        position=p1,
        color=(0.5, 0, 0, 1),
        flat_shading=True,
    )
    c2 = Meshes(
        cube.vertices,
        cube.faces,
        name="C2",
        position=p2 + [3.0, 0.0, 0.0],
        rotation=r2,
        color=(0.3, 0, 0, 1),
        flat_shading=True,
    )
    c3 = Meshes(
        cube.vertices,
        cube.faces,
        name="C3",
        position=p3 + [3.0, 0.0, 0.0],
        scale=s3,
        color=(0.1, 0.1, 0.1, 1),
        flat_shading=True,
    )

    # Some properties of renderable objects can also be animated, such as the vertices of a mesh.
    # Here we create an array of vertices and vertex colors with shape (N, V, 3) and (N, V, 4) respectively.

    # Load a simple sphere.
    sphere = trimesh.load("resources/planet/planet.obj")

    # Create initial and final vertex positions.
    vertices_begin = sphere.vertices
    vertices_end = sphere.vertices.copy()
    for i in range(3):
        # Clamp vertices in each 3D direction preserving the sign to create a cube from the sphere.
        vertices_begin[:, i] = np.sign(vertices_end[:, i]) * np.minimum(
            np.abs(vertices_begin[:, i]), np.full((vertices_begin.shape[0]), 1.7)
        )
    # Linearly interpolate vertex positions.
    vertices = np.linspace(vertices_begin, vertices_end, N)

    # Linearly interpolate vertex colors.
    vertex_colors_begin = np.tile(np.array([0, 0, 0.5, 1]), (vertices.shape[1], 1))
    vertex_colors_end = np.tile(np.array([0, 0.5, 0.0, 1]), (vertices.shape[1], 1))
    vertex_colors = np.linspace(vertex_colors_begin, vertex_colors_end, N)

    # Create the node with the vertices and colors we computed.
    cubesphere = Meshes(
        vertices,
        sphere.faces,
        name="CubeSphere",
        position=[-5, 2, 0],
        scale=0.5,
        vertex_colors=vertex_colors,
        flat_shading=True,
    )

    # Create a viewer.
    v = Viewer()

    # Set the camera position.
    v.scene.camera.position = (0, 10, 20)

    # Add the parent object to the scene.
    v.scene.add(c1)

    # Add the other two cubes as children of the first and second cube respectively creating a hierarchy of 3 nodes.
    # The transform (position, rotation and scale) of each node is also applied to all children.
    c1.add(c2)
    c2.add(c3)

    # Add the animated mesh to the scene.
    v.scene.add(cubesphere)

    v.run()

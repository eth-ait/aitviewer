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
import numpy as np
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.material import Material
from aitviewer.scene.node import Node

def _create_spheres(radius=1.0, rings=16, sectors=32, n_spheres=1, create_faces=True):
    """
    Create a sphere centered at the origin. This is a port of moderngl-window's geometry.sphere() function, but it
    returns the vertices, normals, and faces explicitly instead of directly storing them in a VAO.
    :param radius: Radius of the sphere.
    :param rings: Longitudinal resolution.
    :param sectors: Latitudinal resolution.
    :param n_spheres: How many spheres to create.
    :param create_faces: Whether or not to create and return faces.
    :return: vertices, normals, and faces of the sphere.
    """
    R = 1.0 / (rings - 1)
    S = 1.0 / (sectors - 1)

    vertices = np.zeros([n_spheres, rings * sectors, 3])
    normals = np.zeros([n_spheres, rings * sectors, 3])

    v, n = 0, 0
    for r in range(rings):
        for s in range(sectors):
            y = np.sin(-np.pi / 2 + np.pi * r * R)
            x = np.cos(2 * np.pi * s * S) * np.sin(np.pi * r * R)
            z = np.sin(2 * np.pi * s * S) * np.sin(np.pi * r * R)

            vertices[:, v] = np.array([x, y, z]) * radius
            normals[:, n] = np.array([x, y, z])

            v += 1
            n += 1

    if create_faces:
        faces = np.zeros([n_spheres, rings * sectors * 2, 3], dtype=np.int32)
        i = 0
        for r in range(rings - 1):
            for s in range(sectors - 1):
                faces[:, i] = np.array([r * sectors + s,
                                        (r + 1) * sectors + (s + 1),
                                        r * sectors + (s + 1)])
                faces[:, i + 1] = np.array([r * sectors + s,
                                            (r + 1) * sectors + s,
                                            (r + 1) * sectors + (s + 1)])
                i += 2
    else:
        faces = None

    return {'vertices': vertices, 'normals': normals, 'faces': faces}


class Spheres(Node):
    """Render some simple spheres."""

    def __init__(self,
                 positions,
                 radius=0.01,
                 color=(0.0, 0.0, 1.0, 1.0),
                 rings=16,
                 sectors=32,
                 **kwargs):
        """
        Initializer.
        :param positions: A numpy array of shape (F, N, 3) or (N, 3) containing N sphere positions for F time steps.
        :param radius: Radius of the spheres.
        :param color: Color of the spheres.
        :param rings: Longitudinal resolution.
        :param sectors: Latitudinal resolution.
        :param kwargs: Remaining parameters.
        """
        if len(positions.shape) == 2:
            positions = positions[np.newaxis]
        assert len(positions.shape) == 3

        # Define a default material in case there is None.
        kwargs['material'] = kwargs.get('material', Material(color=color, ambient=0.2))
        super(Spheres, self).__init__(n_frames=positions.shape[0], **kwargs)

        self.sphere_positions = positions
        self.n_spheres = positions.shape[1]
        self.spheres_data = _create_spheres(radius=1.0, rings=rings, sectors=sectors, n_spheres=self.n_spheres)

        self.n_vertices = self.spheres_data['vertices'].shape[1]
        self.sphere_vertices = np.reshape(self.spheres_data['vertices'], [-1, 3])
        self.sphere_normals = np.reshape(self.spheres_data['normals'], [-1, 3])
        sphere_faces = [self.spheres_data['faces'][0]] + [self.spheres_data['faces'][i] + i * self.n_vertices for i in
                                                          range(1, self.n_spheres)]
        self.sphere_faces = np.concatenate(sphere_faces)
        self.radius = radius

        # A mesh representing the spheres for a single frame
        self.mesh = Meshes(self.sphere_vertices, self.sphere_faces, self.sphere_normals, material=self.material,
                           cast_shadow=False, is_selectable=False)
        self.mesh.position = self.position
            
        self.add(self.mesh, show_in_hierarchy=False)

    @property
    def vertex_colors(self):
        return np.full((self.n_spheres * self.n_vertices, 4), self.color)
        
    @vertex_colors.setter
    def vertex_colors(self, vertex_colors):
        self.mesh.vertex_colors = vertex_colors

    @property
    def current_sphere_positions(self):
        return self.sphere_positions[self.current_frame_id]
    
    @current_sphere_positions.setter
    def current_sphere_positions(self, positions):
        assert len(positions.shape) == 2
        self.sphere_positions[self.current_frame_id] = positions

    def on_frame_update(self):  
        self.redraw()
    
    def redraw(self, **kwargs):
        current_pos = self.sphere_positions[self.current_frame_id]
        vertices = np.reshape(self.sphere_vertices, [-1, self.n_vertices, 3]) * self.radius + current_pos[:, np.newaxis]
        self.mesh._vertices =np.reshape(vertices, [-1, 3])[np.newaxis]
        super().redraw(**kwargs)

    @Node.once
    def make_renderable(self, ctx):
        self.redraw()
    
    @property
    def color(self):
        return self.mesh.color
    
    @color.setter
    def color(self, color):
        self.mesh.color = color

    def get_index_from_node_and_triangle(self, node, tri_id):
        if node == self.mesh:
            return tri_id // (self.spheres_data['faces'].shape[1])
        
        return None

    def gui_scale(self, imgui):
        # Scale controls
        u, scale = imgui.drag_float('Radius##radius{}'.format(self.unique_name), self.radius, 0.01,
                                    min_value=0.001,
                                    max_value=10.0, format='%.3f')
        if u:
            self.radius = scale
            self.redraw()

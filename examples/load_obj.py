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
import trimesh

from aitviewer.renderables.meshes import Meshes
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    # Load a simple untextured cube.
    cube = trimesh.load("resources/cube.obj")
    cube_mesh = Meshes(
        cube.vertices,
        cube.faces,
        name="Cube",
        position=[-7.0, 0.0, 0.0],
        flat_shading=True,
    )

    # Load a sphere with a texture.
    planet = trimesh.load("resources/planet/planet.obj")
    texture_image = "resources/planet/mars.png"
    planet_mesh = Meshes(
        planet.vertices,
        planet.faces,
        planet.vertex_normals,
        uv_coords=planet.visual.uv,
        path_to_texture=texture_image,
        position=[7.0, 0.0, 0.0],
    )

    # Load a high-res object with texture and scale it.
    # Drill object taken from https://github.com/mmatl/pyrender/tree/master/examples/models
    drill = trimesh.load("resources/drill/drill.obj")
    texture_image = "resources/drill/drill_uv.png"
    drill_mesh = Meshes(
        drill.vertices,
        drill.faces,
        drill.vertex_normals,
        uv_coords=drill.visual.uv,
        path_to_texture=texture_image,
        scale=50.0,
        color=(1, 1, 1, 0.5),
    )

    # Display in viewer.
    v = Viewer()
    v.scene.camera.dolly_zoom(-100.0)
    v.scene.add(planet_mesh, drill_mesh, cube_mesh)
    v.run()

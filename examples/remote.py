from aitviewer.remote.viewer import RemoteViewer
from aitviewer.remote.renderables.meshes import Meshes

import trimesh
from time import sleep

cube = trimesh.load('resources/cube.obj')

if False:
    v = RemoteViewer("localhost")
    m = Meshes(cube.vertices, cube.faces, flat_shading=True, name=f"Cube", position=(5, 0, 0))
    v.scene.add(m)
    v.close()
else:
    v = RemoteViewer("10.0.0.1")
    m = Meshes(cube.vertices, cube.faces, flat_shading=True, name=f"Cube", position=(5, 0, 0))

    # m.append(cube.vertices)

    v.scene.add(m)
    m.append(cube.vertices)
    v.close()

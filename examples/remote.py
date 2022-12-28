from aitviewer.remote.viewer import RemoteViewer
from aitviewer.remote.renderables.meshes import Meshes

import trimesh

cube = trimesh.load('resources/cube.obj')

v = RemoteViewer()
m = Meshes(cube.vertices, cube.faces, flat_shading=True, name=f"Cube", position=(5, 0, 0))
v.scene.add(m)
v.close()

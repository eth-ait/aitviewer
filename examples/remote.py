from aitviewer.remote import RemoteViewer
import trimesh
from time import sleep

cube = trimesh.load('resources/cube.obj')

v = RemoteViewer() # "localhost")
for i in range(10):
    v.mesh(cube.vertices, cube.faces, flat_shading=True, name=f"Cube {i}", position=(5 * i, 0, 0))
    sleep(2)
v.close()

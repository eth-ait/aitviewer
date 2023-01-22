import numpy as np
import trimesh

from aitviewer.remote.renderables.meshes import RemoteMeshes
from aitviewer.remote.viewer import RemoteViewer

cube = trimesh.load("resources/cube.obj")

# Create a RemoteViewer instance. This represents a connection to a remote viewer and
# must be passed as the first argument to every Remote* renderable constructor.
#
# If no argument is given a new viewer is created locally in a new process.
# Instead, an address can be passed to connect to a running viewer.
v = RemoteViewer()


# Create a cube with a single frame on the remote viewer.
# All arguments after the first argument (the remote viewer) are forwarded to the Meshes constructor.
m = RemoteMeshes(
    v,
    cube.vertices,
    cube.faces,
    flat_shading=True,
    name=f"Cube",
    position=(1, 0, 0),
    scale=0.1,
)


# Append 100 frames of vertex data of shape (V, 3) to the remote viewer one by one.
for val in np.linspace(1, 5, 100):
    m.add_frames(cube.vertices * val)

# Append 50 frames of vertex data of shape (50, V, 3) at the same time.
vertices = np.repeat(cube.vertices[np.newaxis], 50, 0)
m.add_frames(vertices)


# Update frame number 3 of the mesh with new vertex data, the previous vertex data will be overwritten.
m.update_frames(cube.vertices * 0.5, frames=3)

# Update two frames.
vals = np.append((cube.vertices * 0.2)[np.newaxis], (cube.vertices * 3)[np.newaxis], axis=0)
m.update_frames(vals, frames=[80, 83])


# Remove a single frame of data, the total number of frames of the sequence will be reduced by one
# and frames after the one that is removed will be shifted back by one position.
m.remove_frames(frames=85)

# Remove a range of frames. As for the line above this reduces the number of frames and shifts
# back the following frames.
m.remove_frames(frames=range(90, 100))

# Change the current displayed frame of the viewer.
#
# You can also use v.next_frame() and v.previous_frame() to change the current frame.
v.set_frame(50)

# Uncomment the following line to remove the node completely from the viewer.
# m.delete()


# Wait until the viewer is closed.
#
# If the viewer is running locally in a new process this will
# wait for it to exit and print its output to the console.
# If the viewer is not running locally this call will return immediately.
v.wait_close()

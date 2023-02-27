import numpy as np
import trimesh

from aitviewer.remote.renderables.meshes import RemoteMeshes
from aitviewer.remote.viewer import RemoteViewer

# Load and generate 200 frames of vertex data.
N = 200
sphere = trimesh.load("resources/planet/planet.obj")
vertices_begin = sphere.vertices
vertices_end = sphere.vertices.copy()
for i in range(3):
    vertices_begin[:, i] = np.sign(vertices_end[:, i]) * np.minimum(
        np.abs(vertices_begin[:, i]), np.full((vertices_begin.shape[0]), 1.7)
    )
vertices = np.linspace(vertices_begin, vertices_end, N)
faces = sphere.faces


# Create a RemoteViewer instance. This represents a connection to a remote viewer and
# must be passed as the first argument to every Remote* renderable constructor.
#
# The 'create_new_process()' helper method first creates a viewer in a new process and
# then connects to it locally.
#
# You can also use connect to an already running viewer as follows:
# v = RemoteViewer("localhost")
v: RemoteViewer = RemoteViewer.create_new_process()


# Mesh 0

# Create a mesh with a single frame on the remote viewer.
# All arguments after the first argument (the remote viewer) are forwarded to the Meshes constructor.
m0 = RemoteMeshes(v, vertices[0], faces, flat_shading=True, name=f"Mesh 0", position=(-1, 0, 0), scale=0.12)

# Append 99 frames of vertex data of shape (V, 3) to the remote viewer one by one.
for i in range(1, 100):
    m0.add_frames(vertices[i])

# Append the remaining 100 frames of vertex data of shape (100, V, 3) at the same time.
m0.add_frames(vertices[100:200])


# Mesh 1

# Create a new mesh with all 200 frames.
m1 = RemoteMeshes(v, vertices, faces, flat_shading=True, name=f"Mesh 1", position=(0, 0, 0), scale=0.12)

# Update frame number 100 of the mesh with new vertex data, the previous vertex data will be overwritten.
m1.update_frames(vertices[99], frames=100)

# Update frames from 101 to 199 all at the same time
m1.update_frames(np.flip(vertices[0:99], axis=0), frames=range(101, 200))


# Mesh 2

# Create a new mesh with all 200 frames.
m2 = RemoteMeshes(v, vertices, faces, flat_shading=True, name=f"Mesh 2", position=(1, 0, 0), scale=0.12)

# Remove a single frame of data, the total number of frames of the sequence will be reduced by one
# and frames after the one that is removed will be shifted back by one position.
m2.remove_frames(frames=100)

# Remove a range of frames. As for the line above this reduces the number of frames and shifts
# back the following frames.
m2.remove_frames(frames=range(100, 199))

# Uncomment the following line to remove the node completely from the viewer.
# m2.delete()

# Change the current displayed frame of the viewer.
#
# You can also use v.next_frame() and v.previous_frame() to change the current frame.
v.set_frame(1)


# Cleanup

# We close the connection manually using 'v.close_connection()' instead.
# Closing the connection ensures that all the data that is waiting to be sent
# is flushed before exiting.
v.close_connection()

# You can also use a 'with' block on the RemoteViewer to ensure that `close_connection()`
# is closed before exiting. For example:
#
# with RemoteViewer("127.0.0.1") as v:
#     ...

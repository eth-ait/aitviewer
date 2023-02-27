from ..message import Message
from ..node import RemoteNode


class RemoteMeshes(RemoteNode):
    MESSAGE_TYPE = Message.MESHES

    def __init__(self, viewer, vertices, faces, **kwargs):
        """
        This initializer takes a RemoteViewer object and all other arguments are forwarded
        to the Meshes constructor on the remote Viewer.
        See the Meshes class for more information about parameters.

        :param viewer: a RemoteViewer object that will be used to send this node.
        :param vertices: Set of 3D coordinates as a np array of shape (N, V, 3) or (V, 3).
        """
        super().__init__(
            viewer,
            vertices=vertices,
            faces=faces,
            **kwargs,
        )

    def add_frames(self, vertices):
        """
        Add frames to the remote Meshes node by adding new vertices.

        :param vertices: Set of 3D coordinates as a np array of shape (N, V, 3) or (V, 3).
        """
        return super().add_frames(vertices=vertices)

    def update_frames(self, vertices, frames):
        """
        Update frames of the remote Meshes node by adding new vertices.

        :param vertices: Set of 3D coordinates as a np array of shape (N, V, 3) or (V, 3).
        :param frames: a list of integer frame indices of size N or a single integer frame index.
        """
        return super().update_frames(vertices=vertices, frames=frames)

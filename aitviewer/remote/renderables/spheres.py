from ..message import Message
from ..node import RemoteNode


class RemoteSpheres(RemoteNode):
    MESSAGE_TYPE = Message.SPHERES

    def __init__(self, viewer, positions, **kwargs):
        """
        This initializer takes a RemoteViewer object and all other arguments are forwarded
        to the Spheres constructor on the remote Viewer.
        See the Spheres class for more information about parameters.

        :param viewer: a RemoteViewer object that will be used to send this node.
        :param positions: A numpy array of shape (N, P, 3) or (P, 3) containing sphere positions.
        """
        super().__init__(
            viewer,
            positions=positions,
            **kwargs,
        )

    def add_frames(self, positions):
        """
        Add frames to the remote Spheres node by adding new sphere positions.

        :param positions: A numpy array of shape (N, P, 3) or (P, 3) containing sphere positions.
        """
        return super().add_frames(positions=positions)

    def update_frames(self, positions, frames):
        """
        Update frames of the remote Spheres node by updating the sphere positions.

        :param positions: A numpy array of shape (N, P, 3) or (P, 3) containing sphere positions.
        :param frames: a list of integer frame indices of size N or a single integer frame index.
        """
        return super().update_frames(positions=positions, frames=frames)

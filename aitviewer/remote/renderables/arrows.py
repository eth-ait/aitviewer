from ..message import Message
from ..node import RemoteNode


class RemoteArrows(RemoteNode):
    MESSAGE_TYPE = Message.ARROWS

    def __init__(self, viewer, origins, tips, **kwargs):
        """
        This initializer takes a RemoteViewer object and all other arguments are forwarded
        to the Arrows constructor on the remote Viewer.
        See the Arrows class for more information about parameters.

        :param viewer: a RemoteViewer object that will be used to send this node.
        :param origins: Set of 3D coordinates of the base of the arrows as a np array of shape (N, B, 3) or (B, 3).
        :param tips: Set of 3D coordinates denoting the tip of the arrow (N, T, 3) or (T, 3).
        """
        super().__init__(
            viewer,
            origins=origins,
            tips=tips,
            **kwargs,
        )

    def add_frames(self, origins, tips):
        """
        Add frames to the remote Arrows node by adding new origins and tips.

        :param origins: Set of 3D coordinates of the base of the arrows as a np array of shape (N, B, 3) or (B, 3).
        :param tips: Set of 3D coordinates denoting the tip of the arrow (N, T, 3) or (T, 3).
        """
        return super().add_frames(origins=origins, tips=tips)

    def update_frames(self, origins, tips, frames):
        """
        Update frames of the remote Arrows node by updating the origins and tips.

        :param origins: Set of 3D coordinates of the base of the arrows as a np array of shape (N, B, 3) or (B, 3).
        :param tips: Set of 3D coordinates denoting the tip of the arrow (N, T, 3) or (T, 3).
        :param frames: a list of integer frame indices of size N or a single integer frame index.
        """
        return super().update_frames(origins=origins, tips=tips, frames=frames)

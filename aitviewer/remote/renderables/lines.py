from ..message import Message
from ..node import RemoteNode


class RemoteLines(RemoteNode):
    MESSAGE_TYPE = Message.LINES

    def __init__(self, viewer, lines, **kwargs):
        """
        This initializer takes a RemoteViewer object and all other arguments are forwarded
        to the Lines constructor on the remote Viewer.
        See the Lines class for more information about parameters.

        :param viewer: a RemoteViewer object that will be used to send this node.
        :param lines: Set of 3D coordinates as a np array of shape (N, L, 3) or (L, 3).
        """
        super().__init__(
            viewer,
            lines=lines,
            **kwargs,
        )

    def add_frames(self, lines):
        """
        Add frames to the remote Lines node by adding new lines.

        :param lines: Set of 3D coordinates as a np array of shape (N, L, 3) or (L, 3).
        """
        return super().add_frames(lines=lines)

    def update_frames(self, lines, frames):
        """
        Update frames of the remote Lines node by updating the origins and tips.

        :param lines: Set of 3D coordinates as a np array of shape (N, L, 3) or (L, 3).
        :param frames: a list of integer frame indices of size N or a single integer frame index.
        """
        return super().update_frames(lines=lines, frames=frames)

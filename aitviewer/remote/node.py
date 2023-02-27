from .message import Message
from .viewer import RemoteViewer

GLOBAL_ID = 0


def next_id():
    global GLOBAL_ID
    GLOBAL_ID += 1
    return GLOBAL_ID


class RemoteNode:
    MESSAGE_TYPE = Message.NODE

    def __init__(self, viewer: RemoteViewer, *args, **kwargs):
        """
        This initializer takes a RemoteViewer object and all other arguments are forwarded
        to the Node constructor on the remote Viewer.
        See the Node class for more information about parameters.

        :param viewer: a RemoteViewer object that will be used to send this node.
        """
        self.viewer = viewer
        self.uid = next_id()
        self._send_msg(self.MESSAGE_TYPE, *args, **kwargs)

    def _send_msg(self, type, *args, **kwargs):
        # Helper function to send a message with the node uid.
        self.viewer.send_message(type, self.uid, *args, **kwargs)

    def add_frames(self, *args, **kwargs):
        """
        Add frames to the remote Node.
        This function should be implemented by subclasses to match
        the relative add_frames() method on the renderable class.
        """
        self._send_msg(Message.ADD_FRAMES, *args, **kwargs)

    def update_frames(self, *args, **kwargs):
        """
        Update frames of the remote Node.
        This function should be implemented by subclasses to match
        the relative add_frames() method on the renderable class.
        """
        self._send_msg(Message.UPDATE_FRAMES, *args, **kwargs)

    def remove_frames(self, frames):
        """
        Remove frames of the remote Node.

        :param frames: a list of integer frame indices of size N or a single integer frame index to remove.
        """
        self._send_msg(Message.REMOVE_FRAMES, frames=frames)

    def delete(self):
        """Delete the remote Node from the scene."""
        self._send_msg(Message.DELETE)

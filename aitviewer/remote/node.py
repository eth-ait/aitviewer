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
        self.viewer = viewer
        self.uid = next_id()
        self._send_msg(self.MESSAGE_TYPE, *args, **kwargs)

    def _send_msg(self, type, *args, **kwargs):
        self.viewer.send_message(type, self.uid, args, kwargs)

    def add_frames(self, *args, **kwargs):
        self._send_msg(Message.ADD_FRAMES, *args, **kwargs)

    def update_frames(self, *args, **kwargs):
        self._send_msg(Message.UPDATE_FRAMES, *args, **kwargs)

    def remove_frames(self, frames):
        self._send_msg(Message.REMOVE_FRAMES, frames=frames)

    def delete(self):
        self._send_msg(Message.DELETE)

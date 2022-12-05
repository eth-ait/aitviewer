from ..node import Node
from ..message import Message, make_message

class Meshes(Node):
    MESSAGE_TYPE = Message.MESH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def append(self, *args, **kwargs):
        if self.viewer is None:
            raise ValueError("Append can only be called after the node has been added to the scene")
        msg = make_message(Message.MESH_APPEND, self.uid, args, kwargs)
        self.viewer.send_msg(msg)
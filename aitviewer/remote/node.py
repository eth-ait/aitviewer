from .message import make_message, Message

GLOBAL_ID = 0

def next_id():
    global GLOBAL_ID
    GLOBAL_ID += 1
    return GLOBAL_ID

class Node:
    MESSAGE_TYPE=Message.NODE

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.add_kwargs = None
        self.uid = next_id()

        self.viewer = None

        self.nodes = []

    def add(self, node: 'Node', **kwargs):
        node.add_kwargs = kwargs
        self.nodes.append(node)

        if self.viewer is not None and node.viewer is None:
            node._add_node(self.viewer)

    def _add_node(self, viewer):
        self.viewer = viewer
        msg = make_message(self.MESSAGE_TYPE, self.uid, self.args, self.kwargs, self.add_kwargs)
        viewer.send_msg(msg)
        self._clear()
        for n in self.nodes:
            if n.viewer is None:
                n._add_node(viewer)

    def _clear(self):
        self.args = None
        self.kwargs = None
        self.add_kwargs = None

class Scene(Node):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
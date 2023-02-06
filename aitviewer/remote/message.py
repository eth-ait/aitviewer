import enum


class Message(enum.Enum):
    """Enumeration for the type of message."""

    # Messages used to create nodes on the remote viewer.
    NODE = 1
    MESHES = 2
    SPHERES = 3
    LINES = 4
    ARROWS = 5
    RIGID_BODIES = 6
    SMPL = 10

    # Messages used to modify existing nodes on the remote viewer.
    DELETE = 100
    ADD_FRAMES = 101
    UPDATE_FRAMES = 102
    REMOVE_FRAMES = 103

    # Built-in uitilities to interact with the viewer.
    SET_FRAME = 200
    NEXT_FRAME = 201
    PREVIOUS_FRAME = 202

    # All values greater than or equal to Message.USER_MESSAGE are not used internally by the viewer
    # and can be safely used to send custom messages.
    USER_MESSAGE = 10000


def make_message(type, uid, args, kwargs):
    msg = {
        "type": type,
        "uid": uid,
        "args": args,
        "kwargs": kwargs,
    }
    return msg

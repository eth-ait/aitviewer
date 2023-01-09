import enum

class Message(enum.Enum):
    NODE = 1
    MESH = 2

    DELETE = 100
    ADD_FRAMES = 101
    UPDATE_FRAMES = 102
    REMOVE_FRAMES = 103


def make_message(type, uid, args, kwargs):
    msg = {
        'type': type,
        'uid': uid,
        'args': args,
        'kwargs': kwargs,
    }
    return msg
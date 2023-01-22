import enum


class Message(enum.Enum):
    NODE = 1
    MESHES = 2
    SPHERES = 3
    LINES = 4
    ARROWS = 5
    RIGID_BODIES = 6
    SMPL = 10

    DELETE = 100
    ADD_FRAMES = 101
    UPDATE_FRAMES = 102
    REMOVE_FRAMES = 103

    SET_FRAME = 200
    NEXT_FRAME = 201
    PREVIOUS_FRAME = 202

    USER_MESSAGE = 10000


def make_message(type, uid, args, kwargs):
    msg = {
        "type": type,
        "uid": uid,
        "args": args,
        "kwargs": kwargs,
    }
    return msg

import enum

class Message(enum.Enum):
    NODE = 1
    MESH = 2
    MESH_APPEND = 101

def make_message(type, uid, args, kwargs, add_kwargs=None):
    msg = {
        'type': type,
        'uid': uid,
        'args': args,
        'kwargs': kwargs,
    }
    if add_kwargs:
        msg['add_kwargs'] = add_kwargs

    return msg
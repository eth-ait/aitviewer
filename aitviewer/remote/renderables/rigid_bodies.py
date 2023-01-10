from ..message import Message
from ..node import RemoteNode


class RemoteRigidBodies(RemoteNode):
    MESSAGE_TYPE = Message.RIGID_BODIES

    def __init__(self, viewer, rb_pos, rb_ori, **kwargs):
        super().__init__(
            viewer,
            rb_pos=rb_pos,
            rb_ori=rb_ori,
            **kwargs,
        )

    def add_frames(self, rb_pos, rb_ori):
        return super().add_frames(rb_pos=rb_pos, rb_ori=rb_ori)

    def update_frames(self, rb_pos, rb_ori, frames):
        return super().update_frames(rb_pos=rb_pos, rb_ori=rb_ori, frames=frames)

from ..message import Message
from ..node import RemoteNode


class RemoteRigidBodies(RemoteNode):
    MESSAGE_TYPE = Message.RIGID_BODIES

    def __init__(self, viewer, rb_pos, rb_ori, **kwargs):
        """
        This initializer takes a RemoteViewer object and all other arguments are forwarded
        to the RigidBodies constructor on the remote Viewer.
        See the RigidBodies class for more information about parameters.

        :param rb_pos: A np array of shape (N, R, 3) or (R, 3) rigid-body positions.
        :param rb_ori: A np array of shape (N, R, 3, 3) or (R, 3, 3) rigid-body orientations.
        """
        super().__init__(
            viewer,
            rb_pos=rb_pos,
            rb_ori=rb_ori,
            **kwargs,
        )

    def add_frames(self, rb_pos, rb_ori):
        """
        Add frames to the remote RigidBodies node by adding new positions and orientations.

        :param rb_pos: A np array of shape (N, R, 3) or (R, 3) rigid-body positions.
        :param rb_ori: A np array of shape (N, R, 3, 3) or (R, 3, 3) rigid-body orientations.
        """
        return super().add_frames(rb_pos=rb_pos, rb_ori=rb_ori)

    def update_frames(self, rb_pos, rb_ori, frames):
        """
        Update frames of the remote RigidBodies node by updating the positions and orientations.

        :param rb_pos: A np array of shape (N, R, 3) or (R, 3) rigid-body positions.
        :param rb_ori: A np array of shape (N, R, 3, 3) or (R, 3, 3) rigid-body orientations.
        :param frames: a list of integer frame indices of size N or a single integer frame index.
        """
        return super().update_frames(rb_pos=rb_pos, rb_ori=rb_ori, frames=frames)

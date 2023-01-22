from ..message import Message
from ..node import RemoteNode


class RemoteSMPLSequence(RemoteNode):
    MESSAGE_TYPE = Message.SMPL

    def __init__(
        self,
        viewer,
        poses_body,
        **kwargs,
    ):
        super().__init__(
            viewer,
            poses_body=poses_body,
            **kwargs,
        )

    def add_frames(self, poses_body, poses_root=None, trans=None, betas=None):
        return super().add_frames(poses_body=poses_body, poses_root=poses_root, trans=trans, betas=betas)

    def update_frames(self, poses_body, frames, poses_root=None, trans=None, betas=None):
        return super().update_frames(
            poses_body=poses_body,
            frames=frames,
            poses_root=poses_root,
            trans=trans,
            betas=betas,
        )

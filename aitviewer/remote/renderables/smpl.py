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

    def add_frames(self, poses_body, betas):
        return super().add_frames(poses_body=poses_body, betas=betas)

    def update_frames(self, poses_body, betas):
        return super().update_frames(poses_body=poses_body, betas=betas)

from ..message import Message
from ..node import RemoteNode


class RemoteArrows(RemoteNode):
    MESSAGE_TYPE = Message.ARROWS

    def __init__(self, viewer, origins, tips, **kwargs):
        super().__init__(
            viewer,
            origins=origins,
            tips=tips,
            **kwargs,
        )

    def add_frames(self, origins, tips):
        return super().add_frames(origins=origins, tips=tips)

    def update_frames(self, origins, tips, frames):
        return super().update_frames(origins=origins, tips=tips, frames=frames)

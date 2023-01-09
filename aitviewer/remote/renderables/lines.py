from ..message import Message
from ..node import RemoteNode


class RemoteLines(RemoteNode):
    MESSAGE_TYPE = Message.LINES

    def __init__(self, viewer, lines, **kwargs):
        super().__init__(
            viewer,
            lines=lines,
            **kwargs,
        )

    def add_frames(self, lines):
        return super().add_frames(lines=lines)

    def update_frames(self, lines, frames):
        return super().update_frames(lines=lines, frames=frames)

from ..message import Message
from ..node import RemoteNode


class RemoteSpheres(RemoteNode):
    MESSAGE_TYPE = Message.SPHERES

    def __init__(self, viewer, positions, **kwargs):
        super().__init__(
            viewer,
            positions=positions,
            **kwargs,
        )

    def add_frames(self, positions):
        return super().add_frames(positions=positions)

    def update_frames(self, positions, frames):
        return super().update_frames(positions=positions, frames=frames)

    def remove_frames(self, frames):
        return super().remove_frames(frames=frames)

    def delete(self):
        return super().delete()

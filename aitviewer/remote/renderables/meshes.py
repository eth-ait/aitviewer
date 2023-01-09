from ..message import Message, make_message
from ..node import RemoteNode


class RemoteMeshes(RemoteNode):
    MESSAGE_TYPE = Message.MESH

    def __init__(self, viewer, vertices, faces, **kwargs):
        super().__init__(
            viewer,
            vertices=vertices,
            faces=faces,
            **kwargs,
        )

    def add_frames(self, vertices):
        return super().add_frames(vertices=vertices)

    def update_frames(self, vertices, frames):
        return super().update_frames(vertices=vertices, frames=frames)

    def remove_frames(self, frames):
        return super().remove_frames(frames=frames)

    def delete(self):
        return super().delete()

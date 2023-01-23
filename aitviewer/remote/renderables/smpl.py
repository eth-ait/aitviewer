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
        """
        This initializer takes a RemoteViewer object and all other arguments are forwarded
        to the SMPLLayer or SMPLSequence classes.
        The keyword arguments "model_type", "gender" and "num_betas" are forwarded to SMPLLayer,
        while the remaining arguments are forwarded to SMPLSequence.

        :param viewer: a RemoteViewer object that will be used to send this node.
        :param poses_body: An np array of shape (N, N_JOINTS*3) containing the pose parameters of the
          body, i.e. without hands or face parameters.
        """
        super().__init__(
            viewer,
            poses_body=poses_body,
            **kwargs,
        )

    def add_frames(self, poses_body, poses_root=None, trans=None, betas=None):
        """
        Add frames to the remote SMPLSequence node by adding body poses.

        :param poses_body: An np array of shape (N, N_JOINTS*3) or (N_JOINTS) containing the
          pose parameters of the body, i.e. without hands or face parameters.
        :param poses_root: An optional np array of shape (N, 3) or (3) containing the global root orientation.
        :param betas: An optional np array of shape (N, N_BETAS) or (N_BETAS) containing the shape parameters.
        :param trans: An optional np array of shape (N, 3) or (3) containing a global translation that is
          applied to all joints and vertices.

        """
        return super().add_frames(poses_body=poses_body, poses_root=poses_root, trans=trans, betas=betas)

    def update_frames(self, poses_body, frames, poses_root=None, trans=None, betas=None):
        """
        Update frames of the remote SMPLSequence node by updating body poses.

        :param poses_body: An np array of shape (N, N_JOINTS*3) or (N_JOINTS) containing the
          pose parameters of the body, i.e. without hands or face parameters.
        :param poses_root: An optional np array of shape (N, 3) or (3) containing the global root orientation.
        :param betas: An optional np array of shape (N, N_BETAS) or (N_BETAS) containing the shape parameters.
        :param trans: An optional np array of shape (N, 3) or (3) containing a global translation that is
          applied to all joints and vertices.
        """

        return super().update_frames(
            poses_body=poses_body,
            frames=frames,
            poses_root=poses_root,
            trans=trans,
            betas=betas,
        )

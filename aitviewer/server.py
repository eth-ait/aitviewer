import queue
import threading

from aitviewer.models.smpl import SMPLLayer
from aitviewer.remote.message import Message
from aitviewer.renderables.arrows import Arrows
from aitviewer.renderables.lines import Lines
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.rigid_bodies import RigidBodies
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.spheres import Spheres
from aitviewer.scene.node import Node


class ViewerServer:
    def __init__(self, viewer, port):
        """
        Initializer.
        :param viewer: the viewer for which the server processes messages.
        :param port: the TCP port number to which the websocket for listening to incoming
          connections will be bound.
        """
        self.viewer = viewer

        # Dictionary mapping from the remote node uid to the local node uid
        self.remote_to_local_id = {}

        # A queue of messages received by the server. This is used to transfer
        # received messages from the server thread to the main thread.
        self.queue = queue.Queue()

        # A list of connections that are currently open. Each entry is
        # the remote address (host, port) tuple.
        self.connections = []

        # Entry point of server thread
        def entry(queue: queue.Queue):
            import asyncio
            import pickle

            import websockets

            # Called whenever a new connection is enstablished.
            async def serve(websocket):
                addr = websocket.remote_address
                print(f"New connection: {addr[0]}:{addr[1]}")
                self.connections.append(addr)
                try:
                    # Message loop.
                    # This loop is exited whenever the connection is dropped
                    # which causes and exception to be raised.
                    async for message in websocket:
                        data = pickle.loads(message)
                        # Equeue data for the main thread to process.
                        queue.put_nowait(data)
                except:
                    pass
                self.connections.remove(addr)
                print(f"Connection closed: {addr[0]}:{addr[1]}")

            # Async entry point of the main thread.
            async def main():
                server = await websockets.serve(serve, "0.0.0.0", port)
                await server.serve_forever()

            asyncio.run(main())

        # daemon = true means that the thread is abruptly stopped once the main thread exits.
        self.thread = threading.Thread(target=entry, args=(self.queue,), daemon=True)
        self.thread.start()

    def process_messages(self):
        """Processes all messages received since the last time this method was called."""
        while not self.queue.empty():
            msg = self.queue.get_nowait()
            # Call process_message on the viewer so that subclasses of a viewer can intercept messages.
            # By default this will end up calling self.process_message()
            self.viewer.process_message(msg["type"], msg["uid"], msg["args"], msg["kwargs"])

    def process_message(self, type: Message, remote_uid: int, args, kwargs):
        """
        Default processing of messages.

        :param type: the type of the message.
        :param remote_uid: the remote id of the node that this message refers to.
        :param args: positional arguments received with the message.
        :param kwargs: keyword arguments received with the message.
        """

        def add(remote_uid, args, kwargs, type):
            n = type(*args, **kwargs)
            self.viewer.scene.add(n)
            self.remote_to_local_id[remote_uid] = n.uid

        if type == Message.NODE:
            add(remote_uid, args, kwargs, Node)

        elif type == Message.MESHES:
            add(remote_uid, args, kwargs, Meshes)

        elif type == Message.SPHERES:
            add(remote_uid, args, kwargs, Spheres)

        elif type == Message.LINES:
            add(remote_uid, args, kwargs, Lines)

        elif type == Message.ARROWS:
            add(remote_uid, args, kwargs, Arrows)

        elif type == Message.RIGID_BODIES:
            add(remote_uid, args, kwargs, RigidBodies)

        elif type == Message.SMPL:
            layer_arg_names = {"model_type", "gender", "num_betas"}
            layer_kwargs = {k: v for k, v in kwargs.items() if k in layer_arg_names}
            layer = SMPLLayer(**layer_kwargs)

            sequence_kwargs = {k: v for k, v in kwargs.items() if k not in layer_arg_names}
            n = SMPLSequence(*args, smpl_layer=layer, **sequence_kwargs)
            self.viewer.scene.add(n)
            self.remote_to_local_id[remote_uid] = n.uid

        elif type == Message.DELETE:
            node: Node = self.get_node_by_remote_uid(remote_uid)
            if node and node.parent:
                node.parent.remove(node)

        elif type == Message.UPDATE_FRAMES:
            node: Node = self.get_node_by_remote_uid(remote_uid)
            if node:
                node.update_frames(*args, **kwargs)

        elif type == Message.ADD_FRAMES:
            node: Node = self.get_node_by_remote_uid(remote_uid)
            if node:
                node.add_frames(*args, **kwargs)

        elif type == Message.REMOVE_FRAMES:
            node: Node = self.get_node_by_remote_uid(remote_uid)
            if node:
                node.remove_frames(*args, **kwargs)

        elif type == Message.SET_FRAME:
            if not self.viewer.run_animations:
                self.viewer.scene.current_frame_id = args[0]

        elif type == Message.NEXT_FRAME:
            if not self.viewer.run_animations:
                self.viewer.scene.next_frame()

        elif type == Message.PREVIOUS_FRAME:
            if not self.viewer.run_animations:
                self.viewer.scene.previous_frame()

    def get_node_by_remote_uid(self, remote_uid):
        """
        Returns the Node corresponding to the remote uid passed in.

        :param remote_uid: the remote uid to look up.
        :return: Node corresponding to the remote uid.
        """
        return self.viewer.scene.get_node_by_uid(self.remote_to_local_id.get(remote_uid, None))


# If this module is invoke directly it starts an empty viewer
# with the server functionality enabled.
if __name__ == "__main__":
    from aitviewer.viewer import Viewer

    v = Viewer(config={"server_enabled": True})
    v.scene.floor.enabled = False
    v.run()

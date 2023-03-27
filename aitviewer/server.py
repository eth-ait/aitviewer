import asyncio
import pickle
import queue
import threading
from typing import Dict, Tuple

import websockets

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

        # A map of connections that are currently open. Each entry maps
        # the remote address (host, port) to a websocket.
        self.connections: Dict[Tuple[str, str], websockets.server.WebSocketServerProtocol] = {}

        # Entry point of server thread
        def entry():
            # Called whenever a new connection is enstablished.
            async def serve(websocket):
                addr = websocket.remote_address
                self.connections[addr] = websocket

                print(f"New connection: {addr[0]}:{addr[1]}")
                try:
                    # Message loop.
                    # This loop is exited whenever the connection is dropped
                    # which causes and exception to be raised.
                    async for message in websocket:
                        data = pickle.loads(message)
                        # Equeue data for the main thread to process.
                        self.queue.put_nowait((addr, data))
                    await websocket.close()
                except Exception as e:
                    print(f"Except {e}")
                del self.connections[addr]
                print(f"Connection closed: {addr[0]}:{addr[1]}")

            # Async entry point of the main thread.
            async def main():
                async with websockets.serve(serve, "0.0.0.0", port, max_size=None):
                    await self.stop

            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(main())

        self.loop = asyncio.new_event_loop()
        self.stop = self.loop.create_future()
        # daemon = true means that the thread is abruptly stopped once the main thread exits.
        self.thread = threading.Thread(target=entry, daemon=True)
        self.thread.start()

    def process_messages(self):
        """Processes all messages received since the last time this method was called."""
        while not self.queue.empty():
            client, msg = self.queue.get_nowait()
            # Call process_message on the viewer so that subclasses of a viewer can intercept messages.
            # By default this will end up calling self.process_message()
            self.viewer.process_message(msg["type"], msg["uid"], msg["args"], msg["kwargs"], client)

    def process_message(self, type: Message, remote_uid: int, args: list, kwargs: dict, client: Tuple[str, str]):
        """
        Default processing of messages.

        :param type: the type of the message.
        :param remote_uid: the remote id of the node that this message refers to.
        :param args: positional arguments received with the message.
        :param kwargs: keyword arguments received with the message.
        :param client: a tuple (ip, port) describing the address of the client
            that sent this message.
        """

        def add(client, remote_uid, args, kwargs, type):
            n = type(*args, **kwargs)
            self.viewer.scene.add(n)
            self.remote_to_local_id[(client, remote_uid)] = n.uid

        if type == Message.NODE:
            add(client, remote_uid, args, kwargs, Node)

        elif type == Message.MESHES:
            add(client, remote_uid, args, kwargs, Meshes)

        elif type == Message.SPHERES:
            add(client, remote_uid, args, kwargs, Spheres)

        elif type == Message.LINES:
            add(client, remote_uid, args, kwargs, Lines)

        elif type == Message.ARROWS:
            add(client, remote_uid, args, kwargs, Arrows)

        elif type == Message.RIGID_BODIES:
            add(client, remote_uid, args, kwargs, RigidBodies)

        elif type == Message.SMPL:
            layer_arg_names = {"model_type", "gender", "num_betas"}
            layer_kwargs = {k: v for k, v in kwargs.items() if k in layer_arg_names}
            layer = SMPLLayer(**layer_kwargs)

            sequence_kwargs = {k: v for k, v in kwargs.items() if k not in layer_arg_names}
            n = SMPLSequence(*args, smpl_layer=layer, **sequence_kwargs)
            self.viewer.scene.add(n)
            self.remote_to_local_id[(client, remote_uid)] = n.uid

        elif type == Message.DELETE:
            node: Node = self.get_node_by_remote_uid(remote_uid, client)
            if node and node.parent:
                node.parent.remove(node)

        elif type == Message.UPDATE_FRAMES:
            node: Node = self.get_node_by_remote_uid(remote_uid, client)
            if node:
                node.update_frames(*args, **kwargs)

        elif type == Message.ADD_FRAMES:
            node: Node = self.get_node_by_remote_uid(remote_uid, client)
            if node:
                node.add_frames(*args, **kwargs)

        elif type == Message.REMOVE_FRAMES:
            node: Node = self.get_node_by_remote_uid(remote_uid, client)
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

    async def _async_send(self, msg, client):
        data = pickle.dumps(msg)
        if client is None:
            websockets.broadcast(self.connections.values(), data)
        else:
            websocket = self.connections.get(client)
            if websocket is not None:
                await websocket.send(msg)

    def send_message(self, msg, client: Tuple[str, str] = None):
        """
        Send a message to a single client or to all connected clients.

        :param msg: a python object that is serialized with pickle and sent to the client.
        :param client: a tuple (host, port) representing the client to which to send the message,
            if None the message is sent to all connected clients.
        """
        # Send a message by adding a send coroutine to the thread's loop and wait for it to complete.
        asyncio.run_coroutine_threadsafe(self._async_send(msg, client), self.loop).result()

    def get_node_by_remote_uid(self, remote_uid: int, client: Tuple[str, str]):
        """
        Returns the Node corresponding to the remote uid and client passed in.

        :param remote_uid: the remote uid to look up.
        :param client: the client that created the node, this is the value of the 'client'
            parameter that was passed to process_message() when the message was received.
        :return: Node corresponding to the remote uid.
        """
        return self.viewer.scene.get_node_by_uid(self.remote_to_local_id.get((client, remote_uid), None))

    def close(self):
        self.loop.call_soon_threadsafe(self.stop.set_result, None)
        self.thread.join()


# If this module is invoke directly it starts an empty viewer
# with the server functionality enabled.
if __name__ == "__main__":
    from aitviewer.configuration import CONFIG as C
    from aitviewer.viewer import Viewer

    C.update_conf({"server_enabled": True})
    v = Viewer()
    v.scene.floor.enabled = False
    v.run()

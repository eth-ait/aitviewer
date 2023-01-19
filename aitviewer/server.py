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
        self.viewer = viewer
        self.remote_to_local_id = {}
        self.queue = queue.Queue()

        # Entry point of server thread
        def entry(queue: queue.Queue):
            import asyncio
            import pickle

            import websockets

            async def serve(websocket):
                addr = websocket.remote_address
                print(f"New connection: {addr[0]}:{addr[1]}")
                try:
                    async for message in websocket:
                        data = pickle.loads(message)
                        queue.put_nowait(data)
                except:
                    pass
                print(f"Connection closed: {addr[0]}:{addr[1]}")

            async def main():
                server = await websockets.serve(serve, "0.0.0.0", port)
                await server.serve_forever()

            asyncio.run(main())

        # daemon = true means that the thread is abruptly stopped once the main thread exits.
        self.thread = threading.Thread(target=entry, args=(self.queue,), daemon=True)
        self.thread.start()

    def process_messages(self):
        while not self.queue.empty():
            msg = self.queue.get_nowait()
            # Call process_message on the viewer so that subclasses of a viewer can intercept messages.
            # By default this will end up calling self.process_message()
            self.viewer.process_message(
                msg["type"], msg["uid"], msg["args"], msg["kwargs"]
            )

    def process_message(self, type: Message, remote_uid: int, args, kwargs):
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

            sequence_kwargs = {
                k: v for k, v in kwargs.items() if k not in layer_arg_names
            }
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

    def get_node_by_remote_uid(self, remote_uid):
        return self.viewer.scene.get_node_by_uid(self.remote_to_local_id[remote_uid])


if __name__ == "__main__":
    from aitviewer.viewer import Viewer

    v = Viewer(config={"server_enabled": True})
    v.scene.floor.enabled = False
    v.run()
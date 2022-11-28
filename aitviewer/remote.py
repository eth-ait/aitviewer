import asyncio
import websockets
import threading
import subprocess
import pickle
import enum

class Message(enum.Enum):
    MESH = 1

class RemoteViewer:
    def __init__(self, host=None):
        if host is None:
            # self.p = subprocess.Popen(["python", "-um", "aitviewer.viewer"], creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP, stdout=subprocess.PIPE)
            self.p = subprocess.Popen(["python", "-um", "aitviewer.viewer"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for l in self.p.stdout:
                if "OK" in l.decode():
                    break
            host = "localhost"
        else:
            self.p = None

        url = f"ws://{host}:8765"
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._entry, args=(url,), daemon=True)
        self.thread.start()

    def _entry(self, url):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._async_entry(url))
        self.loop.close()

    async def _async_entry(self, url):
        self.queue = asyncio.Queue()
        try:
            self.websocket = await websockets.connect(url)
            while True:
                data = await self.queue.get()
                if data is None:
                    await self.websocket.close()
                    break
                await self.websocket.send(data)
        except Exception as e:
            print(e)

    async def _async_send(self, data):
        await self.queue.put(data)

    def send(self, data):
        try:
            asyncio.run_coroutine_threadsafe(self._async_send(data), self.loop).result()
        except Exception as e:
            print(e)

    def close(self):
        self.send(None)
        self.thread.join()
        if self.p is not None:
            self.p.wait()

    def message(self, type, **kwargs):
        return {
            'type': type,
            'data': kwargs,
        }

    def mesh(self, vertices, faces, flat_shading=False, **kwargs):
        msg = self.message(Message.MESH, vertices=vertices, faces=faces, flat_shading=flat_shading, **kwargs)
        data = pickle.dumps(msg)
        self.send(data)
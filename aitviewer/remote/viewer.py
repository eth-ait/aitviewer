import asyncio
import pickle
import subprocess
import threading

import websockets


class RemoteViewer:
    def __init__(self, host=None, port=8417, timeout=10):
        if host is None:
            self.p = subprocess.Popen(
                ["python", "-um", "aitviewer.viewer"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            host = "localhost"
        else:
            self.p = None

        url = f"ws://{host}:{port}"

        self.timeout = timeout
        self.connected = False
        self.semaphore = threading.Semaphore(0)

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._entry, args=(url,), daemon=True)
        self.thread.start()

        # Wait for the connection to be setup.
        self.semaphore.acquire()
        if not self.connected:
            print(f"Failed to connect to remote viewer at {url}")

    def _entry(self, url):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._async_entry(url))

    async def _async_entry(self, url):
        self.queue = asyncio.Queue()

        # Attempt to connect until 'self.timeout' seconds passed.
        start_time = self.loop.time()
        while self.loop.time() < start_time + self.timeout:
            try:
                self.websocket = await websockets.connect(url)
                self.connected = True
                break
            except:
                pass

        # Notify main thread of connection status.
        self.semaphore.release()
        if not self.connected:
            return

        # Message loop.
        try:
            while True:
                data = await self.queue.get()
                if data is None:
                    await self.websocket.close()
                    break
                await self.websocket.send(data)
        except Exception as e:
            if not self.connected:
                self.semaphore.release()
            self.connected = False
            print(f"Message loop exception: {e}")

    async def _async_send(self, data):
        await self.queue.put(data)

    def send(self, data):
        try:
            if self.connected:
                asyncio.run_coroutine_threadsafe(
                    self._async_send(data), self.loop
                ).result()
        except Exception as e:
            print(f"Send loop exception: {e}")

    def send_msg(self, msg):
        data = pickle.dumps(msg)
        self.send(data)

    def wait_close(self, print_viewer_output=True):
        self.send(None)
        self.thread.join()
        if self.p is not None:
            self.p.wait()
            if print_viewer_output:
                print("\nRemote viewer output:")
                print(self.p.stdout.read().decode())

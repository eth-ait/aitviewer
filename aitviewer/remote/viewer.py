import asyncio
import pickle
import subprocess
import threading

import websockets

from .message import Message, make_message


class RemoteViewer:
    def __init__(self, host=None, port=8417, timeout=10, args=None):
        """
        Initializer.
        :param host: the ip address of a host to connect to as a string or None.
          if None, an empty viewer is created in a new process on the local host.
        :param port: the TCP port to connect to.
        :param timeout: a timeout in seconds for attempting to connect to the viewer.
        :param args: if host is None this parameter can be used to specify an argument or
          a list of arguments that is used to create the local viewer process.
          e.g: args = ["path/to/script.py", "arg1", "arg2"] will invoke the following command:
                python path/to/script.py arg1 arg2
        """
        if host is None:
            if args is None:
                popen_args = ["python", "-m", "aitviewer.server"]
            else:
                if isinstance(args, list):
                    popen_args = ["python"] + args
                else:
                    popen_args = ["python", str(args)]
            self.p = subprocess.Popen(
                popen_args,
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
                asyncio.run_coroutine_threadsafe(self._async_send(data), self.loop).result()
        except Exception as e:
            print(f"Send loop exception: {e}")

    def send_message(self, type, uid=None, args=[], kwargs={}):
        """
        Send a message to the viewer. See Viewer.process_message()
        for information about how these parameters are interpreted
        by the viewer.
        """
        msg = make_message(type, uid, args, kwargs)
        data = pickle.dumps(msg)
        self.send(data)

    def set_frame(self, frame: int):
        """
        Set the current active frame of the remote viewer.

        :param frame: an integer representing the id of the frame.
        """
        self.send_message(Message.SET_FRAME, None, [frame])

    def next_frame(self):
        """Set the current active frame of the remote viewer to the next frame"""
        self.send_message(Message.NEXT_FRAME)

    def previous_frame(self):
        """Set the current active frame of the remote viewer to the previous frame"""
        self.send_message(Message.PREVIOUS_FRAME)

    def wait_close(self, print_viewer_output=True):
        """
        If the viewer was created locally in a separate process wait for it
        to exit and optionally print the standard output of the remote viewer.

        :param print_viewer_output: if True print the output of the remote viewer.
        """
        self.send(None)
        self.thread.join()
        if self.p is not None:
            self.p.wait()
            if print_viewer_output:
                print("\nRemote viewer output:")
                print(self.p.stdout.read().decode())

import asyncio
import pickle
import subprocess
import threading

import websockets

from .message import Message, make_message


class RemoteViewer:
    def __init__(self, host="localhost", port=8417, timeout=10, verbose=True):
        """
        Initializer.
        :param host: the IP address of a host to connect to as a string.
        :param port: the TCP port to connect to.
        :param timeout: a timeout in seconds for attempting to connect to the viewer.
        :param verbose: if True print info messages.
        """
        url = f"ws://{host}:{port}"

        if verbose:
            print(f"Connecting to remote viewer at {url}")

        self.timeout = timeout
        self.connected = False

        # Semaphore used to wait for the connection to be setup by the client thread.
        self.semaphore = threading.Semaphore(0)

        # Create a thread for running the websocket client async loop.
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._entry, args=(url,), daemon=True)
        self.thread.start()

        # Wait for the connection to be setup.
        self.semaphore.acquire()
        if verbose:
            if self.connected:
                print("Connected")
            else:
                print(f"Failed to connect")

        self.process: subprocess.Popen = None

    @classmethod
    def create_new_process(cls, args=None, **kwargs):
        """
        Open a Viewer in a new process and return a RemoteViewer connected to it.

        :param args: This parameter can be used to specify an argument or
          a list of arguments that is used to create the new process.
          e.g: args = ["path/to/script.py", "arg1", "arg2"] will invoke the following command:
                python path/to/script.py arg1 arg2
        """
        # If host is None create a new viewer in a separate process.
        if args is None:
            popen_args = ["python", "-m", "aitviewer.server"]
        else:
            if isinstance(args, list):
                popen_args = ["python"] + args
            else:
                popen_args = ["python", str(args)]

        # Create the viewer process.
        process = subprocess.Popen(
            popen_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Create a remote viewer connected to the child process.
        v = cls(**kwargs)
        v.process = process

        return v

    def _entry(self, url):
        # Entry point of the client thread.

        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._async_entry(url))

    async def _async_entry(self, url):
        # Async entry point of the client thread.

        # Create an async queue for communicating with the main thread.
        self.queue = asyncio.Queue()

        # Attempt to connect until 'self.timeout' seconds passed.
        start_time = self.loop.time()
        try:
            while self.loop.time() < start_time + self.timeout:
                try:
                    self.websocket = await websockets.connect(url)
                    self.connected = True
                    break
                except Exception as e:
                    pass
        finally:
            # Release the semaphore to let the main thread continue after
            # attempting to connect. The main thread will read the
            # self.connected variable to know if we succeded at connecting.
            self.semaphore.release()

        # Exit the client thread if we failed to connect.
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
            print(f"Message loop exception: {e}")

        # Mark the connection as closed.
        self.connected = False

    def __del__(self):
        # Gracefully close the connection when the viewer is deleted.
        self.close_connection()

    async def _async_send(self, data):
        # Append to the mesage queue.
        await self.queue.put(data)

    def send(self, data):
        try:
            if self.connected:
                # Append a message to the client thread queue by adding a send coroutine to the
                # thread's loop and wait for it to complete.
                asyncio.run_coroutine_threadsafe(self._async_send(data), self.loop).result()
        except Exception as e:
            print(f"Send exception: {e}")

    def send_message(self, type, uid=None, *args, **kwargs):
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
        self.send_message(Message.SET_FRAME, None, frame)

    def next_frame(self):
        """Set the current active frame of the remote viewer to the next frame"""
        self.send_message(Message.NEXT_FRAME)

    def previous_frame(self):
        """Set the current active frame of the remote viewer to the previous frame"""
        self.send_message(Message.PREVIOUS_FRAME)

    def close_connection(self):
        """Close the connection with the remote viewer."""
        if self.connected:
            # Send a special None message to signal the client thread to close the connection.
            self.send(None)

            # Wait for the client thread to exit.
            self.thread.join()

    def wait_process(self, print_viewer_output=True):
        """
        If the viewer was created locally in a separate process wait for it
        to exit and optionally print the standard output of the remote viewer.

        :param print_viewer_output: if True print the output of the remote viewer.
        """
        self.close_connection()
        if self.process is not None:
            self.process.wait()
            if print_viewer_output:
                print("\nRemote viewer output:")
                print(self.process.stdout.read().decode())

# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
from aitviewer.scene.node import Node


class Streamable(Node):
    """Interface for renderables."""

    def __init__(self, **kwargs):
        super(Streamable, self).__init__(**kwargs)

        self.is_recording = False

    def start(self):
        pass

    def stop(self):
        pass

    def capture(self):
        """Capture from the sensor"""
        raise NotImplementedError("Must be implemented by the subclass.")

    def record_start(self):
        self.is_recording = True

    def record_capture(self):
        pass

    def record_finish(self):
        self.is_recording = False
        return []

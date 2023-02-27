"""
Copyright (C) 2022  ETH Zurich, Manuel Kaufmann, Velko Vechev

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
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

"""
Copyright (C) 2022  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos

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
import time
from typing import Tuple

from moderngl_window.timers.base import BaseTimer

"""
Timer class adapted from moderngl_window.timers.clock.Timer.
Uses time.perf_counter() instead of time.time() for higher resolution timestamps.
"""


class PerfTimer(BaseTimer):
    """Timer based on python ``time``."""

    def __init__(self, **kwargs):
        self._start_time = None
        self._stop_time = None
        self._pause_time = None
        self._last_frame = None
        self._offset = 0

    @property
    def is_paused(self) -> bool:
        """bool: The pause state of the timer"""
        return self._pause_time is not None

    @property
    def is_running(self) -> bool:
        """bool: Is the timer currently running?"""
        return self._pause_time is None

    @property
    def time(self) -> float:
        """Get or set the current time.
        This can be used to jump around in the timeline.

        Returns:
            The current time in seconds
        """
        if self._start_time is None:
            return 0.0

        if self.is_paused:
            return self._pause_time - self._offset - self._start_time

        return time.perf_counter() - self._start_time - self._offset

    @time.setter
    def time(self, value: float):
        if value < 0:
            value = 0

        self._offset += self.time - value

    def next_frame(self) -> Tuple[float, float]:
        """
        Get the time and frametime for the next frame.
        This should only be called once per frame.

        Returns:
            Tuple[float, float]: current time and frametime
        """
        current = self.time
        delta, self._last_frame = current - self._last_frame, current
        return current, delta

    def start(self):
        """Start the timer by recoding the current ``time.perf_counter()``
        preparing to report the number of seconds since this timestamp.
        """
        if self._start_time is None or self._pause_time is None:
            self._start_time = time.perf_counter()
            self._last_frame = 0.0
        else:
            self._offset += time.perf_counter() - self._pause_time
            self._pause_time = None

    def pause(self):
        """Pause the timer by setting the internal pause time using ``time.perf_counter()``"""
        self._pause_time = time.perf_counter()

    def toggle_pause(self):
        """Toggle the paused state"""
        if self.is_paused:
            self.start()
        else:
            self.pause()

    def stop(self) -> Tuple[float, float]:
        """
        Stop the timer. Should only be called once when stopping the timer.

        Returns:
            Tuple[float, float]: Current position in the timer, actual running duration
        """
        self._stop_time = time.perf_counter()
        return (
            self._stop_time - self._start_time - self._offset,
            self._stop_time - self._start_time,
        )

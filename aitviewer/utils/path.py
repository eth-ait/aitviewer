
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
import numpy as np


def line(start, end, num):
    """
    Returns an array of positions on a straight line (num, 3).
    :param start: start position, np array-like of shape (3).
    :param end: end position, np array-like of shape (3).
    :param num: number of positions in the returned array.
    """
    return np.linspace(np.array(start), np.array(end), num=num)


def circle(center, radius, num, start_angle=0.0, end_angle=360.0):
    """
    Returns an array of positions on a circle (num_frames, 3).
    :param center: center position of the circle, np array-like of shape (3).
    :param radius: radius of the circle, float.
    :param num: number of positions in the returned array.
    :param start_angle: starting angle on the circle in degrees.
    :param end_angle: ending angle on the circle in degrees.
    """
    angles = np.linspace(np.radians(start_angle), np.radians(end_angle), num=num)
    c = np.column_stack((np.cos(angles) * radius, np.zeros(angles.shape), np.sin(angles) * radius))
    return c + center

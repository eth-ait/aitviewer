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
import scipy.ndimage

from aitviewer.scene.node import Node


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


def lock_to_node(node: Node, relative_position, smooth_sigma=None):
    """
    Returns a tuple (positions, targets) of two arrays of shape (N, 3) each where N is the
    number of frames of the node. Each target is computed as the center position of
    the node at each frame and each position is at a constant offset from the target by the amount
    specified in the parameter 'relative_position'. The two arrays can be smoothed
    by passing a value greater than 0 to smooth_sigma.

    :param node: the Node to follow.
    :param relative_position: A position added to the node position to compute the positions array.
    :param smooth_sigma: if not None and greater than 0 the position and target arrays are smoothed.
      with a 1D gaussian kernel of standard deviation equal to this value.
    """
    assert isinstance(node, Node), "Node parameter must be a node"

    relative_position = np.array(relative_position)
    old_current_frame_id = node.current_frame_id

    positions = np.zeros((node.n_frames, 3))
    targets = np.zeros((node.n_frames, 3))
    for i in range(node.n_frames):
        node.current_frame_id = i
        target = node.current_center
        position = target + relative_position
        positions[i] = position
        targets[i] = target

    node.current_frame_id = old_current_frame_id

    if smooth_sigma is not None and smooth_sigma > 0:
        positions = scipy.ndimage.gaussian_filter1d(positions, smooth_sigma, axis=0, mode="nearest")
        targets = scipy.ndimage.gaussian_filter1d(targets, smooth_sigma, axis=0, mode="nearest")

    return positions, targets

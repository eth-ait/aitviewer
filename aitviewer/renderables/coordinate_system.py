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
import numpy as np

from aitviewer.scene.node import Node
from aitviewer.renderables.rigid_bodies import RigidBodies


class CoordinateSystem(Node):
    """
    Render a coordinate system using shaded cylinders.
    """

    def __init__(self,
                 length=1.0,
                 icon="\u0086",
                 **kwargs):
        super(CoordinateSystem, self).__init__(icon=icon, **kwargs)

        r = length / 50
        l = length

        self.rb = RigidBodies(np.array([[[0.0, 0.0, 0.0]]]), np.eye(3)[np.newaxis, np.newaxis], radius=r, length=l)
        self.add(self.rb)

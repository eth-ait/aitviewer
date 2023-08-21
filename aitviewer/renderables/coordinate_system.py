# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import numpy as np

from aitviewer.renderables.rigid_bodies import RigidBodies


class CoordinateSystem(RigidBodies):
    """
    Render a coordinate system using shaded cylinders.
    """

    def __init__(self, length=1.0, icon="\u008a", **kwargs):
        r = length / 50
        l = length
        super(CoordinateSystem, self).__init__(
            np.array([[[0.0, 0.0, 0.0]]]),
            np.eye(3)[np.newaxis, np.newaxis],
            radius=r,
            length=l,
            icon=icon,
            **kwargs,
        )

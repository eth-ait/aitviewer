# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos


class Material(object):
    """Per object material properties."""

    def __init__(
        self,
        diffuse=0.5,
        ambient=0.5,
        specular=0.5,
        color=(0.5, 0.5, 0.5, 1.0),
    ):
        """
        :param diffuse: diffuse coefficient in Phong shading model
        :param ambient: ambient coefficient in Phong shading model
        :param specular: specular coefficient in Phong shading model
        :param color: (R,G,B,A) 0-1 formatted color value
        """
        assert len(color) == 4

        self.diffuse = diffuse
        self.ambient = ambient
        self.specular = specular
        self._color = color

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color):
        self._color = color

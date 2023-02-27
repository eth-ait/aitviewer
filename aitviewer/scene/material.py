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

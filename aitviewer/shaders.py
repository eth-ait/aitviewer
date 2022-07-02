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
from moderngl_window import resources
from moderngl_window.meta import ProgramDescription


def _load(name):
    return resources.programs.load(ProgramDescription(path=name))


def get_simple_unlit_program():
    return _load('simple_unlit.glsl')


def get_smooth_lit_with_edges_program():
    return _load('smooth_lit_with_edges.glsl')


def get_flat_lit_with_edges_program():
    return _load('flat_lit_with_edges.glsl')


def get_cylinder_program():
    return _load('cylinder.glsl')


def get_screen_texture_program():
    return _load('screen_texture.glsl')


def get_smooth_lit_texturized_program():
    return _load('smooth_lit_tex.glsl')

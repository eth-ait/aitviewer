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

import functools

def _load(name, defines={}):
    return resources.programs.load(ProgramDescription(path=name, defines=defines))


@functools.lru_cache()
def get_smooth_lit_with_edges_program():
    return _load('lit_with_edges.glsl', defines={ 'SMOOTH_SHADING': 1, 'TEXTURE': 0 })


@functools.lru_cache()
def get_flat_lit_with_edges_program():
    return _load('lit_with_edges.glsl', defines={ 'SMOOTH_SHADING': 0, 'TEXTURE': 0 })


@functools.lru_cache()
def get_smooth_lit_texturized_program():
    return _load('lit_with_edges.glsl', defines={ 'SMOOTH_SHADING': 1, 'TEXTURE': 1 })


@functools.lru_cache()
def get_simple_unlit_program():
    return _load('simple_unlit.glsl')


@functools.lru_cache()
def get_cylinder_program():
    return _load('cylinder.glsl')


@functools.lru_cache()
def get_screen_texture_program():
    return _load('screen_texture.glsl')


@functools.lru_cache()
def get_chessboard_program():
    return _load('chessboard.glsl')

def clear_shader_cache():
    """Clear all cached shaders."""
    funcs =  [
        get_smooth_lit_with_edges_program,
        get_flat_lit_with_edges_program,
        get_smooth_lit_texturized_program,
        get_simple_unlit_program,
        get_cylinder_program,
        get_screen_texture_program,
        get_chessboard_program,
    ]
    for f in funcs:
        f.cache_clear()


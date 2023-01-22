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
import functools

from moderngl_window import resources
from moderngl_window.meta import ProgramDescription


def _load(name, defines={}):
    return resources.programs.load(ProgramDescription(path=name, defines=defines))


@functools.lru_cache()
def get_lit_program(vs, smooth_shading, texture, face_color, instanced=0):
    defines = {
        "SMOOTH_SHADING": smooth_shading,
        "TEXTURE": texture,
        "FACE_COLOR": face_color,
        "INSTANCED": instanced,
    }
    return resources.programs.load(
        ProgramDescription(
            vertex_shader=vs,
            geometry_shader="lit_with_edges.glsl",
            fragment_shader="lit_with_edges.glsl",
            defines=defines,
        )
    )


def get_smooth_lit_with_edges_program(vs, instanced=0):
    return get_lit_program(vs, smooth_shading=1, texture=0, face_color=0, instanced=instanced)


def get_smooth_lit_with_edges_face_color_program(vs, instanced=0):
    return get_lit_program(vs, smooth_shading=1, texture=0, face_color=1, instanced=instanced)


def get_flat_lit_with_edges_face_color_program(vs, instanced=0):
    return get_lit_program(vs, smooth_shading=0, texture=0, face_color=1, instanced=instanced)


def get_flat_lit_with_edges_program(vs, instanced=0):
    return get_lit_program(vs, smooth_shading=0, texture=0, face_color=0, instanced=instanced)


def get_flat_lit_with_edges_program(vs, instanced=0):
    return get_lit_program(vs, smooth_shading=0, texture=0, face_color=0, instanced=instanced)


def get_smooth_lit_texturized_program(vs, instanced=0):
    return get_lit_program(vs, smooth_shading=1, texture=1, face_color=0, instanced=instanced)


def get_sphere_instanced_program():
    return get_smooth_lit_with_edges_program("sphere_instanced.vs.glsl")


def get_lines_instanced_program():
    return get_smooth_lit_with_edges_program("lines_instanced.vs.glsl")


@functools.lru_cache()
def get_outline_program(vs_path, instanced=0):
    defines = {"INSTANCED": instanced}
    return resources.programs.load(
        ProgramDescription(
            vertex_shader=vs_path,
            fragment_shader="outline/outline_prepare.fs.glsl",
            defines=defines,
        )
    )


@functools.lru_cache()
def get_fragmap_program(vs_path, instanced=0):
    defines = {"INSTANCED": instanced}
    return resources.programs.load(
        ProgramDescription(
            vertex_shader=vs_path,
            fragment_shader="fragment_picking/frag_map.fs.glsl",
            defines=defines,
        )
    )


@functools.lru_cache()
def get_depth_only_program(vs_path, instanced=0):
    defines = {"INSTANCED": instanced}
    return resources.programs.load(
        ProgramDescription(
            vertex_shader=vs_path,
            fragment_shader="shadow_mapping/depth_only.fs.glsl",
            defines=defines,
        )
    )


@functools.lru_cache()
def get_simple_unlit_program():
    return _load("simple_unlit.glsl")


@functools.lru_cache()
def get_cylinder_program():
    return _load("cylinder.glsl")


@functools.lru_cache()
def get_screen_texture_program():
    return _load("screen_texture.glsl")


@functools.lru_cache()
def get_chessboard_program():
    return _load("chessboard.glsl")


def clear_shader_cache():
    """Clear all cached shaders."""
    funcs = [
        get_lit_program,
        get_outline_program,
        get_fragmap_program,
        get_depth_only_program,
        get_simple_unlit_program,
        get_cylinder_program,
        get_screen_texture_program,
        get_chessboard_program,
    ]
    for f in funcs:
        f.cache_clear()

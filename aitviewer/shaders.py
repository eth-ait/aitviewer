# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import functools
import os

import moderngl
from moderngl_window import resources
from moderngl_window.meta import ProgramDescription


def load_program(name, defines: dict = None):
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
    return load_program("simple_unlit.glsl")


@functools.lru_cache()
def get_cylinder_program():
    return load_program("cylinder.glsl")


@functools.lru_cache()
def get_screen_texture_program():
    return load_program("screen_texture.glsl")


@functools.lru_cache()
def get_chessboard_program():
    return load_program("chessboard.glsl")


@functools.lru_cache()
def get_marching_cubes_shader(name, BX, BY, BZ, COMPACT_GROUP_SIZE) -> moderngl.ComputeShader:
    defines = {
        "BLOCK_SIZE_X": BX,
        "BLOCK_SIZE_Y": BY,
        "BLOCK_SIZE_Z": BZ,
        "COMPACT_GROUP_SIZE": COMPACT_GROUP_SIZE,
    }
    path = os.path.join("marching_cubes", name)
    return resources.programs.load(ProgramDescription(compute_shader=path, defines=defines))


@functools.lru_cache()
def get_sort_program(name):
    defines = {
        "ENTRY_PARALLEL_SORT_" + name: 1,
    }
    path = os.path.join("gaussian_splatting", "sort.glsl")
    return resources.programs.load(ProgramDescription(compute_shader=path, defines=defines))


@functools.lru_cache()
def get_gaussian_splat_prepare_program(PREPARE_GROUP_SIZE):
    defines = {
        "PREPARE_GROUP_SIZE": PREPARE_GROUP_SIZE,
    }
    path = os.path.join("gaussian_splatting", "prepare.glsl")
    return resources.programs.load(ProgramDescription(compute_shader=path, defines=defines))


@functools.lru_cache()
def get_gaussian_splat_draw_program():
    return load_program(os.path.join("gaussian_splatting", "draw.glsl"))


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
        get_marching_cubes_shader,
        get_gaussian_splat_draw_program,
        get_gaussian_splat_prepare_program,
        get_sort_program,
    ]
    for f in funcs:
        f.cache_clear()

#version 400

#include directional_lights.glsl

#if defined VERTEX_SHADER

    uniform mat4 view_projection_matrix;
    uniform mat4 model_matrix;

    in vec3 in_position;
    in vec3 in_normal;
    in vec2 in_uv;

    out vec3 v_vert;
    out vec3 v_norm;
    out vec2 v_uv;
    out vec4 v_vert_light[NR_DIR_LIGHTS];

    void main() {
        v_norm = (model_matrix * vec4(in_normal, 0.0)).xyz;
        v_uv = in_uv;

        vec3 world_position = (model_matrix * vec4(in_position, 1.0)).xyz;
        v_vert = world_position;
        gl_Position = view_projection_matrix * vec4(world_position, 1.0);

        for(int i = 0; i < NR_DIR_LIGHTS; i++) {
            v_vert_light[i] = dirLights[i].matrix * vec4(in_position, 1.0);
        }
    }



#elif defined FRAGMENT_SHADER

    uniform vec4 color_1;
    uniform vec4 color_2;
    uniform float n_tiles;
    uniform bool tiling_enabled;

    in vec3 v_vert;
    in vec3 v_norm;
    in vec2 v_uv;
    in vec4 v_vert_light[NR_DIR_LIGHTS];

    out vec4 f_color;

    // Box filtered checkerboard by Inigo Quilez:
    // https://iquilezles.org/articles/filterableprocedurals/
    vec4 checkerboard(vec4 c1, vec4 c2, vec2 p) {
        vec2 w = max(abs(dFdx(p)), abs(dFdy(p))) + 0.01;
        vec2 i = 2.0 * (abs(fract((p - 0.5 * w) / 2.0) - 0.5) - abs(fract((p + 0.5 * w) / 2.0) - 0.5)) / w;
        float weight = 0.5 - 0.5 * i.x * i.y;
        return mix(c1, c2, weight);
    }

    void main() {
        // Compute normal
        vec3 normal = normalize(v_norm);

        // Compute base color from UVs and chessboard parameters
        vec4 t_color;
        if(tiling_enabled) {
            t_color = checkerboard(color_1, color_2, v_uv * n_tiles);
        } else {
            t_color = color_1;
        }

        // Compute shadow value
        vec3 color = compute_lighting(t_color.rgb, v_vert, normal, v_vert_light);

        // Output resulting color with the original alpha value
        f_color = vec4(color, t_color.a);
    }


#endif
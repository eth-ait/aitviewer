#version 400

// Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos

#if defined VERTEX_SHADER

    in vec3 in_position;
    in vec4 in_color;

    uniform mat4 model_matrix;
    uniform mat4 model_view_matrix;
    uniform mat4 view_projection_matrix;

    uniform float base_point_size;
    uniform float min_point_size;
    uniform float max_point_size;
    uniform float projection_scale;
    uniform int orthographic;

    out vec4 v_color;

    void main() {
        vec4 world_position = model_matrix * vec4(in_position, 1.0);
        vec4 view_position = model_view_matrix * vec4(in_position, 1.0);

        float depth = max(0.0001, abs(view_position.z));
        float adaptive_size = base_point_size;
        if (orthographic == 0) {
            adaptive_size = base_point_size * projection_scale / depth;
        }
        adaptive_size = clamp(adaptive_size, min_point_size, max_point_size);

        gl_Position = view_projection_matrix * world_position;
        gl_PointSize = adaptive_size;
        v_color = in_color;
    }

#elif defined FRAGMENT_SHADER

    in vec4 v_color;

    out vec4 f_color;

    void main() {
        vec2 coords = gl_PointCoord * 2.0 - 1.0;
        float radius_sq = dot(coords, coords);

        if (radius_sq > 1.0) {
            discard;
        }

        f_color = vec4(v_color.rgb, v_color.a);
    }

#endif

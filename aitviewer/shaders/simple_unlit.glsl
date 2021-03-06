#version 330

#if defined VERTEX_SHADER

    in vec3 in_position;
    uniform mat4 mvp;

    in vec4 in_color;
    out vec4 v_color;

    void main() {
        v_color = in_color;
        gl_Position = mvp * vec4(in_position, 1.0);
    }

#elif defined FRAGMENT_SHADER

    out vec4 f_color;
    in vec4 v_color;

    void main() {
        f_color = v_color;
    }

#endif
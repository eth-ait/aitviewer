#version 400

#if defined VERTEX_SHADER

    in vec3 in_position;
    in vec2 in_texcoord_0;

    out vec2 uv;

    void main() {
        gl_Position = vec4(in_position, 1.0);
        uv = in_texcoord_0;
    }

#elif defined FRAGMENT_SHADER

    layout(location=0) out vec4 out_color;
    in vec2 uv;

    uniform sampler2D texture0;

    void main() {
        float depth_value = texture(texture0, uv).r;
        out_color = vec4(vec3(depth_value), 1.0);
    }

#endif

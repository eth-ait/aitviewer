#version 400

#if defined VERTEX_SHADER

    in vec3 in_position;
    in vec2 in_texcoord_0;
    out vec2 uv0;

    uniform mat4 mvp;

    void main() {
        gl_Position = mvp * vec4(in_position, 1);
        uv0 = in_texcoord_0;
    }

#elif defined FRAGMENT_SHADER

    out vec4 fragColor;
    uniform sampler2D texture0;
    uniform float transparency;
    in vec2 uv0;

    void main() {
        fragColor = vec4(vec3(texture(texture0, uv0 * vec2(-1.0, -1.0))), transparency);
    }

#endif
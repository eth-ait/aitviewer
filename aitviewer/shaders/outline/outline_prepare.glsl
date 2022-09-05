#version 400

#if defined VERTEX_SHADER

    in vec3 in_position;

    uniform mat4 mvp;

    void main() {
        // Transform vertices with the given mvp matrix.
        gl_Position = mvp * vec4(in_position, 1.0);
    }

#elif defined FRAGMENT_SHADER
    out vec4 color;
    void main() {
        // Draw everything white.
        color = vec4(1.0);
    }
#endif

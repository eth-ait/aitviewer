#version 400

#if defined VERTEX_SHADER

    in vec3 in_position;

    uniform mat4 mvp;

    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
    }

#elif defined FRAGMENT_SHADER

    void main() {
        // This is not needed because this always computed by OpenGL.
        // gl_FragDepth = gl_FragCoord.z;
    }

#endif

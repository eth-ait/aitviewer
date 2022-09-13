#version 400

#if defined VERTEX_SHADER

    in vec3 in_position;

    uniform mat4 view_projection_matrix;
    uniform mat4 model_matrix;

    void main() {
        vec3 world_position = (model_matrix * vec4(in_position, 1.0)).xyz;
        gl_Position = view_projection_matrix * vec4(world_position, 1.0);
    }

#elif defined FRAGMENT_SHADER

    void main() {
        // This is not needed because this always computed by OpenGL.
        // gl_FragDepth = gl_FragCoord.z;
    }

#endif

#version 400

#if defined FRAGMENT_SHADER

#include clipping.glsl
    in vec3 local_pos;

    void main() {
        discard_if_clipped(local_pos);
        // This is not needed because this always computed by OpenGL.
        // gl_FragDepth = gl_FragCoord.z;
    }

#endif

#version 400

// Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos

#if defined FRAGMENT_SHADER

#include clipping.glsl
    in vec3 local_pos;

    void main() {
        discard_if_clipped(local_pos);
        // This is not needed because this always computed by OpenGL.
        // gl_FragDepth = gl_FragCoord.z;
    }

#endif

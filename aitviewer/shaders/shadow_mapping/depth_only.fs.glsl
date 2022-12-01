#version 400

#if defined FRAGMENT_SHADER

    void main() {
        // This is not needed because this always computed by OpenGL.
        // gl_FragDepth = gl_FragCoord.z;
    }

#endif

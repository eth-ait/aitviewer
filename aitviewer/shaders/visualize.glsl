#version 400

#if defined VERTEX_SHADER

    in vec3 in_position;
    in vec2 in_texcoord_0;
    out vec2 uv0;

    void main() {
        gl_Position = vec4(in_position, 1);
        uv0 = in_texcoord_0;
    }

#elif defined FRAGMENT_SHADER

    out vec4 fragColor;
    uniform sampler2D texture0;
    uniform bool hash_color;
    in vec2 uv0;

    // MurmurHash3 32-bit finalizer.
    uint hash(uint h) {
        h ^= h >> 16;
        h *= 0x85ebca6bu;
        h ^= h >> 13;
        h *= 0xc2b2ae35u;
        h ^= h >> 16;
        return h;
    }

    // Hash an int value and return a color.
    vec3 int_to_color(uint i) {
        uint h = hash(i);

        vec3 c = vec3(
            (h >>  0u) & 255u,
            (h >>  8u) & 255u,
            (h >> 16u) & 255u
        );

        return c * (1.0 / 255.0);
    }

    void main() {
        if(hash_color) {
            // Visualize the red channel hashed
            uint id = floatBitsToUint(texture(texture0, uv0).r);
            fragColor = vec4(int_to_color(id), 1.0);
        } else {
            fragColor = texture(texture0, uv0);
        }
    }

#endif
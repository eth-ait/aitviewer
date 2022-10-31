#version 400

#if defined VERTEX_SHADER

    in vec3 in_position;

    void main() {
        gl_Position = vec4(in_position, 1.0);
    }

#elif defined FRAGMENT_SHADER
    out vec4 fragColor;

    uniform vec4 outline_color;
    uniform sampler2D outline;

    void main() {
        // Transform pixel coordinates from float to int.
        ivec2 coords = ivec2(gl_FragCoord.xy);

        // Load the value at the current pixel.
        float v = texelFetch(outline, coords, 0).r;

        // If inside one of the objects to outline, discard.
        if(v > 0) {
            discard;
        }

        // Check if any of the neighbouring pixels is inside an object to outline.
        bool ok = false;
        for(int y = -1; y <= 1; y++) {
            for(int x = -1; x <= 1; x++) {
                if(y == 0 && x == 0) {
                    continue;
                }

                // Load a pixel at distance 2 and check if it's inside an object
                float v = texelFetch(outline, coords + ivec2(x, y) * 2, 0).r;
                if(v > 0) {
                    ok = true;
                }
            }
        }

        // If none of the pixels were inside discard
        if(!ok) {
            discard;
        }

        // Otherwise draw this pixel with the outline color
        fragColor = outline_color;
    }
#endif

#version 330

#if defined VERTEX_SHADER

    uniform mat4 mvp;
    uniform mat4 light_mvp;

    in vec3 in_position;
    in vec3 in_normal;
    in vec2 in_uv;

    out vec3 v_vert;
    out vec3 v_norm;
    out vec2 v_uv;
    out vec4 v_vert_light;

    void main() {
        v_vert = in_position;
        v_norm = in_normal;
        v_uv = in_uv;
        gl_Position = mvp * vec4(in_position, 1.0);
        v_vert_light = light_mvp * vec4(in_position, 1.0);
    }



#elif defined FRAGMENT_SHADER


    #include directional_lights.glsl
    #include utils.glsl
    #include shadow_calculation.glsl

    uniform DirLight dirLight[NR_DIR_LIGHTS];
    uniform vec4 color_1;
    uniform vec4 color_2;
    uniform float n_tiles;
    uniform bool tiling_enabled;

    in vec3 v_vert;
    in vec3 v_norm;
    in vec2 v_uv;
    in vec4 v_vert_light;

    out vec4 f_color;

    // Box filtered checkerboard by Inigo Quilez: 
    // https://iquilezles.org/articles/filterableprocedurals/
    vec4 checkerboard(vec4 c1, vec4 c2, vec2 p) {
        vec2 w = max(abs(dFdx(p)), abs(dFdy(p))) + 0.01; 
        vec2 i = 2.0 * (abs(fract((p - 0.5 * w) / 2.0) - 0.5) - abs(fract((p + 0.5 * w) / 2.0) - 0.5)) / w;
        float weight = 0.5 - 0.5 * i.x * i.y;
        return mix(c1, c2, weight);
    }

    void main() {
        // Compute normal
        vec3 normal = normalize(v_norm);

        // Compute base color from UVs and chessboard parameters
        vec4 t_color;
        if(tiling_enabled) {
            t_color = checkerboard(color_1, color_2, v_uv * n_tiles);   
        } else {
            t_color = color_1;
        }

        // Compute shadow value
        float shadow = shadow_calculation(v_vert_light, dirLight[0].pos, normal);
        vec3 color = vec3(0.0, 0.0, 0.0);
        for(int i = 0; i < NR_DIR_LIGHTS; i++){
            // We only have shadows for the first light in the scene.
            float s = i == 0 ? shadow : 0.0f;
            color += directionalLight(dirLight[i], t_color.rgb, v_vert, normal, s);
        }

        // Output resulting color with the original alpha value
        f_color = vec4(color, t_color.a);
    }


#endif
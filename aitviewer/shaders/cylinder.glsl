#version 400

#if defined VERTEX_SHADER

    in vec3 in_position;
    in vec4 in_color;

    out vec4 v_color;

    void main() {
        v_color = in_color;
        gl_Position = vec4(in_position, 1.0);
    }

#elif defined GEOMETRY_SHADER

    layout (lines) in;
    layout (triangle_strip, max_vertices=128) out;

    // Based on https://github.com/torbjoern/polydraw_scripts/blob/master/geometry/drawcone_geoshader.pss

    uniform mat4 mvp;
    uniform float r1;
    uniform float r2;

    in vec4 v_color[2];

    out vec3 g_vert;
    out vec3 g_norm;
    out vec4 g_color;

    // given to points p1 and p2 create a vector out
    // that is perpendicular to (p2-p1)
    vec3 createPerp(vec3 p1, vec3 p2) {
        vec3 invec = normalize(p2 - p1);
        vec3 ret = cross( invec, vec3(0.0, 0.0, 1.0) );
        if (length(ret) == 0.0) {
            ret = cross( invec, vec3(0.0, 1.0, 0.0) );
        }
        return ret;
    }

    void main() {
        vec3 v1 = gl_in[0].gl_Position.xyz;
        vec3 v2 = gl_in[1].gl_Position.xyz;
        vec3 axis = v2 - v1;

        vec3 perpx = createPerp( v2, v1 );
        vec3 perpy = cross( normalize(axis), perpx );
        int segs = 8;

        for(int i = 0; i < segs; i++) {
            float a = i / float(segs - 1) * 2.0 * 3.14159;
            float ca = cos(a);
            float sa = sin(a);
            vec3 normal = vec3( ca*perpx.x + sa*perpy.x,
                              ca*perpx.y + sa*perpy.y,
                              ca*perpx.z + sa*perpy.z );

            normal = normalize(normal);
            vec3 p1 = v1 + r1*normal;
            vec3 p2 = v2 + r2*normal;

            gl_Position = mvp * vec4(p2, 1.0);
            g_vert = p2;
            g_norm = normal;
            g_color = v_color[0];
            EmitVertex();

            gl_Position = mvp * vec4(p1, 1.0);
            g_vert = p1;
            g_norm = normal;
            g_color = v_color[0];
            EmitVertex();
        }
        EndPrimitive();

        // close the cylinder
        vec3 axis_norm = normalize(axis);
        if (r1 > 0.0) {
            for(int i=0; i < segs; i++) {
                float a = i/float(segs-1) * 2.0 * 3.14159;
                float ca = cos(a);
                float sa = sin(a);

                vec3 normal = vec3( ca*perpx.x + sa*perpy.x,
                             ca*perpx.y + sa*perpy.y,
                             ca*perpx.z + sa*perpy.z );

                normal = normalize(normal);
                vec3 p1 = v1 + r1*normal;

                gl_Position = mvp * vec4(p1, 1.0);
                g_vert = p1;
                g_norm = axis_norm;
                g_color = v_color[0];
                EmitVertex();

                gl_Position = mvp * vec4(v1, 1.0);
                g_vert = v1;
                g_norm = axis_norm;
                g_color = v_color[0];
                EmitVertex();
            }
        }
        EndPrimitive();

        if (r2 > 0.0) {
            for(int i=0; i < segs; i++) {
                float a = i / float(segs - 1) * 2.0 * 3.14159;
                float ca = cos(a);
                float sa = sin(a);

                vec3 normal = vec3( ca*perpx.x + sa*perpy.x,
                             ca*perpx.y + sa*perpy.y,
                             ca*perpx.z + sa*perpy.z );

                normal = normalize(normal);
                vec3 p2 = v2 + r2*normal;

                gl_Position = mvp * vec4(v2, 1.0);
                g_vert = v2;
                g_norm = axis_norm;
                g_color = v_color[0];
                EmitVertex();

                gl_Position = mvp * vec4(p2, 1.0);
                g_vert = p2;
                g_norm = axis_norm;
                g_color = v_color[0];
                EmitVertex();
            }
        }

        EndPrimitive();
    }

#elif defined FRAGMENT_SHADER

    #include directional_lights.glsl

    in vec3 g_vert;
    in vec3 g_norm;
    in vec4 g_color;

    out vec4 f_color;

    void main() {
        vec3 normal = normalize(g_norm);

        vec3 color = vec3(0.0, 0.0, 0.0);
        for(int i = 0; i < NR_DIR_LIGHTS; i++){
            color += directionalLight(dirLights[i], g_color.rgb, g_vert, normal, 0.0f);
        }
        f_color = vec4(color, g_color.w);
    }

#endif
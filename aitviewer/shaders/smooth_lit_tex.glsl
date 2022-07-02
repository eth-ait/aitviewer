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

#elif defined GEOMETRY_SHADER

    #include directional_lights.glsl
    #include utils.glsl

    layout (triangles) in;
    layout (triangle_strip, max_vertices=3) out;

    uniform vec2 win_size;
    noperspective out vec3 dist;

    // pass-through variables
    in vec3 v_norm[3];
    out vec3 g_norm;

    in vec2 v_uv[3];
    out vec2 g_uv;

    in vec3 v_vert[3];
    out vec3 g_vert;

    in vec4 v_vert_light[3];
    out vec4 g_vert_light;

    void main() {
        vec3 edge_dist = distanceToEdge(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_in[2].gl_Position, win_size);
    //    vec4 color = (v_color[0] + v_color[1] + v_color[2]) / 3.0;

        dist = vec3(edge_dist[0], 0, 0);
        gl_Position = gl_in[0].gl_Position;
        g_norm = v_norm[0];
        g_vert = v_vert[0];
        g_uv = v_uv[0];
        g_vert_light = v_vert_light[0];
        EmitVertex();

        dist = vec3(0, edge_dist[1], 0);
        gl_Position = gl_in[1].gl_Position;
        g_norm = v_norm[1];
        g_vert = v_vert[1];
        g_uv = v_uv[1];
        g_vert_light = v_vert_light[1];
        EmitVertex();

        dist = vec3(0, 0, edge_dist[2]);
        gl_Position = gl_in[2].gl_Position;
        g_norm = v_norm[2];
        g_vert = v_vert[2];
        g_uv = v_uv[2];
        g_vert_light = v_vert_light[2];
        EmitVertex();

        EndPrimitive();
    }

#elif defined FRAGMENT_SHADER

    #include phong_tex.glsl

#endif
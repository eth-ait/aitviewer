#version 400

#include directional_lights.glsl

#if defined VERTEX_SHADER

    uniform mat4 mvp;

    in vec3 in_position;
    in vec3 in_normal;
    in vec2 in_uv;

    out VS_OUT {
        vec3 vert;
        vec3 norm;
        vec2 uv;
        vec4 vert_light[NR_DIR_LIGHTS];
    } vs_out;

    void main() {
        vs_out.vert = in_position;
        vs_out.norm = in_normal;
        vs_out.uv = in_uv;
        gl_Position = mvp * vec4(in_position, 1.0);

        for(int i = 0; i < NR_DIR_LIGHTS; i++) {
            vs_out.vert_light[i] = dirLights[i].matrix * vec4(in_position, 1.0);
        }
    }

#elif defined GEOMETRY_SHADER

    #include utils.glsl

    layout (triangles) in;
    layout (triangle_strip, max_vertices=3) out;

    uniform vec2 win_size;

    // computed variables
    noperspective out vec3 dist;

    // pass-through variables
    in VS_OUT {
        vec3 vert;
        vec3 norm;
        vec2 uv;
        vec4 vert_light[NR_DIR_LIGHTS];
    } gs_in[];

    out vec3 g_norm;
    out vec2 g_uv;
    out vec3 g_vert;
    out vec4 g_vert_light[NR_DIR_LIGHTS];

    void main() {
        vec3 edge_dist = distanceToEdge(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_in[2].gl_Position, win_size);

        
        vec3 dist_vecs[3] = vec3[3] (
            vec3(edge_dist[0], 0, 0),
            vec3(0, edge_dist[1], 0),
            vec3(0, 0, edge_dist[2])
        );
    //    vec4 color = (v_color[0] + v_color[1] + v_color[2]) / 3.0;

        dist = dist_vecs[0];
        gl_Position = gl_in[0].gl_Position;
        g_vert = gs_in[0].vert;
        g_norm = gs_in[0].norm;
        g_uv = gs_in[0].uv;
        for(int j = 0; j < NR_DIR_LIGHTS; j++) {
            g_vert_light[j] = gs_in[0].vert_light[j];
        } 
        EmitVertex();

        dist = dist_vecs[1];
        gl_Position = gl_in[1].gl_Position;
        g_vert = gs_in[1].vert;
        g_norm = gs_in[1].norm;
        g_uv = gs_in[1].uv;
        for(int j = 0; j < NR_DIR_LIGHTS; j++) {
            g_vert_light[j] = gs_in[1].vert_light[j];
        } 
        EmitVertex();
        
        dist = dist_vecs[2];
        gl_Position = gl_in[2].gl_Position;
        g_vert = gs_in[2].vert;
        g_norm = gs_in[2].norm;
        g_uv = gs_in[2].uv;
        for(int j = 0; j < NR_DIR_LIGHTS; j++) {
            g_vert_light[j] = gs_in[2].vert_light[j];
        } 
        EmitVertex();

        EndPrimitive();
    }

#elif defined FRAGMENT_SHADER

    #include phong_tex.glsl

#endif
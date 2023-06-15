#version 400

#define SMOOTH_SHADING 0
#define TEXTURE 0
#define FACE_COLOR 0
#define INSTANCED 0

#include directional_lights.glsl

#if defined VERTEX_SHADER

    uniform mat4 view_projection_matrix;
    uniform mat4 model_matrix;


    in vec3 in_position;
#if INSTANCED
    in mat4 instance_transform;
#endif

#if SMOOTH_SHADING
    in vec3 in_normal;
#endif

#if TEXTURE
    in vec2 in_uv;
#endif
    in vec4 in_color;

    out VS_OUT {
        vec3 vert;
        vec3 local_vert;

#if SMOOTH_SHADING
        vec3 norm;
#endif

#if TEXTURE
        vec2 uv;
#endif
#if !FACE_COLOR
        vec4 color;
#endif
        vec4 vert_light[NR_DIR_LIGHTS];
    } vs_out;


    void main() {

#if INSTANCED
        mat4 transform = model_matrix * instance_transform;
#else
        mat4 transform = model_matrix;
#endif

#if SMOOTH_SHADING
        vs_out.norm = (transform * vec4(in_normal, 0.0)).xyz;
#endif

#if TEXTURE
        vs_out.uv = in_uv;
#endif
#if !FACE_COLOR
        vs_out.color = in_color;
#endif
        vs_out.local_vert = in_position;
        vec3 world_position = (transform * vec4(in_position, 1.0)).xyz;
        vs_out.vert = world_position;
        gl_Position = view_projection_matrix * vec4(world_position, 1.0);

        for(int i = 0; i < NR_DIR_LIGHTS; i++) {
#if INSTANCED
            vs_out.vert_light[i] = dirLights[i].matrix * (instance_transform * vec4(in_position, 1.0));
#else
            vs_out.vert_light[i] = dirLights[i].matrix * vec4(in_position, 1.0);
#endif
        }
    }

#elif defined GEOMETRY_SHADER

    layout (triangles) in;
    layout (triangle_strip, max_vertices=3) out;

    uniform vec2 win_size;

    // computed variables
    noperspective out vec3 dist;

    // pass-through variables
    in VS_OUT {
        vec3 vert;
        vec3 local_vert;

#if SMOOTH_SHADING
        vec3 norm;
#endif

#if TEXTURE
        vec2 uv;
#endif
#if !FACE_COLOR
        vec4 color;
#endif
        vec4 vert_light[NR_DIR_LIGHTS];
    } gs_in[];

    out vec3 g_norm;

#if FACE_COLOR
    // [#faces x 1] texture of per face colors.
    uniform sampler2D face_colors;
#endif

#if TEXTURE
    out vec2 g_uv;
#endif

    out vec4 g_color;
    out vec3 g_vert;
    out vec3 g_local_vert;
    out vec4 g_vert_light[NR_DIR_LIGHTS];

    vec3 distanceToEdge(vec4 v0, vec4 v1, vec4 v2, vec2 win_size) {
        // From "Single-Pass Wireframe Rendering".
        vec2 p0 = win_size * v0.xy/v0.w;
        vec2 p1 = win_size * v1.xy/v1.w;
        vec2 p2 = win_size * v2.xy/v2.w;
        vec2 pp0 = p2-p1;
        vec2 pp1 = p2-p0;
        vec2 pp2 = p1-p0;
        float area = abs(pp1.x * pp2.y - pp1.y * pp2.x);
        vec3 dist = vec3(area / length(pp0),
                        area / length(pp1),
                        area / length(pp2));
        return dist;
    }

    void main() {
        vec3 edge_dist = distanceToEdge(gl_in[0].gl_Position, gl_in[1].gl_Position, gl_in[2].gl_Position, win_size);

        vec3 dist_vecs[3] = vec3[3] (
            vec3(edge_dist[0], 0, 0),
            vec3(0, edge_dist[1], 0),
            vec3(0, 0, edge_dist[2])
        );

#if !SMOOTH_SHADING
        vec3 norm = normalize(cross(gs_in[1].vert - gs_in[0].vert, gs_in[2].vert - gs_in[0].vert));
#endif

        for(int i = 0; i < 3; i++) {
            dist = dist_vecs[i];
            gl_Position = gl_in[i].gl_Position;
            g_vert = gs_in[i].vert;
            g_local_vert = gs_in[i].local_vert;

#if SMOOTH_SHADING
            g_norm = gs_in[i].norm;
#else
            g_norm = norm;
#endif

#if TEXTURE
            g_uv = gs_in[i].uv;
#endif
#if !FACE_COLOR
            g_color = gs_in[i].color;
#else
            g_color = texelFetch(face_colors, ivec2(gl_PrimitiveIDIn % 8192, gl_PrimitiveIDIn / 8192), 0);
#endif

            for(int j = 0; j < NR_DIR_LIGHTS; j++) {
                g_vert_light[j] = gs_in[i].vert_light[j];
            }
            EmitVertex();
        }

        EndPrimitive();
    }

#elif defined FRAGMENT_SHADER

#if TEXTURE
    uniform sampler2D diffuse_texture;
#else
    uniform bool norm_coloring;
#endif

    uniform float draw_edges;
    uniform bool use_uniform_color;
    uniform vec4 uniform_color;

    const vec4 edge_color = vec4(0.0, 0.0, 0.0, 1.0);

    in vec3 g_local_vert;
    in vec3 g_vert;
    in vec3 g_norm;

#if TEXTURE
    in vec2 g_uv;
#endif
    in vec4 g_color;

    in vec4 g_vert_light[NR_DIR_LIGHTS];
    noperspective in vec3 dist;

    out vec4 f_color;

#include clipping.glsl

    void main() {
        discard_if_clipped(g_local_vert);

        // Determine distance of this fragment to the closest edge.
        float d = min(min(dist[0], dist[1]), dist[2]);
        float ei = exp2(-1.0*d*d);
        ei = ei * ei * ei * ei * draw_edges;

        vec3 normal = normalize(g_norm);

        vec4 base_color = g_color;
        if(use_uniform_color) {
            base_color = uniform_color;
        }

#if TEXTURE
        // Texture lookup.
        base_color.xyz = texture(diffuse_texture, g_uv).rgb;
#endif

        vec3 color = compute_lighting(base_color.rgb, g_vert, normal, g_vert_light);
        f_color = vec4(color, base_color.a);

#if !TEXTURE
        if (norm_coloring) {
            float v = compute_diffuse_shadows(g_vert, normal, g_vert_light);
            vec3 c = .5 + .5 * normal;
            f_color = vec4(c * (v * 0.3 + 0.7), base_color.a);
        }
#endif


        f_color = ei * edge_color + (1.0 - ei) * f_color;
    }

#endif


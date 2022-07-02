#include directional_lights.glsl
#include utils.glsl
#include shadow_calculation.glsl

uniform sampler2D diffuse_texture;
uniform DirLight dirLight[NR_DIR_LIGHTS];

in vec3 g_vert;
in vec3 g_norm;
in vec2 g_uv;
in vec4 g_vert_light;

out vec4 f_color;

noperspective in vec3 dist;
const vec4 edge_color = vec4(0.0, 0.0, 0.0, 1.0);
uniform float draw_edges;
uniform float texture_alpha = 1.0;

void main() {
    // Determine distance of this fragment to the closest edge.
    float d = min(min(dist[0], dist[1]), dist[2]);
    float ei = exp2(-1.0*d*d);
    ei = ei * ei * ei * ei * draw_edges;

    vec3 normal = normalize(g_norm);
    float shadow = shadow_calculation(g_vert_light, dirLight[0].pos, normal);

    // Texture lookup.
    vec4 t_color = texture(diffuse_texture, g_uv);

    vec3 color = vec3(0.0, 0.0, 0.0);
    for(int i = 0; i < NR_DIR_LIGHTS; i++){
        // We only have shadows for the first light in the scene.
        float s = i == 0 ? shadow : 0.0f;
        color += directionalLight(dirLight[i], t_color.rgb, g_vert, normal, s);
    }

    // Shading.
    f_color = vec4(color, texture_alpha);
    f_color = ei * edge_color + (1.0 - ei) * f_color;
}
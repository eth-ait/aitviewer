#include utils.glsl
#include shadow_calculation.glsl

in vec3 g_vert;
in vec3 g_norm;
in vec4 g_color;
in vec4 g_vert_light[NR_DIR_LIGHTS];

out vec4 f_color;

noperspective in vec3 dist;
const vec4 edge_color = vec4(0.0, 0.0, 0.0, 1.0);
uniform float draw_edges;
uniform bool norm_coloring;


void main() {
    // Determine distance of this fragment to the closest edge.
    float d = min(min(dist[0], dist[1]), dist[2]);
    float ei = exp2(-1.0*d*d);
    ei = ei * ei * ei * ei * draw_edges;

    vec3 normal = normalize(g_norm);

    vec3 color = vec3(0.0, 0.0, 0.0);

    if (norm_coloring) {
        f_color = vec4(.5 + .5 * normal, g_color.w);
    } else {
        for (int i = 0; i < NR_DIR_LIGHTS; i++){
            // We only have shadows for the first light in the scene.
            float shadow = dirLights[i].shadow_enabled ? shadow_calculation(shadow_maps[i], g_vert_light[i], dirLights[i].pos, normal) : 0.0;
            color += directionalLight(dirLights[i], g_color.rgb, g_vert, normal, shadow);
        }
        
        f_color = vec4(color, g_color.w);
    }
    f_color = ei * edge_color + (1.0 - ei) * f_color;
}
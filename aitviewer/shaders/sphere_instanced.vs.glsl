#version 400

#include directional_lights.glsl

#if defined VERTEX_SHADER
    uniform mat4 view_projection_matrix;
    uniform mat4 model_matrix;

    uniform float radius;

    in vec3 in_position;
    in vec3 instance_position;
    in vec4 instance_color;

    out VS_OUT {
        vec3 vert;
        vec3 local_vert;
        vec3 norm;
        vec4 color;
        vec4 vert_light[NR_DIR_LIGHTS];
    } vs_out;


    void main() {
        vec3 position = in_position * radius + instance_position;
        vec3 world_position = (model_matrix * vec4(position, 1.0)).xyz;
        vs_out.local_vert = position;
        vs_out.vert = world_position;
        vs_out.norm = (model_matrix * vec4(in_position, 0.0)).xyz;
        vs_out.color = instance_color;

        gl_Position = view_projection_matrix * vec4(world_position, 1.0);

        for(int i = 0; i < NR_DIR_LIGHTS; i++) {
            vs_out.vert_light[i] = dirLights[i].matrix * vec4(position, 1.0);
        }
    }
#endif
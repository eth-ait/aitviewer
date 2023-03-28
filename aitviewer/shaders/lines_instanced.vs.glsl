#version 400

#include directional_lights.glsl

#if defined VERTEX_SHADER
    uniform mat4 view_projection_matrix;
    uniform mat4 model_matrix;

    uniform float r_base;
    uniform float r_tip;

    in vec3 in_position;
    in vec3 instance_base;
    in vec3 instance_tip;
    in vec4 instance_color;

    out VS_OUT {
        vec3 vert;
        vec3 local_vert;
        vec3 norm;
        vec4 color;
        vec4 vert_light[NR_DIR_LIGHTS];
    } vs_out;

    mat3 coordinateSystem(vec3 a) {
        vec3 c;
        if (abs(a.x) > abs(a.y)) {
            float invLen = 1.0 / length(a.xz);
            c = vec3(a.z * invLen, 0.0, -a.x * invLen);
        } else {
            float invLen = 1.0 / length(a.yz);
            c = vec3(0.0, a.z * invLen, -a.y * invLen);
        }

        vec3 b = cross(c, a);
        return mat3(b, c, a);
    }

    void main() {
        vec3 p0 = instance_base;
        vec3 p1 = instance_tip;

        vec3 p = in_position;
        if(p.z < 0.5) {
            p.xy *= r_base;
        } else {
            p.xy *= r_tip;
        }

        float d = length(p1 - p0);
        p.z *= d;

        vec3 v = normalize(p1 - p0);
        mat3 tbn = coordinateSystem(v);

        vec3 n;
        if(in_position.x != 0) {
            float z = (r_base - r_tip) / d;
            vec2 xy = in_position.xy * sqrt(1 - z * z);
            n = normalize(vec3(xy, z));
        } else {
            if(in_position.z > 0.5) {
                n = vec3(0, 0, 1);
            } else {
                n = vec3(0, 0, -1);
            }
        }

        vec3 position = tbn * p + p0;
        vec3 normal = tbn * n;

        vs_out.local_vert = position;
        vec3 world_position = (model_matrix * vec4(position, 1.0)).xyz;
        vs_out.vert = world_position;
        vs_out.norm = (model_matrix * vec4(normal, 0.0)).xyz;
        vs_out.color = instance_color;

        gl_Position = view_projection_matrix * vec4(world_position, 1.0);

        for(int i = 0; i < NR_DIR_LIGHTS; i++) {
            vs_out.vert_light[i] = dirLights[i].matrix * vec4(position, 1.0);
        }
    }
#endif
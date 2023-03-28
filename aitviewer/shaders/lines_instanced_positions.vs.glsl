#version 400

#include directional_lights.glsl

#if defined VERTEX_SHADER
    uniform mat4 view_projection_matrix;
    uniform mat4 model_matrix;

    uniform float r_base;
    uniform float r_tip;
    uniform vec4 color;

    in vec3 in_position;
    in vec3 instance_base;
    in vec3 instance_tip;

    out vec4 pos;
    out vec3 local_pos;
    flat out int instance_id;

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

        vec3 position = tbn * p + p0;
        vec3 world_position = (model_matrix * vec4(position, 1.0)).xyz;
        gl_Position = view_projection_matrix * vec4(world_position, 1.0);
        pos = vec4(world_position, 1.0);
        instance_id = gl_InstanceID;
        local_pos = position;
    }
#endif
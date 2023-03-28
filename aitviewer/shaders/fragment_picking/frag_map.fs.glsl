#version 400

#if defined FRAGMENT_SHADER

layout(location=0) out vec4 out_position;
layout(location=1) out vec4 out_obj_info;

in vec3 local_pos;
in vec4 pos;
flat in int instance_id;

int tri_id = gl_PrimitiveID;
uniform int obj_id;

#include clipping.glsl

void main() {
    discard_if_clipped(local_pos);

    out_obj_info = vec4(obj_id, tri_id, instance_id, 0.0);
    out_position = pos;
}

#endif

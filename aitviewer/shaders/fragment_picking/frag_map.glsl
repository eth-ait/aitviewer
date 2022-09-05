#version 400

#if defined VERTEX_SHADER

uniform mat4 modelview;
uniform mat4 projection;

in vec3 in_position;

out vec3 pos;

void main() {
    vec4 p = modelview * vec4(in_position, 1.0);
    gl_Position = projection * p;
    pos = p.xyz;

}

#elif defined FRAGMENT_SHADER

layout(location=0) out vec4 out_position;
layout(location=1) out vec4 out_obj_info;

in vec3 pos;
int tri_id = gl_PrimitiveID;
uniform int obj_id;

void main() {
    out_obj_info = vec4(obj_id, tri_id, 0.0, 0.0);
    out_position = vec4(pos, 0.0);
}
#endif

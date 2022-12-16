#version 400
//
// Picks a point from the depth buffer and returns the position in camera space, along with the object and triangle id
// Adapted from https://github.com/moderngl/moderngl-window/blob/master/examples/resources/programs/fragment_picking/picker.glsl
//

#if defined VERTEX_SHADER

uniform sampler2D position_texture;
uniform sampler2D obj_info_texture;

//uniform mat4 modelview;
uniform ivec2 texel_pos;

out vec3 out_position;
out int out_obj_id;
out int out_tri_id;
out int out_instance_id;

void main() {
    vec4 viewpos = texelFetch(position_texture, texel_pos, 0).rgba;
    if (viewpos.w == 0.0) {
        out_position = vec3(0.0);
        out_obj_id = -1;
        out_tri_id = -1;

    } else {
        out_position = viewpos.xyz;
        out_obj_id = int(texelFetch(obj_info_texture, texel_pos, 0).r);
        out_tri_id = int(texelFetch(obj_info_texture, texel_pos, 0).g);
        out_instance_id = int(texelFetch(obj_info_texture, texel_pos, 0).b);
    }
}
#endif
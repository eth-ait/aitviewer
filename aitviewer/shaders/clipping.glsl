uniform ivec3 clip_control = ivec3(0, 0, 0);
uniform vec3 clip_value;

// Clip fragments if clipping is enabled.
void discard_if_clipped(vec3 local_position) {
    if(clip_control.x != 0) {
        if(clip_control.x < 0 && local_position.x < clip_value.x) {
            discard;
        }
        if(clip_control.x > 0 && local_position.x > clip_value.x) {
            discard;
        }
    }

    if(clip_control.y != 0) {
        if(clip_control.y < 0 && local_position.y < clip_value.y) {
            discard;
        }
        if(clip_control.y > 0 && local_position.y > clip_value.y) {
            discard;
        }
    }

    if(clip_control.z != 0) {
        if(clip_control.z < 0 && local_position.z < clip_value.z) {
            discard;
        }
        if(clip_control.z > 0 && local_position.z > clip_value.z) {
            discard;
        }
    }
}

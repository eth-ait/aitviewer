#version 400

#if defined VERTEX_SHADER
    uniform mat4 view_projection_matrix;
    uniform mat4 model_matrix;
    uniform float radius;

    in vec3 in_position;
    in vec3 instance_position;

    out vec4 pos;
    out vec3 local_pos;
    flat out int instance_id;

    void main() {
        vec3 position = in_position * radius + instance_position;
        vec3 world_position = (model_matrix * vec4(position, 1.0)).xyz;
        gl_Position = view_projection_matrix * vec4(world_position, 1.0);
        pos = vec4(world_position, 1.0);
        instance_id = gl_InstanceID;
        local_pos = position;
    }
#endif
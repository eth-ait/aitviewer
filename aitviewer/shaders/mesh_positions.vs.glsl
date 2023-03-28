#version 400

#define INSTANCED 0

#if defined VERTEX_SHADER
    in vec3 in_position;
#if INSTANCED
    in mat4 instance_transform;
#endif

    uniform mat4 view_projection_matrix;
    uniform mat4 model_matrix;

    out vec4 pos;
    out vec3 local_pos;
    flat out int instance_id;


    void main() {

#if INSTANCED
        mat4 transform = model_matrix * instance_transform;
#else
        mat4 transform = model_matrix;
#endif
        vec3 world_position = (transform * vec4(in_position, 1.0)).xyz;
        gl_Position = view_projection_matrix * vec4(world_position, 1.0);
        instance_id = gl_InstanceID;

        pos = vec4(world_position, 1.0);
        local_pos = in_position;
    }
#endif
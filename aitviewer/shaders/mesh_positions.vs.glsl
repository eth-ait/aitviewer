#version 430

#define INSTANCED 0

#if defined VERTEX_SHADER
    in vec3 in_position;

    uniform mat4 view_projection_matrix;
    uniform mat4 model_matrix;

    out vec3 pos;


#if INSTANCED
    layout(std430, binding=1) buffer instance_data_buf
    {
        mat4 transforms[];
    } instance_data;
#endif

    void main() {

#if INSTANCED
        mat4 transform = model_matrix * instance_data.transforms[gl_InstanceID];
#else
        mat4 transform = model_matrix;
#endif
        vec3 world_position = (transform * vec4(in_position, 1.0)).xyz;
        gl_Position = view_projection_matrix * vec4(world_position, 1.0);

        pos = in_position;
    }
#endif
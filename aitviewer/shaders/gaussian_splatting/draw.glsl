#version 450

// Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos

#include gaussian_splatting/common.glsl

#if defined VERTEX_SHADER

layout(std430, binding=2) buffer in_splat_views
{
    SplatView views[];
} InSplatViews;

layout(std430, binding=3) buffer in_splat_sorted_indices
{
    uint indices[];
} InSplatSortedIndices;

uniform vec2 u_screen_size;

out vec2 position;
out vec4 color;

void main() {
    uint splat_index = gl_InstanceID;
    uint vertex_index = gl_VertexID;

    uint sorted_index = InSplatSortedIndices.indices[splat_index];
    SplatView view = InSplatViews.views[sorted_index];
    if(view.position.w <= 0.0) {
        gl_Position.x = uintBitsToFloat(0x7fc00000); // NaN discards the primitive
    } else {
        vec2 quad = vec2(vertex_index & 1, (vertex_index >> 1) & 1) * 2.0 - 1.0;
        quad *= 2.0;

        vec2 delta = (quad.x * view.axis1 + quad.y * view.axis2) * 2.0 / u_screen_size;
		vec4 p = view.position;
        p.xy += delta * view.position.w;

        color = view.color;
        position = quad;
        gl_Position = p;
    }
}

#elif defined FRAGMENT_SHADER

in vec2 position;
in vec4 color;

out vec4 out_color;

void main() {
    float alpha = clamp(exp(-dot(position, position)) * color.a, 0.0, 1.0);

    if(alpha < 1.0/255.0) {
        discard;
    }
    out_color = vec4(color.rgb, alpha);
}


#endif
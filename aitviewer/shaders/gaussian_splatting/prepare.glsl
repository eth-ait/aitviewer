#version 450

// Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos

#include gaussian_splatting/common.glsl

#define PREPARE_GROUP_SIZE 128

layout (local_size_x = PREPARE_GROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

uniform float u_opacity_scale;
uniform float u_scale2;

uniform uint u_num_splats;
uniform float u_limit;
uniform float u_focal;

uniform mat4 u_world_from_object;
uniform mat4 u_view_from_world;
uniform mat4 u_clip_from_world;

layout(std430, binding=0) buffer in_splat_positions
{
    float positions[];
} InSplatPositions;

layout(std430, binding=1) buffer in_splat_data
{
    SplatData data[];
} InSplatData;

layout(std430, binding=2) buffer out_splat_views
{
    SplatView views[];
} OutSplatViews;

layout(std430, binding=3) buffer out_splat_distances
{
    uint distances[];
} OutSplatDistances;

layout(std430, binding=4) buffer out_splat_indices
{
    uint indices[];
} OutSplatIndices;

uint floatToSortableUint(float v) {
    uint fu = floatBitsToUint(v);
    uint mask = -(int(fu >> 31)) | 0x80000000;
    return fu ^ mask;
}

// from "EWA Splatting" (Zwicker et al 2002) eq. 31
vec3 covariance2D(vec3 world_pos, vec3 cov3d0, vec3 cov3d1)
{
    vec3 view_pos = (u_view_from_world * vec4(world_pos, 1)).xyz;

    view_pos.x = clamp(view_pos.x / view_pos.z, -u_limit, u_limit) * view_pos.z;
    view_pos.y = clamp(view_pos.y / view_pos.z, -u_limit, u_limit) * view_pos.z;

    mat3 J = transpose(mat3(
        u_focal / view_pos.z, 0, -(u_focal * view_pos.x) / (view_pos.z * view_pos.z),
        0, u_focal / view_pos.z, -(u_focal * view_pos.y) / (view_pos.z * view_pos.z),
        0, 0, 0
    ));

    mat3 T = J * mat3(u_view_from_world);

    mat3 V = mat3(
        cov3d0.x, cov3d0.y, cov3d0.z,
        cov3d0.y, cov3d1.x, cov3d1.y,
        cov3d0.z, cov3d1.y, cov3d1.z
    );
    mat3 cov = T * V * transpose(T);

    // Low pass filter to make each splat at least 1px size.
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;

    return vec3(cov[0][0], cov[1][0], cov[1][1]);
}

void axisFromCovariance2D(vec3 cov2d, out vec2 v1, out vec2 v2) {
    float diag1 = cov2d.x, diag2 = cov2d.z, offDiag = cov2d.y;

    float mid = 0.5 * (diag1 + diag2);
    float radius = length(vec2((diag1 - diag2) / 2.0, offDiag));
    float lambda1 = mid + radius;
    float lambda2 = max(mid - radius, 0.1);
    vec2 diagVec = normalize(vec2(offDiag, lambda1 - diag1));
    float maxSize = 4096.0;

    v1 = min(sqrt(2.0 * lambda1), maxSize) * diagVec;
    v2 = min(sqrt(2.0 * lambda2), maxSize) * vec2(diagVec.y, -diagVec.x);
}

Splat loadSplat(uint index) {
    Splat splat;
    splat.position.x = InSplatPositions.positions[index * 3 + 0];
    splat.position.y = InSplatPositions.positions[index * 3 + 1];
    splat.position.z = InSplatPositions.positions[index * 3 + 2];

    SplatData data = InSplatData.data[index];
    splat.color = data.color;
    splat.opacity = data.opacity;
    splat.scale = data.scale;
    splat.rotation = data.rotation;

    return splat;
}

mat3 matrixFromQuaternionScale(vec4 q, vec3 s) {
    mat3 ms = mat3(
        s.x, 0, 0,
        0, s.y, 0,
        0, 0, s.z
    );

    float x = q.x;
    float y = q.y;
    float z = q.z;
    float w = q.w;
    mat3 mr = transpose(mat3(
        1-2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y),
          2*(x*y + w*z), 1-2*(x*x + z*z),   2*(y*z - w*x),
          2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x + y*y)
    ));

    return mr * ms;
}

void main() {
    uint thread_idx = gl_GlobalInvocationID.x;

    // Check if block valid.
    if(thread_idx >= u_num_splats) {
        return;
    }

    Splat splat = loadSplat(thread_idx);

    vec3 world_pos = (u_world_from_object * vec4(splat.position, 1.0)).xyz;
    vec3 view_pos = (u_view_from_world * vec4(world_pos, 1.0)).xyz;
    vec4 clip_pos = u_clip_from_world  * vec4(world_pos, 1.0);

    mat3 rotation = mat3(u_world_from_object) * matrixFromQuaternionScale(splat.rotation, splat.scale);

    vec2 v1 = vec2(0.0);
    vec2 v2 = vec2(0.0);
    if(clip_pos.w > 0) {
        mat3 cov_matrix = rotation * transpose(rotation);
        vec3 cov3d0 = vec3(cov_matrix[0][0], cov_matrix[0][1], cov_matrix[0][2]) * u_scale2;
        vec3 cov3d1 = vec3(cov_matrix[1][1], cov_matrix[1][2], cov_matrix[2][2]) * u_scale2;

        vec3 cov2d = covariance2D(world_pos, cov3d0, cov3d1);
        axisFromCovariance2D(cov2d, v1, v2);

        // vec3 world_view_dir = u_camera_pos - world_pos;
        // vec3 object_view_diw = u_object_from_world * world_view_dir;
        // TODO: SH
    }

    SplatView view;
    view.position = clip_pos;
    view.axis1 = v1;
    view.axis2 = v2;
    view.color = vec4(splat.color, splat.opacity * u_opacity_scale);

    OutSplatViews.views[thread_idx] = view;
    OutSplatDistances.distances[thread_idx] = floatToSortableUint(view_pos.z);
    OutSplatIndices.indices[thread_idx] = thread_idx;
}
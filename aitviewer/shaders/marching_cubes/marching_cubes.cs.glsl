#version 450
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

/* Copyright (C) 2023 Dario Mylonopoulos */

#include marching_cubes/common.glsl
#include marching_cubes/tables.glsl

layout (local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = BLOCK_SIZE_Z) in;

// Number of blocks and blocks array.
layout(std430, binding=0) buffer in_blocks
{
    uint num_blocks;
    uint _groups_y;
    uint _groups_z;
    uint blocks[];
} In;

// Triangles look-up table.
layout(std430, binding=1) buffer in_tri_table
{
    uint triangles[];
} TriangleTable;

// Output vertices array.
layout(std430, binding=2) buffer out_vertices
{
    float vertices[];
} OutVertices;

// Output normals array.
layout(std430, binding=3) buffer out_normals
{
    float normals[];
} OutNormals;

// Output triangle indices array.
layout(std430, binding=4) buffer out_indices
{
    uint indices[];
} OutIndices;

// Atomic counter for number of vertices.
layout(std430, binding=5) buffer out_num_vertices
{
    uint num_vertices;
} OutNumVertices;

// Atomic counter for number of indices in triangles.
layout(std430, binding=6) buffer out_num_indices
{
    uint num_indices;
} OutNumIndices;


// SDF volume.
layout(r32f, location=0) readonly uniform image3D u_volume;

// Level set at which to extract the surface.
uniform float u_level;

// Number of blocks in the X and Y directions.
uniform uvec2 u_blocks;

// Spacing of volume cells.
uniform vec3 u_volume_spacing;

// Maximum number of vertices.
uniform uint u_max_vertices;

// Maximum number of triangle indices.
uniform uint u_max_indices;

// Boolean flag for inverting normals.
uniform bool u_invert_normals;


// OPT: Can reduce shared memory usage by overlapping some of these arrays that are not used at the same time.
shared float s_volume_data[BLOCK_SIZE_Z + 1][BLOCK_SIZE_Y + 1][BLOCK_SIZE_X + 1];
shared vec3 s_gradient[BLOCK_SIZE_Z][BLOCK_SIZE_Y][BLOCK_SIZE_X];

shared uint s_scan_block[BLOCK_SIZE_Z][BLOCK_SIZE_Y][BLOCK_SIZE_X];

shared uint s_scan_vertices[max(gl_NumSubgroups, (BLOCK_TOTAL_SIZE + gl_NumSubgroups - 1) / gl_NumSubgroups)];
shared uint s_scan_triangles[max(gl_NumSubgroups, (BLOCK_TOTAL_SIZE + gl_NumSubgroups - 1) / gl_NumSubgroups)];

shared uint s_vertices_base;
shared uint s_triangles_base;

// Lookup block index in volume from the compacted index.
uvec3 getBlockIdx(uint idx) {
    uint block = In.blocks[idx];
    uvec3 block_idx;
    block_idx.x = block % u_blocks.x;
    block_idx.y = (block % (u_blocks.x * u_blocks.y)) / u_blocks.x;
    block_idx.z = block / (u_blocks.x * u_blocks.y);
    return block_idx;
}

// Return t such that a * (1 - t) + b * t = v
float inverse_lerp(float v, float a, float b) {
    return (v - a) / (b - a);
}

// Lookup vertex index from neighbouring cell in shared memory.
uint get_vertex_index(uvec3 cell_idx, uint cube_index, uint tri_index) {
    uint edge = TriangleTable.triangles[cube_index * 16 + tri_index];
    uvec4 edge_map = table_edge_mapping[edge];

    uint bitmask = s_scan_block[cell_idx.z + edge_map.z][cell_idx.y + edge_map.y][cell_idx.x + edge_map.x];
    uint n_offset = bitmask & 0x1FFF;

    // OPT: This can be optimized by storing a mask in w and using a popcnt or a lookup table.
    // (can also maybe expand the lookup table for edge_mapping with 3 extra bits?)
    uint n_index = 0;
    for(int j = 0; j < edge_map.w; j++) {
        n_index += (bitmask >> (13 + j)) & 1;
    }

    return n_index + n_offset;
}

// Lookup gradients from neighbouring cells and interpolate them.
vec3 getNormal(uvec3 idx_a, uvec3 idx_b, float t) {
    vec3 a = s_gradient[idx_a.z][idx_a.y][idx_a.x];
    vec3 b = s_gradient[idx_b.z][idx_b.y][idx_b.x];
    vec3 n = normalize(mix(a, b, t));

    if(u_invert_normals) {
        n = -n;
    }

    return n;
}

void main() {
    // Index of the block inside the volume.
    uvec3 block_idx = getBlockIdx(gl_WorkGroupID.x);

    // Index of the cell inside the block.
    uvec3 cell_idx = gl_LocalInvocationID;

	// Index of the cell inside the volume.
    uvec3 volume_idx = getVolumeIdx(block_idx, cell_idx);

    uvec3 volume_size = uvec3(imageSize(u_volume));
    float value;
	if(all(lessThan(volume_idx, volume_size))) {
		// Load volume data if inside the volume.
		value = imageLoad(u_volume, ivec3(volume_idx)).r;
	} else {
        value = u_level;
	}

    // Store sdf value of this thread in shared memory.
    s_volume_data[cell_idx.z][cell_idx.y][cell_idx.x] = value;

    // Ensure writes to s_volume_data are visible.
    barrier();

    // Extract sdf values from neighbours for vertices of this cell.
    float f000 = s_volume_data[cell_idx.z    ][cell_idx.y    ][cell_idx.x    ];
	float f001 = s_volume_data[cell_idx.z    ][cell_idx.y    ][cell_idx.x + 1];
    float f010 = s_volume_data[cell_idx.z    ][cell_idx.y + 1][cell_idx.x    ];
	float f011 = s_volume_data[cell_idx.z    ][cell_idx.y + 1][cell_idx.x + 1];
	float f100 = s_volume_data[cell_idx.z + 1][cell_idx.y    ][cell_idx.x    ];
	float f101 = s_volume_data[cell_idx.z + 1][cell_idx.y    ][cell_idx.x + 1];
	float f110 = s_volume_data[cell_idx.z + 1][cell_idx.y + 1][cell_idx.x    ];
	float f111 = s_volume_data[cell_idx.z + 1][cell_idx.y + 1][cell_idx.x  +1];

    // Check if border cells.
    bvec3 cell_border_low = equal(cell_idx, uvec3(0));
    bvec3 cell_border_high = equal(cell_idx, uvec3(BLOCK_SIZE_X - 1, BLOCK_SIZE_Y - 1, BLOCK_SIZE_Z - 1));

    // Check if at volume border or outside.
    bvec3 volume_border_low = equal(volume_idx, uvec3(0));
    bvec3 volume_border_high = greaterThanEqual(volume_idx, volume_size - uvec3(1));

    // Lookup neighbours for computing gradient with finite differences.
    float f00n, f0n0, fn00;

    // Lower neighbours, can be in shared memory or have to be looked up.
    if(cell_border_low.x && !volume_border_low.x) {
        f00n = imageLoad(u_volume, ivec3(volume_idx.x - 1, volume_idx.y, volume_idx.z)).r;
    } else {
        f00n = s_volume_data[cell_idx.z][cell_idx.y][cell_idx.x - 1];
    }
    if(cell_border_low.y && !volume_border_low.y) {
        f0n0 = imageLoad(u_volume, ivec3(volume_idx.x, volume_idx.y - 1, volume_idx.z)).r;
    } else {
        f0n0 = s_volume_data[cell_idx.z][cell_idx.y - 1][cell_idx.x];
    }
    if(cell_border_low.z && !volume_border_low.z) {
        fn00 = imageLoad(u_volume, ivec3(volume_idx.x, volume_idx.y, volume_idx.z - 1)).r;
    } else {
        fn00 = s_volume_data[cell_idx.z - 1][cell_idx.y][cell_idx.x];
    }

    // Higher neighbours have to be looked up if at cell border.
    if(cell_border_high.x && !volume_border_high.x) {
        f001 = imageLoad(u_volume, ivec3(volume_idx.x + 1, volume_idx.y, volume_idx.z)).r;
    }
    if(cell_border_high.y && !volume_border_high.y) {
        f010 = imageLoad(u_volume, ivec3(volume_idx.x, volume_idx.y + 1, volume_idx.z)).r;
    }
    if(cell_border_high.z && !volume_border_high.z) {
        f100 = imageLoad(u_volume, ivec3(volume_idx.x, volume_idx.y, volume_idx.z + 1)).r;
    }


    // Compute gradient with central differences for internal cells
    // and using forward/backward differences at volume boundaries.
    vec3 g;
    if(volume_border_high.x) {
        g.x = (f000 - f00n) * u_volume_spacing.x * 2.0;
    } else if(volume_border_low.x) {
        g.x = (f001 - f000) * u_volume_spacing.x * 2.0;
    } else {
        g.x = (f001 - f00n) * u_volume_spacing.x;
    }

    if(volume_border_high.y) {
        g.y = (f000 - f0n0) * u_volume_spacing.y * 2.0;
    } else if(volume_border_low.y) {
        g.y = (f010 - f000) * u_volume_spacing.y * 2.0;
    } else {
        g.y = (f010 - f0n0) * u_volume_spacing.y;
    }

    if(volume_border_high.z) {
        g.z = (f000 - fn00) * u_volume_spacing.z * 2.0;
    } else if(volume_border_low.z) {
        g.z = (f100 - f000) * u_volume_spacing.z * 2.0;
    } else {
        g.z = (f100 - fn00) * u_volume_spacing.z;
    }

    // Store gradient in shared memory.
    s_gradient[cell_idx.z][cell_idx.y][cell_idx.x] = g;

    // Classify cell comparing with surface level value.
    uint field[8];
    field[0] = uint(f000 > u_level);
    field[1] = uint(f001 > u_level);
    field[2] = uint(f011 > u_level);
    field[3] = uint(f010 > u_level);
    field[4] = uint(f100 > u_level);
    field[5] = uint(f101 > u_level);
    field[6] = uint(f111 > u_level);
    field[7] = uint(f110 > u_level);

    // Pack booleans into an 8 bit integer for the lookup table.
    uint cube_index =
        field[0] << 0 |
        field[1] << 1 |
        field[2] << 2 |
        field[3] << 3 |
        field[4] << 4 |
        field[5] << 5 |
        field[6] << 6 |
        field[7] << 7;

    // Compute which edges of this cell generate a vertex.
    uvec3 edge_index;
    edge_index.x = field[0] ^ field[1];
    edge_index.y = field[0] ^ field[3];
    edge_index.z = field[0] ^ field[4];

    // Discard vertices that go outside the cell or outside the volume.
    bvec3 border = cell_border_high || volume_border_high;

	if(border.x) {
		edge_index.x = 0;
    }

	if(border.y) {
		edge_index.y = 0;
    }

	if(border.z) {
		edge_index.z = 0;
    }

    // Compute number of vertices and triangles required by this cell.
    uint num_vertices = edge_index.x + edge_index.y + edge_index.z;
    uint num_triangles = 0;

    // Border cells might generate vertices but always generate 0 triangles.
    if(!any(border)) {
        // OPT: store this table already divided by 3.
        num_triangles = table_num_vertices[cube_index] / 3;
    }


    // Perform per-block exclusive scan of number of vertices and number of triangles
    // that will be created by this block.

    // Per subgroup scan.
    uint vertex_index = subgroupExclusiveAdd(num_vertices);
    uint triangle_index = subgroupExclusiveAdd(num_triangles);

    // Store highest value for each subgroup in shared memory.
    uint highest_in_subgroup = subgroupBallotFindMSB(subgroupBallot(true));
    if(gl_SubgroupInvocationID == highest_in_subgroup) {
        s_scan_vertices[gl_SubgroupID] = vertex_index + num_vertices;
        s_scan_triangles[gl_SubgroupID] = triangle_index + num_triangles;
    }

    barrier();

    // Subgroup 0 performs prefix sum across the top values of each block.
    if(gl_SubgroupID == 0) {
        // TODO: for now we assume we can do this scan on block values in a single step.
        // ideally we need to loop here.
        // Since our block size is 512 this would only be a problem if 512 / subgroupSize > subgroupSize
        // thus if subgroupSize ** 2 < 512 <=> subgroupSize < 32
        uint sum_vertices = 0;
        uint sum_triangles = 0;
        if(gl_SubgroupInvocationID < gl_NumSubgroups) {
            sum_vertices = s_scan_vertices[gl_SubgroupInvocationID];
            sum_triangles = s_scan_triangles[gl_SubgroupInvocationID];
        }

        s_scan_vertices[gl_SubgroupInvocationID]  = subgroupInclusiveAdd(sum_vertices);
        s_scan_triangles[gl_SubgroupInvocationID] = subgroupInclusiveAdd(sum_triangles);
    }

    barrier();

    // Add scan of previous subgroup to all values. Subgroup 0 skips this because has no previous subgroup.
    if(gl_SubgroupID > 0) {
        vertex_index += s_scan_vertices[gl_SubgroupID - 1];
        triangle_index += s_scan_triangles[gl_SubgroupID - 1];
    }

    // Last thread in the block increments atomic counters for triangles and vertices;
    if(gl_LocalInvocationIndex == (BLOCK_TOTAL_SIZE - 1)) {
        s_vertices_base = atomicAdd(OutNumVertices.num_vertices, vertex_index + num_vertices);
        s_triangles_base = atomicAdd(OutNumIndices.num_indices, (triangle_index + num_triangles) * 3);
    }

    // Write out vertex information for other threads in the group.
    s_scan_block[cell_idx.z][cell_idx.y][cell_idx.x] = (edge_index.z << 15) | (edge_index.y << 14) | (edge_index.x << 13) | vertex_index;

    // Ensure writes to s_scan_block, s_vertices_offset and s_triangles_offset are visible to the rest of the group.
    barrier();

    // Compute offsets into output arrays.
    uint vertices_base = s_vertices_base;
    uint triangles_base = s_triangles_base;

    uint global_vertex_index = vertices_base + vertex_index;
    uint global_triangle_index = triangles_base + triangle_index * 3;

    // Return early if out of bounds in the vertices or indices array.
    if(global_vertex_index + num_vertices >= u_max_vertices ||
       global_triangle_index + num_triangles * 3 >= u_max_indices) {
        return;
    }

    // Output vertices.
    if(edge_index.x > 0) {
        float t = inverse_lerp(u_level, f000, f001);

        vec3 v = vec3(volume_idx.x + t, volume_idx.y, volume_idx.z) * u_volume_spacing;
        OutVertices.vertices[global_vertex_index * 3 + 0] = v.x;
        OutVertices.vertices[global_vertex_index * 3 + 1] = v.y;
        OutVertices.vertices[global_vertex_index * 3 + 2] = v.z;

        vec3 n = getNormal(cell_idx, cell_idx + uvec3(1, 0, 0), t);
        OutNormals.normals[global_vertex_index * 3 + 0] = n.x;
        OutNormals.normals[global_vertex_index * 3 + 1] = n.y;
        OutNormals.normals[global_vertex_index * 3 + 2] = n.z;

        global_vertex_index += 1;
    }

    if(edge_index.y > 0) {
        float t = inverse_lerp(u_level, f000, f010);

        vec3 v = vec3(volume_idx.x, volume_idx.y + t, volume_idx.z) * u_volume_spacing;
        OutVertices.vertices[global_vertex_index * 3 + 0] = v.x;
        OutVertices.vertices[global_vertex_index * 3 + 1] = v.y;
        OutVertices.vertices[global_vertex_index * 3 + 2] = v.z;

        vec3 n = getNormal(cell_idx, cell_idx + uvec3(0, 1, 0), t);
        OutNormals.normals[global_vertex_index * 3 + 0] = n.x;
        OutNormals.normals[global_vertex_index * 3 + 1] = n.y;
        OutNormals.normals[global_vertex_index * 3 + 2] = n.z;

        global_vertex_index += 1;
    }

    if(edge_index.z > 0) {
        float t = inverse_lerp(u_level, f000, f100);

        vec3 v = vec3(volume_idx.x, volume_idx.y, volume_idx.z + t) * u_volume_spacing;
        OutVertices.vertices[global_vertex_index * 3 + 0] = v.x;
        OutVertices.vertices[global_vertex_index * 3 + 1] = v.y;
        OutVertices.vertices[global_vertex_index * 3 + 2] = v.z;

        vec3 n = getNormal(cell_idx, cell_idx + uvec3(0, 0, 1), t);
        OutNormals.normals[global_vertex_index * 3 + 0] = n.x;
        OutNormals.normals[global_vertex_index * 3 + 1] = n.y;
        OutNormals.normals[global_vertex_index * 3 + 2] = n.z;
    }

    // Output triangles.
    for(uint i = 0; i < num_triangles; i++) {
        uint i0 = vertices_base + get_vertex_index(cell_idx, cube_index, i * 3 + 0);
        uint i1 = vertices_base + get_vertex_index(cell_idx, cube_index, i * 3 + 1);
        uint i2 = vertices_base + get_vertex_index(cell_idx, cube_index, i * 3 + 2);

        OutIndices.indices[global_triangle_index + 0] = i0;
        OutIndices.indices[global_triangle_index + 1] = i1;
        OutIndices.indices[global_triangle_index + 2] = i2;
        global_triangle_index += 3;
    }
}
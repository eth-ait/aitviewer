#version 450
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

/* Copyright (C) 2023 Dario Mylonopoulos */


#include marching_cubes/common.glsl

layout (local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = BLOCK_SIZE_Z) in;

// Output array of valid blocks. Each element in the array is set to 1 if the block is valid and 0 otherwise.
layout(std430, binding=0) buffer out_valid
{
    uint valid[];
} Out;

// SDF volume.
layout(r32f, location=0) readonly uniform image3D volume;

// Level set for which to check for surface.
uniform float u_level;

// Shared memory array of number of bit set in each ballot.
shared uint s_ballot[gl_NumSubgroups];

// Shared memory array of number of threads in the subgroup that are inside the bounds of the volume.
shared uint s_valid[gl_NumSubgroups];


void main() {
	// Index of the block inside the volume.
	uvec3 block_idx = gl_WorkGroupID;

	// Index of the cell inside the volume.
    uvec3 volume_idx = getVolumeIdx(block_idx, gl_LocalInvocationID);

	// Index of the thread inside the group.
	uint thread_idx = gl_LocalInvocationIndex;


	float value;
	uint valid;
	if(all(lessThan(volume_idx, uvec3(imageSize(volume))))) {
		// Load volume data if inside the volume.
		value = imageLoad(volume, ivec3(volume_idx)).r;
		valid = 1;
	} else {
		// Set the value to level to ensure that this bit is not set in the ballot.
		value = u_level;
		valid = 0;
	}

	// Count how many invocations are inside the grid.
	uint num_valid = subgroupAdd(valid);

	// Compute ballot of values below surface level.
	uvec4 ballot = subgroupBallot(value > u_level);
	if(subgroupElect()) {
		uint count = subgroupBallotBitCount(ballot);
		s_ballot[gl_SubgroupID] = count;
		s_valid[gl_SubgroupID] = num_valid;
	}

	// Ensure all writes to shared memory are visible.
	barrier();

	// Thread 0 in the block checks if the whole block can be skipped.
	if(thread_idx == 0) {
		// Accumulate ballot results of all threads and total number of threads in the group inside the volume.
		uint total_active = 0;
		uint total_valid = 0;
		for(int i = 0; i < gl_NumSubgroups; i++) {
			total_active += uint(s_ballot[i]);
			total_valid += uint(s_valid[i]);
		}

		// Store if this block intersect the surface.
		// The block is completely inside if all points are inside or if all the valid points
		// are inside or all the valid points are outside.
		uint idx = gl_NumWorkGroups.y *  gl_NumWorkGroups.x * block_idx.z + gl_NumWorkGroups.x * block_idx.y + block_idx.x;
		Out.valid[idx] = uint(!(total_active == 0 || total_active == total_valid));
	}
}

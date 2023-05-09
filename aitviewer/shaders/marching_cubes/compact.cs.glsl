#version 450
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

/* Copyright (C) 2023 Dario Mylonopoulos */


#define COMPACT_GROUP_SIZE 128

layout (local_size_x = COMPACT_GROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

// Number of input blocks.
uniform uint u_num_blocks;

// Input blocks, valid = 1 means the block is kept and valid = 0 means that the block is discarded.
layout(std430, binding=0) buffer in_valid
{
    uint valid[];
} In;

// Number of blocks and output blocks array. Indices of valid blocks are compacted and written to this array.
layout(std430, binding=1) buffer out_blocks
{
    uint num_blocks;
    uint _groups_y;
    uint _groups_z;
    uint blocks[];
} Out;

void main() {
    uint thread_idx = gl_GlobalInvocationID.x;

    // Check if block valid.
    uint valid = 0;
    if(thread_idx < u_num_blocks) {
        valid = In.valid[thread_idx];
    }

    // Compute ballot of valid.
    uvec4 ballot = subgroupBallot(bool(valid));

    // Exclusive scan of ballot.
    uint local_index = subgroupBallotExclusiveBitCount(ballot);

    // Highest index of valid in subgroup.
    uint highest = subgroupBallotFindMSB(ballot);

    uint global_index = 0;
    if (highest == gl_SubgroupInvocationID) {
        // Increment number of blocks for each valid block in the subgroup.
        uint local_size = local_index + valid;
        global_index = atomicAdd(Out.num_blocks, local_size);
    }

    // Share global_index across subgroup.
    global_index = subgroupMax(global_index);
    if(valid != 0) {
        // Write out block index.
        Out.blocks[global_index + local_index] = thread_idx;
    }
}
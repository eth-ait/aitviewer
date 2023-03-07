/* Copyright (C) 2023 Dario Mylonopoulos */

#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8
#define BLOCK_SIZE_Z 8

#define BLOCK_TOTAL_SIZE (BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z)

// Map from block index and local cell index in the block to cell index in the volume.
uvec3 getVolumeIdx(uvec3 block_idx, uvec3 cell_idx) {
    return block_idx * uvec3(BLOCK_SIZE_X - 1, BLOCK_SIZE_Y - 1, BLOCK_SIZE_Z - 1) + cell_idx;
}

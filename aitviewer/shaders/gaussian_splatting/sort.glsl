#version 450
#extension GL_KHR_shader_subgroup_quad : require
#extension GL_ARB_shader_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require

// Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos

// Adapted from parallelsort algorithm in FidelityFX-SDK
// https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/tree/main
//
// FidelityFX-SDK License
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Entry point defines
#define ENTRY_PARALLEL_SORT_COUNT 0
#define ENTRY_PARALLEL_SORT_SCAN_REDUCE 0
#define ENTRY_PARALLEL_SORT_SCAN 0
#define ENTRY_PARALLEL_SORT_SCAN_ADD 0
#define ENTRY_PARALLEL_SORT_SCATTER 0

// Config defines
#define FFX_PARALLELSORT_COPY_VALUE

// BINDINGS DEFINES
#define FFX_PARALLELSORT_BIND_UAV_SOURCE_KEYS 0
#define FFX_PARALLELSORT_BIND_UAV_DEST_KEYS 1
#define FFX_PARALLELSORT_BIND_UAV_SOURCE_PAYLOADS 2
#define FFX_PARALLELSORT_BIND_UAV_DEST_PAYLOADS 3
#define FFX_PARALLELSORT_BIND_UAV_SUM_TABLE 4
#define FFX_PARALLELSORT_BIND_UAV_REDUCE_TABLE 5
#define FFX_PARALLELSORT_BIND_UAV_SCAN_SOURCE 6
#define FFX_PARALLELSORT_BIND_UAV_SCAN_DEST 7
#define FFX_PARALLELSORT_BIND_UAV_SCAN_SCRATCH 8

// --- GLSL Defines
#define FFX_GLSL
#define FFX_GROUPSHARED shared
#define FFX_GROUP_MEMORY_BARRIER() groupMemoryBarrier(); barrier()

#define FfxUInt32    uint
#define FfxInt32     int

#define FFX_ATOMIC_ADD(x, y) atomicAdd(x, y)

// --- Uniform buffer
layout(binding = 0, std140) uniform cbParallelSort_t
{
    FfxUInt32 numKeys;
    FfxUInt32 numBlocksPerThreadGroup;
    FfxUInt32 numThreadGroups;
    FfxUInt32 numThreadGroupsWithAdditionalBlocks;
    FfxUInt32 numReduceThreadgroupPerBin;
    FfxUInt32 numScanValues;
    FfxUInt32 shiftBit;
    FfxUInt32 padding;
} u_constants;

uint FfxNumKeys() { return u_constants.numKeys; }
uint FfxNumBlocksPerThreadGroup() { return u_constants.numBlocksPerThreadGroup; }
uint FfxNumThreadGroups() { return u_constants.numThreadGroups; }
uint FfxNumThreadGroupsWithAdditionalBlocks() { return u_constants.numThreadGroupsWithAdditionalBlocks; }
uint FfxNumReduceThreadgroupPerBin() { return u_constants.numReduceThreadgroupPerBin; }
uint FfxNumScanValues() { return u_constants.numScanValues; }
uint FfxShiftBit() { return u_constants.shiftBit; }

// --- Buffers
layout(binding = FFX_PARALLELSORT_BIND_UAV_SOURCE_KEYS, std430)  coherent buffer ParallelSortSrcKeys_t { uint source_keys[]; } rw_source_keys;
layout(binding = FFX_PARALLELSORT_BIND_UAV_DEST_KEYS, std430)  coherent buffer ParallelSortDstKeys_t { uint dest_keys[]; } rw_dest_keys;
layout(binding = FFX_PARALLELSORT_BIND_UAV_SOURCE_PAYLOADS, std430)  coherent buffer ParallelSortSrcPayload_t { uint source_payloads[]; } rw_source_payloads;
layout(binding = FFX_PARALLELSORT_BIND_UAV_DEST_PAYLOADS, std430)  coherent buffer ParallelSortDstPayload_t { uint dest_payloads[]; } rw_dest_payloads;
layout(binding = FFX_PARALLELSORT_BIND_UAV_SUM_TABLE, std430)  coherent buffer ParallelSortSumTable_t { uint sum_table[]; } rw_sum_table;
layout(binding = FFX_PARALLELSORT_BIND_UAV_REDUCE_TABLE, std430)  coherent buffer ParallelSortReduceTable_t { uint reduce_table[]; } rw_reduce_table;
layout(binding = FFX_PARALLELSORT_BIND_UAV_SCAN_SOURCE, std430)  coherent buffer ParallelSortScanSrc_t { uint scan_source[]; } rw_scan_source;
layout(binding = FFX_PARALLELSORT_BIND_UAV_SCAN_DEST, std430)  coherent buffer ParallelSortScanDst_t { uint scan_dest[]; } rw_scan_dest;
layout(binding = FFX_PARALLELSORT_BIND_UAV_SCAN_SCRATCH, std430)  coherent buffer ParallelSortScanScratch_t { uint scan_scratch[]; } rw_scan_scratch;

FfxUInt32 LoadSourceKey(FfxUInt32 index)
{
    return rw_source_keys.source_keys[index];
}

void StoreDestKey(FfxUInt32 index, FfxUInt32 value)
{
    rw_dest_keys.dest_keys[index] = value;
}

FfxUInt32 LoadSourcePayload(FfxUInt32 index)
{
    return rw_source_payloads.source_payloads[index];
}

void StoreDestPayload(FfxUInt32 index, FfxUInt32 value)
{
    rw_dest_payloads.dest_payloads[index] = value;
}

FfxUInt32 LoadSumTable(FfxUInt32 index)
{
    return rw_sum_table.sum_table[index];
}

void StoreSumTable(FfxUInt32 index, FfxUInt32 value)
{
    rw_sum_table.sum_table[index] = value;
}

void StoreReduceTable(FfxUInt32 index, FfxUInt32 value)
{
    rw_reduce_table.reduce_table[index] = value;
}

FfxUInt32 LoadScanSource(FfxUInt32 index)
{
    return rw_scan_source.scan_source[index];
}

void StoreScanDest(FfxUInt32 index, FfxUInt32 value)
{
    rw_scan_dest.scan_dest[index] = value;
}

FfxUInt32 LoadScanScratch(FfxUInt32 index)
{
    return rw_scan_scratch.scan_scratch[index];
}

FfxUInt32 FfxLoadKey(FfxUInt32 index)
{
    return LoadSourceKey(index);
}

void FfxStoreKey(FfxUInt32 index, FfxUInt32 value)
{
    StoreDestKey(index, value);
}

FfxUInt32 FfxLoadPayload(FfxUInt32 index)
{
    return LoadSourcePayload(index);
}

void FfxStorePayload(FfxUInt32 index, FfxUInt32 value)
{
    StoreDestPayload(index, value);
}

FfxUInt32 FfxLoadSum(FfxUInt32 index)
{
    return LoadSumTable(index);
}

void FfxStoreSum(FfxUInt32 index, FfxUInt32 value)
{
    StoreSumTable(index, value);
}

void FfxStoreReduce(FfxUInt32 index, FfxUInt32 value)
{
    StoreReduceTable(index, value);
}

FfxUInt32 FfxLoadScanSource(FfxUInt32 index)
{
    return LoadScanSource(index);
}

void FfxStoreScanDest(FfxUInt32 index, FfxUInt32 value)
{
    StoreScanDest(index, value);
}

FfxUInt32 FfxLoadScanScratch(FfxUInt32 index)
{
    return LoadScanScratch(index);
}


// --- Implementation

/// @defgroup FfxGPUParallelSort FidelityFX Parallel Sort
/// FidelityFX Parallel Sort GPU documentation
///
/// @ingroup FfxGPUEffects

/// The number of bits we are sorting per pass.
/// Changing this value requires
/// internal changes in LDS distribution and count,
/// reduce, scan, and scatter passes
///
/// @ingroup FfxGPUParallelSort
#define FFX_PARALLELSORT_SORT_BITS_PER_PASS		    4

/// The number of bins used for the counting phase
/// of the algorithm. Changing this value requires
/// internal changes in LDS distribution and count,
/// reduce, scan, and scatter passes
///
/// @ingroup FfxGPUParallelSort
#define	FFX_PARALLELSORT_SORT_BIN_COUNT			    (1 << FFX_PARALLELSORT_SORT_BITS_PER_PASS)

/// The number of elements dealt with per running
/// thread
///
/// @ingroup FfxGPUParallelSort
#define FFX_PARALLELSORT_ELEMENTS_PER_THREAD	    4

/// The number of threads to execute in parallel
/// for each dispatch group
///
/// @ingroup FfxGPUParallelSort
#define FFX_PARALLELSORT_THREADGROUP_SIZE		    128

/// The maximum number of thread groups to run
/// in parallel. Modifying this value can help
/// or hurt GPU occupancy, but is very hardware
/// class specific
///
/// @ingroup FfxGPUParallelSort
#define FFX_PARALLELSORT_MAX_THREADGROUPS_TO_RUN    800

FFX_GROUPSHARED FfxUInt32 gs_FFX_PARALLELSORT_Histogram[FFX_PARALLELSORT_THREADGROUP_SIZE * FFX_PARALLELSORT_SORT_BIN_COUNT];
void ffxParallelSortCountUInt(FfxUInt32 localID, FfxUInt32 groupID, FfxUInt32 ShiftBit)
{
    // Start by clearing our local counts in LDS
    for (FfxInt32 i = 0; i < FFX_PARALLELSORT_SORT_BIN_COUNT; i++)
        gs_FFX_PARALLELSORT_Histogram[(i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID] = 0;

    // Wait for everyone to catch up
    FFX_GROUP_MEMORY_BARRIER();

    // Data is processed in blocks, and how many we process can changed based on how much data we are processing
    // versus how many thread groups we are processing with
    FfxInt32 BlockSize = FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE;

    // Figure out this thread group's index into the block data (taking into account thread groups that need to do extra reads)
    FfxUInt32 NumBlocksPerThreadGroup = FfxNumBlocksPerThreadGroup();
    FfxUInt32 NumThreadGroups = FfxNumThreadGroups();
    FfxUInt32 NumThreadGroupsWithAdditionalBlocks = FfxNumThreadGroupsWithAdditionalBlocks();
    FfxUInt32 NumKeys = FfxNumKeys();

    FfxUInt32 ThreadgroupBlockStart = (BlockSize * NumBlocksPerThreadGroup * groupID);
    FfxUInt32 NumBlocksToProcess = NumBlocksPerThreadGroup;

    if (groupID >= NumThreadGroups - NumThreadGroupsWithAdditionalBlocks)
    {
        ThreadgroupBlockStart += (groupID - (NumThreadGroups - NumThreadGroupsWithAdditionalBlocks)) * BlockSize;
        NumBlocksToProcess++;
    }

    // Get the block start index for this thread
    FfxUInt32 BlockIndex = ThreadgroupBlockStart + localID;

    // Count value occurrence
    for (FfxUInt32 BlockCount = 0; BlockCount < NumBlocksToProcess; BlockCount++, BlockIndex += BlockSize)
    {
        FfxUInt32 DataIndex = BlockIndex;

        // Pre-load the key values in order to hide some of the read latency
        FfxUInt32 srcKeys[FFX_PARALLELSORT_ELEMENTS_PER_THREAD];
        srcKeys[0] = FfxLoadKey(DataIndex);
        srcKeys[1] = FfxLoadKey(DataIndex + FFX_PARALLELSORT_THREADGROUP_SIZE);
        srcKeys[2] = FfxLoadKey(DataIndex + (FFX_PARALLELSORT_THREADGROUP_SIZE * 2));
        srcKeys[3] = FfxLoadKey(DataIndex + (FFX_PARALLELSORT_THREADGROUP_SIZE * 3));

        for (FfxUInt32 i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; i++)
        {
            if (DataIndex < NumKeys)
            {
                FfxUInt32 localKey = (srcKeys[i] >> ShiftBit) & 0xf;
                FFX_ATOMIC_ADD(gs_FFX_PARALLELSORT_Histogram[(localKey * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID], 1);
                DataIndex += FFX_PARALLELSORT_THREADGROUP_SIZE;
            }
        }
    }

    // Even though our LDS layout guarantees no collisions, our thread group size is greater than a wave
    // so we need to make sure all thread groups are done counting before we start tallying up the results
    FFX_GROUP_MEMORY_BARRIER();

    if (localID < FFX_PARALLELSORT_SORT_BIN_COUNT)
    {
        FfxUInt32 sum = 0;
        for (FfxInt32 i = 0; i < FFX_PARALLELSORT_THREADGROUP_SIZE; i++)
        {
            sum += gs_FFX_PARALLELSORT_Histogram[localID * FFX_PARALLELSORT_THREADGROUP_SIZE + i];
        }
        FfxStoreSum(localID * NumThreadGroups + groupID, sum);
    }
}

FFX_GROUPSHARED FfxUInt32 gs_FFX_PARALLELSORT_LDSSums[FFX_PARALLELSORT_THREADGROUP_SIZE];
FfxUInt32 ffxParallelSortThreadgroupReduce(FfxUInt32 localSum, FfxUInt32 localID)
{
    // Do wave local reduce
#if defined(FFX_HLSL)
    FfxUInt32 waveReduced = WaveActiveSum(localSum);

    // First lane in a wave writes out wave reduction to LDS (this accounts for num waves per group greater than HW wave size)
    // Note that some hardware with very small HW wave sizes (i.e. <= 8) may exhibit issues with this algorithm, and have not been tested.
    FfxUInt32 waveID = localID / WaveGetLaneCount();
    if (WaveIsFirstLane())
        gs_FFX_PARALLELSORT_LDSSums[waveID] = waveReduced;

    // Wait for everyone to catch up
    FFX_GROUP_MEMORY_BARRIER();

    // First wave worth of threads sum up wave reductions
    if (!waveID)
        waveReduced = WaveActiveSum((localID < FFX_PARALLELSORT_THREADGROUP_SIZE / WaveGetLaneCount()) ? gs_FFX_PARALLELSORT_LDSSums[localID] : 0);

#elif defined(FFX_GLSL)

    FfxUInt32 waveReduced = subgroupAdd(localSum);

    // First lane in a wave writes out wave reduction to LDS (this accounts for num waves per group greater than HW wave size)
    // Note that some hardware with very small HW wave sizes (i.e. <= 8) may exhibit issues with this algorithm, and have not been tested.
    FfxUInt32 waveID = localID / gl_SubGroupSizeARB;
    if (subgroupElect())
        gs_FFX_PARALLELSORT_LDSSums[waveID] = waveReduced;

    // Wait for everyone to catch up
    FFX_GROUP_MEMORY_BARRIER();

    // First wave worth of threads sum up wave reductions
    if (waveID == 0)
        waveReduced = subgroupAdd((localID < FFX_PARALLELSORT_THREADGROUP_SIZE / gl_SubGroupSizeARB) ? gs_FFX_PARALLELSORT_LDSSums[localID] : 0);

#endif // #if defined(FFX_HLSL)

    // Returned the reduced sum
    return waveReduced;
}

void ffxParallelSortReduceCount(FfxUInt32 localID, FfxUInt32 groupID)
{
    FfxUInt32 NumReduceThreadgroupPerBin = FfxNumReduceThreadgroupPerBin();
    FfxUInt32 NumThreadGroups = FfxNumThreadGroups();

    // Figure out what bin data we are reducing
    FfxUInt32 BinID = groupID / NumReduceThreadgroupPerBin;
    FfxUInt32 BinOffset = BinID * NumThreadGroups;

    // Get the base index for this thread group
    FfxUInt32 BaseIndex = (groupID % NumReduceThreadgroupPerBin) * FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE;

    // Calculate partial sums for entries this thread reads in
    FfxUInt32 threadgroupSum = 0;
    for (FfxUInt32 i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; ++i)
    {
        FfxUInt32 DataIndex = BaseIndex + (i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID;
        threadgroupSum += (DataIndex < NumThreadGroups) ? FfxLoadSum(BinOffset + DataIndex) : 0;
    }

    // Reduce across the entirety of the thread group
    threadgroupSum = ffxParallelSortThreadgroupReduce(threadgroupSum, localID);

    // First thread of the group writes out the reduced sum for the bin
    if (localID == 0)
        FfxStoreReduce(groupID, threadgroupSum);

    // What this will look like in the reduced table is:
    //	[ [bin0 ... bin0] [bin1 ... bin1] ... ]
}

FfxUInt32 ffxParallelSortBlockScanPrefix(FfxUInt32 localSum, FfxUInt32 localID)
{
#if defined(FFX_HLSL)

    // Do wave local scan-prefix
    FfxUInt32 wavePrefixed = WavePrefixSum(localSum);

    // Since we are dealing with thread group sizes greater than HW wave size, we need to account for what wave we are in.
    FfxUInt32 waveID = localID / WaveGetLaneCount();
    FfxUInt32 laneID = WaveGetLaneIndex();

    // Last element in a wave writes out partial sum to LDS
    if (laneID == WaveGetLaneCount() - 1)
        gs_FFX_PARALLELSORT_LDSSums[waveID] = wavePrefixed + localSum;

    // Wait for everyone to catch up
    FFX_GROUP_MEMORY_BARRIER();

    // First wave prefixes partial sums
    if (!waveID)
        gs_FFX_PARALLELSORT_LDSSums[localID] = WavePrefixSum(gs_FFX_PARALLELSORT_LDSSums[localID]);

    // Wait for everyone to catch up
    FFX_GROUP_MEMORY_BARRIER();

    // Add the partial sums back to each wave prefix
    wavePrefixed += gs_FFX_PARALLELSORT_LDSSums[waveID];

#elif defined(FFX_GLSL)

    // Do wave local scan-prefix
    FfxUInt32 wavePrefixed = subgroupExclusiveAdd(localSum);

    // Since we are dealing with thread group sizes greater than HW wave size, we need to account for what wave we are in.
    FfxUInt32 waveID = localID / gl_SubGroupSizeARB;
    FfxUInt32 laneID = gl_SubGroupInvocationARB;

    // Last element in a wave writes out partial sum to LDS
    if (laneID == gl_SubGroupSizeARB - 1)
        gs_FFX_PARALLELSORT_LDSSums[waveID] = wavePrefixed + localSum;

    // Wait for everyone to catch up
    FFX_GROUP_MEMORY_BARRIER();

    // First wave prefixes partial sums
    if (waveID == 0)
        gs_FFX_PARALLELSORT_LDSSums[localID] = subgroupExclusiveAdd(gs_FFX_PARALLELSORT_LDSSums[localID]);

    // Wait for everyone to catch up
    FFX_GROUP_MEMORY_BARRIER();

    // Add the partial sums back to each wave prefix
    wavePrefixed += gs_FFX_PARALLELSORT_LDSSums[waveID];

#endif // #if defined(FFX_HLSL)

    return wavePrefixed;
}

// This is to transform uncoalesced loads into coalesced loads and
// then scattered loads from LDS
FFX_GROUPSHARED FfxUInt32 gs_FFX_PARALLELSORT_LDS[FFX_PARALLELSORT_ELEMENTS_PER_THREAD][FFX_PARALLELSORT_THREADGROUP_SIZE];
void ffxParallelSortScanPrefix(FfxUInt32 numValuesToScan, FfxUInt32 localID, FfxUInt32 groupID, FfxUInt32 BinOffset, FfxUInt32 BaseIndex, bool AddPartialSums)
{
    // Perform coalesced loads into LDS
    for (FfxUInt32 i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; i++)
    {
        FfxUInt32 DataIndex = BaseIndex + (i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID;

        FfxUInt32 col = ((i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID) / FFX_PARALLELSORT_ELEMENTS_PER_THREAD;
        FfxUInt32 row = ((i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID) % FFX_PARALLELSORT_ELEMENTS_PER_THREAD;
        gs_FFX_PARALLELSORT_LDS[row][col] = (DataIndex < numValuesToScan) ? FfxLoadScanSource(BinOffset + DataIndex) : 0;
    }

    // Wait for everyone to catch up
    FFX_GROUP_MEMORY_BARRIER();

    FfxUInt32 threadgroupSum = 0;
    // Calculate the local scan-prefix for current thread
    for (FfxUInt32 i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; i++)
    {
        FfxUInt32 tmp = gs_FFX_PARALLELSORT_LDS[i][localID];
        gs_FFX_PARALLELSORT_LDS[i][localID] = threadgroupSum;
        threadgroupSum += tmp;
    }

    // Scan prefix partial sums
    threadgroupSum = ffxParallelSortBlockScanPrefix(threadgroupSum, localID);

    // Add reduced partial sums if requested
    FfxUInt32 partialSum = 0;
    if (AddPartialSums)
    {
        // Partial sum additions are a little special as they are tailored to the optimal number of
        // thread groups we ran in the beginning, so need to take that into account
        partialSum = FfxLoadScanScratch(groupID);
    }

    // Add the block scanned-prefixes back in
    for (FfxUInt32 i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; i++)
        gs_FFX_PARALLELSORT_LDS[i][localID] += threadgroupSum;

    // Wait for everyone to catch up
    FFX_GROUP_MEMORY_BARRIER();

    // Perform coalesced writes to scan dst
    for (FfxUInt32 i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; i++)
    {
        FfxUInt32 DataIndex = BaseIndex + (i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID;

        FfxUInt32 col = ((i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID) / FFX_PARALLELSORT_ELEMENTS_PER_THREAD;
        FfxUInt32 row = ((i * FFX_PARALLELSORT_THREADGROUP_SIZE) + localID) % FFX_PARALLELSORT_ELEMENTS_PER_THREAD;

        if (DataIndex < numValuesToScan)
            FfxStoreScanDest(BinOffset + DataIndex, gs_FFX_PARALLELSORT_LDS[row][col] + partialSum);
    }
}

// Offset cache to avoid loading the offsets all the time
FFX_GROUPSHARED FfxUInt32 gs_FFX_PARALLELSORT_BinOffsetCache[FFX_PARALLELSORT_THREADGROUP_SIZE];
// Local histogram for offset calculations
FFX_GROUPSHARED FfxUInt32 gs_FFX_PARALLELSORT_LocalHistogram[FFX_PARALLELSORT_SORT_BIN_COUNT];
// Scratch area for algorithm
FFX_GROUPSHARED FfxUInt32 gs_FFX_PARALLELSORT_LDSScratch[FFX_PARALLELSORT_THREADGROUP_SIZE];

void ffxParallelSortScatterUInt(FfxUInt32 localID, FfxUInt32 groupID, FfxUInt32 ShiftBit)
{
    FfxUInt32 NumBlocksPerThreadGroup = FfxNumBlocksPerThreadGroup();
    FfxUInt32 NumThreadGroups = FfxNumThreadGroups();
    FfxUInt32 NumThreadGroupsWithAdditionalBlocks = FfxNumThreadGroupsWithAdditionalBlocks();
    FfxUInt32 NumKeys = FfxNumKeys();

    // Load the sort bin threadgroup offsets into LDS for faster referencing
    if (localID < FFX_PARALLELSORT_SORT_BIN_COUNT)
        gs_FFX_PARALLELSORT_BinOffsetCache[localID] = FfxLoadSum(localID * NumThreadGroups + groupID);

    // Wait for everyone to catch up
    FFX_GROUP_MEMORY_BARRIER();

    // Data is processed in blocks, and how many we process can changed based on how much data we are processing
    // versus how many thread groups we are processing with
    int BlockSize = FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE;

    // Figure out this thread group's index into the block data (taking into account thread groups that need to do extra reads)
    FfxUInt32 ThreadgroupBlockStart = (BlockSize * NumBlocksPerThreadGroup * groupID);
    FfxUInt32 NumBlocksToProcess = NumBlocksPerThreadGroup;

    if (groupID >= NumThreadGroups - NumThreadGroupsWithAdditionalBlocks)
    {
        ThreadgroupBlockStart += (groupID - (NumThreadGroups - NumThreadGroupsWithAdditionalBlocks)) * BlockSize;
        NumBlocksToProcess++;
    }

    // Get the block start index for this thread
    FfxUInt32 BlockIndex = ThreadgroupBlockStart + localID;

    // Count value occurences
    FfxUInt32 newCount;
    for (int BlockCount = 0; BlockCount < NumBlocksToProcess; BlockCount++, BlockIndex += BlockSize)
    {
        FfxUInt32 DataIndex = BlockIndex;

        // Pre-load the key values in order to hide some of the read latency
        FfxUInt32 srcKeys[FFX_PARALLELSORT_ELEMENTS_PER_THREAD];
        srcKeys[0] = FfxLoadKey(DataIndex);
        srcKeys[1] = FfxLoadKey(DataIndex + FFX_PARALLELSORT_THREADGROUP_SIZE);
        srcKeys[2] = FfxLoadKey(DataIndex + (FFX_PARALLELSORT_THREADGROUP_SIZE * 2));
        srcKeys[3] = FfxLoadKey(DataIndex + (FFX_PARALLELSORT_THREADGROUP_SIZE * 3));

#ifdef FFX_PARALLELSORT_COPY_VALUE
        FfxUInt32 srcValues[FFX_PARALLELSORT_ELEMENTS_PER_THREAD];
        srcValues[0] = FfxLoadPayload(DataIndex);
        srcValues[1] = FfxLoadPayload(DataIndex + FFX_PARALLELSORT_THREADGROUP_SIZE);
        srcValues[2] = FfxLoadPayload(DataIndex + (FFX_PARALLELSORT_THREADGROUP_SIZE * 2));
        srcValues[3] = FfxLoadPayload(DataIndex + (FFX_PARALLELSORT_THREADGROUP_SIZE * 3));
#endif // FFX_PARALLELSORT_COPY_VALUE

        for (int i = 0; i < FFX_PARALLELSORT_ELEMENTS_PER_THREAD; i++)
        {
            // Clear the local histogram
            if (localID < FFX_PARALLELSORT_SORT_BIN_COUNT)
                gs_FFX_PARALLELSORT_LocalHistogram[localID] = 0;

            FfxUInt32 localKey = (DataIndex < NumKeys ? srcKeys[i] : 0xffffffff);
#ifdef FFX_PARALLELSORT_COPY_VALUE
            FfxUInt32 localValue = (DataIndex < NumKeys ? srcValues[i] : 0);
#endif // FFX_PARALLELSORT_COPY_VALUE

            // Sort the keys locally in LDS
            for (FfxUInt32 bitShift = 0; bitShift < FFX_PARALLELSORT_SORT_BITS_PER_PASS; bitShift += 2)
            {
                // Figure out the keyIndex
                FfxUInt32 keyIndex = (localKey >> ShiftBit) & 0xf;
                FfxUInt32 bitKey = (keyIndex >> bitShift) & 0x3;

                // Create a packed histogram
                FfxUInt32 packedHistogram = 1 << (bitKey * 8);

                // Sum up all the packed keys (generates counted offsets up to current thread group)
                FfxUInt32 localSum = ffxParallelSortBlockScanPrefix(packedHistogram, localID);

                // Last thread stores the updated histogram counts for the thread group
                // Scratch = 0xsum3|sum2|sum1|sum0 for thread group
                if (localID == (FFX_PARALLELSORT_THREADGROUP_SIZE - 1))
                    gs_FFX_PARALLELSORT_LDSScratch[0] = localSum + packedHistogram;

                // Wait for everyone to catch up
                FFX_GROUP_MEMORY_BARRIER();

                // Load the sums value for the thread group
                packedHistogram = gs_FFX_PARALLELSORT_LDSScratch[0];

                // Add prefix offsets for all 4 bit "keys" (packedHistogram = 0xsum2_1_0|sum1_0|sum0|0)
                packedHistogram = (packedHistogram << 8) + (packedHistogram << 16) + (packedHistogram << 24);

                // Calculate the proper offset for this thread's value
                localSum += packedHistogram;

                // Calculate target offset
                FfxUInt32 keyOffset = (localSum >> (bitKey * 8)) & 0xff;

                // Re-arrange the keys (store, sync, load)
                gs_FFX_PARALLELSORT_LDSSums[keyOffset] = localKey;
                FFX_GROUP_MEMORY_BARRIER();
                localKey = gs_FFX_PARALLELSORT_LDSSums[localID];

                // Wait for everyone to catch up
                FFX_GROUP_MEMORY_BARRIER();

#ifdef FFX_PARALLELSORT_COPY_VALUE
                // Re-arrange the values if we have them (store, sync, load)
                gs_FFX_PARALLELSORT_LDSSums[keyOffset] = localValue;
                FFX_GROUP_MEMORY_BARRIER();
                localValue = gs_FFX_PARALLELSORT_LDSSums[localID];

                // Wait for everyone to catch up
                FFX_GROUP_MEMORY_BARRIER();
#endif // FFX_PARALLELSORT_COPY_VALUE
            }

            // Need to recalculate the keyIndex on this thread now that values have been copied around the thread group
            FfxUInt32 keyIndex = (localKey >> ShiftBit) & 0xf;

            // Reconstruct histogram
            FFX_ATOMIC_ADD(gs_FFX_PARALLELSORT_LocalHistogram[keyIndex], 1);

            // Wait for everyone to catch up
            FFX_GROUP_MEMORY_BARRIER();

            // Prefix histogram
#if defined(FFX_HLSL)
            FfxUInt32 histogramPrefixSum = WavePrefixSum(localID < FFX_PARALLELSORT_SORT_BIN_COUNT ? gs_FFX_PARALLELSORT_LocalHistogram[localID] : 0);
#elif defined(FFX_GLSL)
            FfxUInt32 histogramPrefixSum = subgroupExclusiveAdd(localID < FFX_PARALLELSORT_SORT_BIN_COUNT ? gs_FFX_PARALLELSORT_LocalHistogram[localID] : 0);
#endif // #if defined(FFX_HLSL)

            // Broadcast prefix-sum via LDS
            if (localID < FFX_PARALLELSORT_SORT_BIN_COUNT)
                gs_FFX_PARALLELSORT_LDSScratch[localID] = histogramPrefixSum;

            // Get the global offset for this key out of the cache
            FfxUInt32 globalOffset = gs_FFX_PARALLELSORT_BinOffsetCache[keyIndex];

            // Wait for everyone to catch up
            FFX_GROUP_MEMORY_BARRIER();

            // Get the local offset (at this point the keys are all in increasing order from 0 -> num bins in localID 0 -> thread group size)
            FfxUInt32 localOffset = localID - gs_FFX_PARALLELSORT_LDSScratch[keyIndex];

            // Write to destination
            FfxUInt32 totalOffset = globalOffset + localOffset;

            if (totalOffset < NumKeys)
            {
                FfxStoreKey(totalOffset, localKey);

#ifdef FFX_PARALLELSORT_COPY_VALUE
                FfxStorePayload(totalOffset, localValue);
#endif // FFX_PARALLELSORT_COPY_VALUE
            }

            // Wait for everyone to catch up
            FFX_GROUP_MEMORY_BARRIER();

            // Update the cached histogram for the next set of entries
            if (localID < FFX_PARALLELSORT_SORT_BIN_COUNT)
                gs_FFX_PARALLELSORT_BinOffsetCache[localID] += gs_FFX_PARALLELSORT_LocalHistogram[localID];

            DataIndex += FFX_PARALLELSORT_THREADGROUP_SIZE;	// Increase the data offset by thread group size
        }
    }
}

// --- Entry points

#if ENTRY_PARALLEL_SORT_SCAN_REDUCE
// Buffers: rw_sum_table, rw_reduce_table
layout (local_size_x = FFX_PARALLELSORT_THREADGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
void main()
{
    uint LocalID = gl_LocalInvocationID.x;
    uint GroupID = gl_WorkGroupID.x;
    ffxParallelSortReduceCount(LocalID, GroupID);
}
#endif

#if ENTRY_PARALLEL_SORT_SCAN_ADD
// Buffers: rw_scan_source, rw_scan_dest, rw_scan_scratch
layout (local_size_x = FFX_PARALLELSORT_THREADGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
void main()
{
    uint LocalID = gl_LocalInvocationID.x;
    uint GroupID = gl_WorkGroupID.x;
    // When doing adds, we need to access data differently because reduce
    // has a more specialized access pattern to match optimized count
    // Access needs to be done similarly to reduce
    // Figure out what bin data we are reducing
    uint BinID = GroupID / FfxNumReduceThreadgroupPerBin();
    uint BinOffset = BinID * FfxNumThreadGroups();

    // Get the base index for this thread group
    uint BaseIndex = (GroupID % FfxNumReduceThreadgroupPerBin()) * FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE;

    ffxParallelSortScanPrefix(FfxNumThreadGroups(), LocalID, GroupID, BinOffset, BaseIndex, true);
}
#endif

#if ENTRY_PARALLEL_SORT_SCAN
// Buffers: rw_scan_source, rw_scan_dest
layout (local_size_x = FFX_PARALLELSORT_THREADGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
void main()
{
    uint LocalID = gl_LocalInvocationID.x;
    uint GroupID = gl_WorkGroupID.x;
    uint BaseIndex = FFX_PARALLELSORT_ELEMENTS_PER_THREAD * FFX_PARALLELSORT_THREADGROUP_SIZE * GroupID;
    ffxParallelSortScanPrefix(FfxNumScanValues(), LocalID, GroupID, 0, BaseIndex, false);
}
#endif

#if ENTRY_PARALLEL_SORT_SCATTER
// Buffers: rw_source_keys, rw_dest_keys, rw_sum_table, rw_source_payloads, rw_dest_payloads
layout (local_size_x = FFX_PARALLELSORT_THREADGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
void main()
{
    uint LocalID = gl_LocalInvocationID.x;
    uint GroupID = gl_WorkGroupID.x;
    ffxParallelSortScatterUInt(LocalID, GroupID, FfxShiftBit());
}
#endif

#if ENTRY_PARALLEL_SORT_COUNT
// Buffers: rw_source_keys, rw_sum_table
layout (local_size_x = FFX_PARALLELSORT_THREADGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
void main()
{
    uint LocalID = gl_LocalInvocationID.x;
    uint GroupID = gl_WorkGroupID.x;
    ffxParallelSortCountUInt(LocalID, GroupID, FfxShiftBit());
}
#endif
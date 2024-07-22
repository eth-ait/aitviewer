# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import moderngl
import numpy as np

from aitviewer.shaders import get_sort_program

# Adapted from parallelsort algorithm in FidelityFX-SDK
# https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/tree/main
#
# FidelityFX-SDK License
#
# Copyright (C) 2023 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


class GpuSort:
    # All these constants must match the respective constants in the sort.glsl shader.
    FFX_PARALLELSORT_ELEMENTS_PER_THREAD = 4
    FFX_PARALLELSORT_THREADGROUP_SIZE = 128
    FFX_PARALLELSORT_SORT_BITS_PER_PASS = 4
    FFX_PARALLELSORT_SORT_BIN_COUNT = 1 << FFX_PARALLELSORT_SORT_BITS_PER_PASS
    FFX_PARALLELSORT_MAX_THREADGROUPS_TO_RUN = 800

    FFX_PARALLELSORT_BIND_UAV_SOURCE_KEYS = 0
    FFX_PARALLELSORT_BIND_UAV_DEST_KEYS = 1
    FFX_PARALLELSORT_BIND_UAV_SOURCE_PAYLOADS = 2
    FFX_PARALLELSORT_BIND_UAV_DEST_PAYLOADS = 3
    FFX_PARALLELSORT_BIND_UAV_SUM_TABLE = 4
    FFX_PARALLELSORT_BIND_UAV_REDUCE_TABLE = 5
    FFX_PARALLELSORT_BIND_UAV_SCAN_SOURCE = 6
    FFX_PARALLELSORT_BIND_UAV_SCAN_DEST = 7
    FFX_PARALLELSORT_BIND_UAV_SCAN_SCRATCH = 8

    def __init__(self, ctx: moderngl.Context, count: int):
        self.count = count

        # Create programs.
        self.prog_count: moderngl.ComputeShader = get_sort_program("COUNT")
        self.prog_scan_reduce: moderngl.ComputeShader = get_sort_program("SCAN_REDUCE")
        self.prog_scan: moderngl.ComputeShader = get_sort_program("SCAN")
        self.prog_scan_add: moderngl.ComputeShader = get_sort_program("SCAN_ADD")
        self.prog_scatter: moderngl.ComputeShader = get_sort_program("SCATTER")

        def div_round_up(n, d):
            return (n + d - 1) // d

        # Buffers.
        block_size = self.FFX_PARALLELSORT_ELEMENTS_PER_THREAD * self.FFX_PARALLELSORT_THREADGROUP_SIZE
        num_blocks = div_round_up(count, block_size)
        num_reduced_blocks = div_round_up(num_blocks, block_size)

        scratch_buffer_size = self.FFX_PARALLELSORT_SORT_BIN_COUNT * num_blocks
        reduce_scratch_buffer_size = self.FFX_PARALLELSORT_SORT_BIN_COUNT * num_reduced_blocks

        self.sort_scratch_buf = ctx.buffer(reserve=count * 4)
        self.payload_scratch_buf = ctx.buffer(reserve=count * 4)
        self.scratch_buf = ctx.buffer(reserve=scratch_buffer_size * 4)
        self.reduced_scratch_buf = ctx.buffer(reserve=reduce_scratch_buffer_size * 4)

        # Constants.
        num_thread_groups_to_run = self.FFX_PARALLELSORT_MAX_THREADGROUPS_TO_RUN
        blocks_per_thread_group = num_blocks // num_thread_groups_to_run
        num_thread_groups_with_additional_blocks = num_blocks % num_thread_groups_to_run

        if num_blocks < num_thread_groups_to_run:
            blocks_per_thread_group = 1
            num_thread_groups_to_run = num_blocks
            num_thread_groups_with_additional_blocks = 0

        num_reduce_thread_groups_to_run = self.FFX_PARALLELSORT_SORT_BIN_COUNT * (
            1 if block_size > num_thread_groups_to_run else div_round_up(num_thread_groups_to_run, block_size)
        )
        num_reduce_thread_groups_per_bin = num_reduce_thread_groups_to_run // self.FFX_PARALLELSORT_SORT_BIN_COUNT
        num_scan_values = num_reduce_thread_groups_to_run

        constants = np.array(
            [
                count,
                blocks_per_thread_group,
                num_thread_groups_to_run,
                num_thread_groups_with_additional_blocks,
                num_reduce_thread_groups_per_bin,
                num_scan_values,
                0,
                0,
            ],
            np.uint32,
        )
        self.constants_buf = ctx.buffer(constants.tobytes())

        self.num_thread_groups_to_run = num_thread_groups_to_run
        self.num_reduce_thread_groups_to_run = num_reduce_thread_groups_to_run

    def run(self, ctx: moderngl.Context, keys: moderngl.Buffer, values: moderngl.Buffer):
        shift_data = np.array([0], np.uint32)

        src_keys = keys
        src_payload = values
        dst_keys = self.sort_scratch_buf
        dst_payload = self.payload_scratch_buf

        for shift in range(0, 32, self.FFX_PARALLELSORT_SORT_BITS_PER_PASS):
            shift_data[0] = shift
            self.constants_buf.write(shift_data.tobytes(), offset=6 * 4)
            self.constants_buf.bind_to_uniform_block(0)

            src_keys.bind_to_storage_buffer(self.FFX_PARALLELSORT_BIND_UAV_SOURCE_KEYS)
            self.scratch_buf.bind_to_storage_buffer(self.FFX_PARALLELSORT_BIND_UAV_SUM_TABLE)
            self.prog_count.run(self.num_thread_groups_to_run)
            ctx.memory_barrier()

            self.scratch_buf.bind_to_storage_buffer(self.FFX_PARALLELSORT_BIND_UAV_SUM_TABLE)
            self.reduced_scratch_buf.bind_to_storage_buffer(self.FFX_PARALLELSORT_BIND_UAV_REDUCE_TABLE)
            self.prog_scan_reduce.run(self.num_reduce_thread_groups_to_run)
            ctx.memory_barrier()

            self.reduced_scratch_buf.bind_to_storage_buffer(self.FFX_PARALLELSORT_BIND_UAV_SCAN_SOURCE)
            self.reduced_scratch_buf.bind_to_storage_buffer(self.FFX_PARALLELSORT_BIND_UAV_SCAN_DEST)
            self.prog_scan.run(1)
            ctx.memory_barrier()

            self.scratch_buf.bind_to_storage_buffer(self.FFX_PARALLELSORT_BIND_UAV_SCAN_SOURCE)
            self.scratch_buf.bind_to_storage_buffer(self.FFX_PARALLELSORT_BIND_UAV_SCAN_DEST)
            self.reduced_scratch_buf.bind_to_storage_buffer(self.FFX_PARALLELSORT_BIND_UAV_SCAN_SCRATCH)
            self.prog_scan_add.run(self.num_reduce_thread_groups_to_run)
            ctx.memory_barrier()

            src_keys.bind_to_storage_buffer(self.FFX_PARALLELSORT_BIND_UAV_SOURCE_KEYS)
            dst_keys.bind_to_storage_buffer(self.FFX_PARALLELSORT_BIND_UAV_DEST_KEYS)
            src_payload.bind_to_storage_buffer(self.FFX_PARALLELSORT_BIND_UAV_SOURCE_PAYLOADS)
            dst_payload.bind_to_storage_buffer(self.FFX_PARALLELSORT_BIND_UAV_DEST_PAYLOADS)
            self.scratch_buf.bind_to_storage_buffer(self.FFX_PARALLELSORT_BIND_UAV_SUM_TABLE)
            self.prog_scatter.run(self.num_thread_groups_to_run)
            ctx.memory_barrier()

            src_keys, dst_keys = dst_keys, src_keys
            src_payload, dst_payload = dst_payload, src_payload

    def release(self):
        self.sort_scratch_buf.release()
        self.payload_scratch_buf.release()
        self.scratch_buf.release()
        self.reduced_scratch_buf.release()
        self.constants_buf.release()

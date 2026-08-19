/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// GPU-side argmax for greedy token selection.
//
// Eliminates the D2H transfer of the full logit vector (~600KB for 151K vocab)
// per decode step. Instead, runs a parallel reduction on GPU and copies back
// only the single int32 token ID (4 bytes).
//
// Architecture: single-block parallel reduction using shared memory.
// For vocab_size up to ~256K, one block of 256 threads is sufficient
// (each thread handles ceil(vocab_size/256) elements).

#include "argmax_kernel.h"

#include <cuda_runtime.h>
#include <cfloat>

namespace trtmc {

static constexpr int kBlockSize = 256;

__global__ void argmax_reduce_kernel(
    const float* __restrict__ logits,
    int32_t vocab_size,
    int32_t* __restrict__ out_token_id,
    float* __restrict__ out_logit)
{
    __shared__ float s_vals[kBlockSize];
    __shared__ int32_t s_idxs[kBlockSize];

    const int tid = threadIdx.x;

    // Each thread finds the max over its strided range
    float best_val = -FLT_MAX;
    int32_t best_idx = 0;

    for (int i = tid; i < vocab_size; i += kBlockSize)
    {
        float v = logits[i];
        if (v > best_val)
        {
            best_val = v;
            best_idx = i;
        }
    }

    s_vals[tid] = best_val;
    s_idxs[tid] = best_idx;
    __syncthreads();

    // Tree reduction
    for (int stride = kBlockSize / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            if (s_vals[tid + stride] > s_vals[tid])
            {
                s_vals[tid] = s_vals[tid + stride];
                s_idxs[tid] = s_idxs[tid + stride];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes result
    if (tid == 0)
    {
        *out_token_id = s_idxs[0];
        if (out_logit)
        {
            *out_logit = s_vals[0];
        }
    }
}

void lance_gpu_argmax(
    const float* d_logits,
    int32_t vocab_size,
    int32_t* d_token_id,
    float* d_logit_val,
    cudaStream_t stream)
{
    if (vocab_size <= 0) return;
    argmax_reduce_kernel<<<1, kBlockSize, 0, stream>>>(
        d_logits, vocab_size, d_token_id, d_logit_val);
}

} // namespace trtmc

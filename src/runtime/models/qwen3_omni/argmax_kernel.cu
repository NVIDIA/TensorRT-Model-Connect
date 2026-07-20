/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/qwen3_omni/argmax_kernel.h"

#include <cfloat>
#include <climits>
#include <cuda_runtime.h>

namespace trtmc {

namespace {

constexpr int32_t kBlockSize = 256;

__global__ void argmax_kernel(const float* logits, int32_t vocab_size, int32_t* token_id) {
    __shared__ float values[kBlockSize];
    __shared__ int32_t indices[kBlockSize];

    const int32_t thread = threadIdx.x;
    float best_value = -FLT_MAX;
    int32_t best_index = INT_MAX;
    for (int32_t index = thread; index < vocab_size; index += kBlockSize) {
        const float value = logits[index];
        if (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    values[thread] = best_value;
    indices[thread] = best_index;
    __syncthreads();

    for (int32_t stride = kBlockSize / 2; stride > 0; stride >>= 1) {
        if (thread < stride) {
            const float candidate_value = values[thread + stride];
            const int32_t candidate_index = indices[thread + stride];
            if (candidate_value > values[thread] ||
                (candidate_value == values[thread] && candidate_index < indices[thread])) {
                values[thread] = candidate_value;
                indices[thread] = candidate_index;
            }
        }
        __syncthreads();
    }
    if (thread == 0)
        *token_id = indices[0];
}

} // namespace

void qwen3_omni_gpu_argmax(const float* logits, int32_t vocab_size, int32_t* token_id,
                           cudaStream_t stream) {
    if (logits == nullptr || token_id == nullptr || vocab_size <= 0)
        return;
    argmax_kernel<<<1, kBlockSize, 0, stream>>>(logits, vocab_size, token_id);
}

} // namespace trtmc

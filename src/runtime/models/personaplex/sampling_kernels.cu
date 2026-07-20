/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/personaplex/sampling_kernels.h"

#include <cfloat>
#include <climits>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace trtmc {

namespace {

constexpr int32_t kArgmaxThreads = 256;
constexpr int32_t kSampleThreads = 128;
constexpr int32_t kMaxTopK = 4096;

__device__ bool candidate_is_better(float lhs_value, int32_t lhs_index, float rhs_value,
                                    int32_t rhs_index) {
    return lhs_value > rhs_value || (lhs_value == rhs_value && lhs_index < rhs_index);
}

__device__ bool candidate_precedes_cutoff(float value, int32_t index, int32_t rank,
                                          float cutoff_value, int32_t cutoff_index) {
    if (rank == 0)
        return false;
    if (value > cutoff_value)
        return true;
    if (value < cutoff_value)
        return false;
    return index <= cutoff_index;
}

__device__ void find_thread_candidate(const float* logits, int32_t vocab_size, int32_t rank,
                                      float cutoff_value, int32_t cutoff_index, int32_t thread,
                                      float& best_value, int32_t& best_index) {
    best_value = -FLT_MAX;
    best_index = INT_MAX;
    for (int32_t index = thread; index < vocab_size; index += kSampleThreads) {
        const float value = logits[index];
        if (candidate_precedes_cutoff(value, index, rank, cutoff_value, cutoff_index))
            continue;
        if (candidate_is_better(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
        }
    }
}

__device__ void reduce_thread_candidates(float* values, int32_t* indices, int32_t thread) {
    for (int32_t stride = kSampleThreads / 2; stride > 0; stride >>= 1) {
        if (thread < stride &&
            candidate_is_better(values[thread + stride], indices[thread + stride], values[thread],
                                indices[thread])) {
            values[thread] = values[thread + stride];
            indices[thread] = indices[thread + stride];
        }
        __syncthreads();
    }
}

__device__ float topk_probability_sum(const float* top_values, int32_t keep, float temperature) {
    const float max_logit = top_values[0];
    float sum = 0.0F;
    for (int32_t slot = 0; slot < keep; ++slot)
        sum += expf((top_values[slot] - max_logit) / temperature);
    return sum;
}

__device__ float next_uniform(uint64_t* rng_state) {
    uint64_t state = *rng_state;
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    *rng_state = state;
    return static_cast<float>(state & 0xFFFFFFFFULL) / 4294967296.0F;
}

__device__ int32_t sample_rank(const float* top_values, const int32_t* top_indices, int32_t keep,
                               float temperature, float probability_sum, float draw) {
    const float max_logit = top_values[0];
    float cumulative = 0.0F;
    for (int32_t slot = 0; slot < keep; ++slot) {
        const float unnormalized = expf((top_values[slot] - max_logit) / temperature);
        const float probability = probability_sum > 0.0F ? unnormalized / probability_sum
                                                         : 1.0F / static_cast<float>(keep);
        cumulative += probability;
        if (draw < cumulative)
            return top_indices[slot];
    }
    return top_indices[keep - 1];
}

__global__ void argmax_kernel(const float* logits, int32_t vocab_size, int32_t max_token_id,
                              int32_t* token_id) {
    __shared__ float values[kArgmaxThreads];
    __shared__ int32_t indices[kArgmaxThreads];

    const int32_t thread = threadIdx.x;
    float best_value = -FLT_MAX;
    int32_t best_index = INT_MAX;
    for (int32_t index = thread; index < vocab_size; index += kArgmaxThreads) {
        const float value = logits[index];
        if (candidate_is_better(value, index, best_value, best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    values[thread] = best_value;
    indices[thread] = best_index;
    __syncthreads();

    for (int32_t stride = kArgmaxThreads / 2; stride > 0; stride >>= 1) {
        if (thread < stride &&
            candidate_is_better(values[thread + stride], indices[thread + stride], values[thread],
                                indices[thread])) {
            values[thread] = values[thread + stride];
            indices[thread] = indices[thread + stride];
        }
        __syncthreads();
    }
    if (thread == 0)
        *token_id = min(indices[0], max_token_id);
}

__global__ void sample_topk_kernel(const float* logits, int32_t vocab_size, float temperature,
                                   int32_t top_k, int32_t max_token_id, uint64_t* rng_state,
                                   int32_t* token_id) {
    __shared__ float reduction_values[kSampleThreads];
    __shared__ int32_t reduction_indices[kSampleThreads];
    __shared__ float top_values[kMaxTopK];
    __shared__ int32_t top_indices[kMaxTopK];
    const int32_t thread = threadIdx.x;
    const int32_t keep = min(top_k, vocab_size);
    for (int32_t rank = 0; rank < keep; ++rank) {
        const float cutoff_value = rank > 0 ? top_values[rank - 1] : 0.0F;
        const int32_t cutoff_index = rank > 0 ? top_indices[rank - 1] : 0;
        float local_value;
        int32_t local_index;
        find_thread_candidate(logits, vocab_size, rank, cutoff_value, cutoff_index, thread,
                              local_value, local_index);
        reduction_values[thread] = local_value;
        reduction_indices[thread] = local_index;
        __syncthreads();
        reduce_thread_candidates(reduction_values, reduction_indices, thread);
        if (thread == 0) {
            top_values[rank] = reduction_values[0];
            top_indices[rank] = reduction_indices[0];
        }
        __syncthreads();
    }

    if (thread != 0)
        return;
    const float probability_sum = topk_probability_sum(top_values, keep, temperature);
    *token_id = min(sample_rank(top_values, top_indices, keep, temperature, probability_sum,
                                next_uniform(rng_state)),
                    max_token_id);
}

template <typename HiddenT>
__device__ float load_hidden(const HiddenT* hidden, int32_t index) {
    return static_cast<float>(hidden[index]);
}

template <>
__device__ float load_hidden<__half>(const __half* hidden, int32_t index) {
    return __half2float(hidden[index]);
}

template <>
__device__ float load_hidden<__nv_bfloat16>(const __nv_bfloat16* hidden, int32_t index) {
    return __bfloat162float(hidden[index]);
}

__device__ float text_seed(const float* embedding, int64_t elements, const int32_t* selected_tokens,
                           int32_t forced_token, bool token_is_forced, int32_t vocab_size,
                           int32_t hidden_size, int32_t row) {
    if (!embedding || vocab_size <= 0)
        return 0.0F;
    const int32_t selected = token_is_forced ? forced_token : selected_tokens[0];
    const int32_t token = max(0, min(selected, vocab_size - 1));
    const int64_t offset = static_cast<int64_t>(token) * hidden_size + row;
    return offset < elements ? embedding[offset] : 0.0F;
}

__device__ float audio_seed(const float* embeddings, int64_t elements,
                            const int32_t* selected_tokens, int32_t codebook, int32_t forced_token,
                            bool token_is_forced, int32_t vocab_size, int32_t hidden_size,
                            int32_t num_embeddings, int32_t row) {
    if (!embeddings || vocab_size <= 0 || codebook <= 0 || codebook > num_embeddings)
        return 0.0F;
    const int32_t selected = token_is_forced ? forced_token : selected_tokens[codebook];
    const int32_t token = max(0, min(selected, vocab_size - 1));
    const int64_t table = static_cast<int64_t>(codebook - 1) * vocab_size * hidden_size;
    const int64_t offset = table + static_cast<int64_t>(token) * hidden_size + row;
    return offset < elements ? embeddings[offset] : 0.0F;
}

template <typename HiddenT>
__device__ float projected_hidden(const HiddenT* temporal_hidden, const float* depth_projection,
                                  int64_t projection_elements, int32_t codebook,
                                  int32_t depth_hidden, int32_t temporal_hidden_dim, int32_t row) {
    if (!depth_projection || !temporal_hidden || temporal_hidden_dim <= 0)
        return 0.0F;
    const int64_t matrix = static_cast<int64_t>(codebook) * depth_hidden * temporal_hidden_dim;
    const int64_t row_offset = matrix + static_cast<int64_t>(row) * temporal_hidden_dim;
    if (row_offset + temporal_hidden_dim > projection_elements)
        return 0.0F;
    float value = 0.0F;
    for (int32_t column = 0; column < temporal_hidden_dim; ++column) {
        value += depth_projection[row_offset + column] * load_hidden(temporal_hidden, column);
    }
    return value;
}

template <typename HiddenT>
__global__ void prepare_depth_embedding_kernel(
    float* output, const HiddenT* temporal_hidden, const float* depth_projection,
    int64_t depth_projection_elements, const float* depth_text_embedding,
    int64_t depth_text_embedding_elements, const float* depth_audio_embeddings,
    int64_t depth_audio_embedding_elements, const int32_t* selected_tokens, int32_t codebook,
    int32_t forced_text_token, bool text_token_is_forced, int32_t forced_previous_token,
    bool previous_token_is_forced, int32_t depth_hidden, int32_t temporal_hidden_dim,
    int32_t depth_text_vocab, int32_t audio_vocab_size, int32_t num_depformer_embeddings) {
    const int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= depth_hidden)
        return;
    const float seed =
        codebook == 0
            ? text_seed(depth_text_embedding, depth_text_embedding_elements, selected_tokens,
                        forced_text_token, text_token_is_forced, depth_text_vocab, depth_hidden,
                        row)
            : audio_seed(depth_audio_embeddings, depth_audio_embedding_elements, selected_tokens,
                         codebook, forced_previous_token, previous_token_is_forced,
                         audio_vocab_size, depth_hidden, num_depformer_embeddings, row);
    output[row] =
        seed + projected_hidden(temporal_hidden, depth_projection, depth_projection_elements,
                                codebook, depth_hidden, temporal_hidden_dim, row);
}

} // namespace

bool personaplex_select_token(const float* logits, int32_t vocab_size, float temperature,
                              int32_t top_k, int32_t max_token_id, uint64_t* rng_state,
                              int32_t* token_id, cudaStream_t stream) {
    if (!logits || !token_id || vocab_size <= 0)
        return false;
    max_token_id = max(0, min(max_token_id, vocab_size - 1));
    if (temperature < 1.0e-6F || top_k <= 0 || !rng_state) {
        argmax_kernel<<<1, kArgmaxThreads, 0, stream>>>(logits, vocab_size, max_token_id, token_id);
        return cudaGetLastError() == cudaSuccess;
    }
    if (top_k > kMaxTopK)
        return false;
    sample_topk_kernel<<<1, kSampleThreads, 0, stream>>>(logits, vocab_size, temperature, top_k,
                                                         max_token_id, rng_state, token_id);
    return cudaGetLastError() == cudaSuccess;
}

void personaplex_prepare_depth_embedding(
    float* output, const void* temporal_hidden, DType temporal_hidden_dtype,
    const float* depth_projection, int64_t depth_projection_elements,
    const float* depth_text_embedding, int64_t depth_text_embedding_elements,
    const float* depth_audio_embeddings, int64_t depth_audio_embedding_elements,
    const int32_t* selected_tokens, int32_t codebook, int32_t forced_text_token,
    bool text_token_is_forced, int32_t forced_previous_token, bool previous_token_is_forced,
    int32_t depth_hidden, int32_t temporal_hidden_dim, int32_t depth_text_vocab,
    int32_t audio_vocab_size, int32_t num_depformer_embeddings, cudaStream_t stream) {
    if (!output || !selected_tokens || depth_hidden <= 0)
        return;
    constexpr int32_t threads = 256;
    const int32_t blocks = (depth_hidden + threads - 1) / threads;
#define TRTMC_LAUNCH_DEPTH_EMBED(HIDDEN_TYPE)                                                      \
    prepare_depth_embedding_kernel<<<blocks, threads, 0, stream>>>(                                \
        output, static_cast<const HIDDEN_TYPE*>(temporal_hidden), depth_projection,                \
        depth_projection_elements, depth_text_embedding, depth_text_embedding_elements,            \
        depth_audio_embeddings, depth_audio_embedding_elements, selected_tokens, codebook,         \
        forced_text_token, text_token_is_forced, forced_previous_token, previous_token_is_forced,  \
        depth_hidden, temporal_hidden_dim, depth_text_vocab, audio_vocab_size,                     \
        num_depformer_embeddings)
    switch (temporal_hidden_dtype) {
    case DType::kFloat16:
        TRTMC_LAUNCH_DEPTH_EMBED(__half);
        break;
    case DType::kBFloat16:
        TRTMC_LAUNCH_DEPTH_EMBED(__nv_bfloat16);
        break;
    case DType::kFloat32:
    default:
        TRTMC_LAUNCH_DEPTH_EMBED(float);
        break;
    }
#undef TRTMC_LAUNCH_DEPTH_EMBED
}

} // namespace trtmc

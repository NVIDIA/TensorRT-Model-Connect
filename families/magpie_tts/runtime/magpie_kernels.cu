/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/magpie_tts/runtime/magpie_kernels.h"

#include <cfloat>
#include <cstdint>

namespace trtmc {

// ---------------------------------------------------------------------------
// Kernel: segmented argmax over num_codebooks segments of codebook_size each.
// Each block handles one codebook. Uses shared-memory parallel reduction.
// Writes:
//   d_codes_out[cb]       = argmax within [0, audio_range)
//   d_full_argmax_out[cb] = argmax within [0, codebook_size) (for EOS check)
// ---------------------------------------------------------------------------

__global__ void segmented_argmax_kernel(const float* __restrict__ d_logits, int32_t codebook_size,
                                        int32_t audio_range, int32_t* __restrict__ d_codes_out,
                                        int32_t* __restrict__ d_full_argmax_out) {
    const int32_t cb = static_cast<int32_t>(blockIdx.x);
    const int32_t tid = static_cast<int32_t>(threadIdx.x);
    const int32_t block_size = static_cast<int32_t>(blockDim.x);
    const float* cb_logits = d_logits + cb * codebook_size;

    // Each thread finds local best over its strided range
    float best_val_full = -FLT_MAX;
    int32_t best_idx_full = 0;
    float best_val_audio = -FLT_MAX;
    int32_t best_idx_audio = 0;

    // Full range [0, codebook_size)
    for (int32_t i = tid; i < codebook_size; i += block_size) {
        float v = cb_logits[i];
        if (v > best_val_full) {
            best_val_full = v;
            best_idx_full = i;
        }
        // Audio range [0, audio_range)
        if (i < audio_range && v > best_val_audio) {
            best_val_audio = v;
            best_idx_audio = i;
        }
    }

    // Shared memory reduction
    // Layout: [block_size] floats for full vals, [block_size] ints for full idx,
    //         [block_size] floats for audio vals, [block_size] ints for audio idx
    extern __shared__ char smem[];
    float* s_full_val = reinterpret_cast<float*>(smem);
    int32_t* s_full_idx = reinterpret_cast<int32_t*>(s_full_val + block_size);
    float* s_audio_val = reinterpret_cast<float*>(s_full_idx + block_size);
    int32_t* s_audio_idx = reinterpret_cast<int32_t*>(s_audio_val + block_size);

    s_full_val[tid] = best_val_full;
    s_full_idx[tid] = best_idx_full;
    s_audio_val[tid] = best_val_audio;
    s_audio_idx[tid] = best_idx_audio;
    __syncthreads();

    // Tree reduction
    for (int32_t stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_full_val[tid + stride] > s_full_val[tid]) {
                s_full_val[tid] = s_full_val[tid + stride];
                s_full_idx[tid] = s_full_idx[tid + stride];
            }
            if (s_audio_val[tid + stride] > s_audio_val[tid]) {
                s_audio_val[tid] = s_audio_val[tid + stride];
                s_audio_idx[tid] = s_audio_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_codes_out[cb] = s_audio_idx[0];
        d_full_argmax_out[cb] = s_full_idx[0];
    }
}

void magpie_greedy_sample_device(const float* d_logits, int32_t num_codebooks,
                                 int32_t codebook_size, int32_t audio_range, int32_t* d_codes_out,
                                 int32_t* d_full_argmax_out, cudaStream_t stream) {
    // Use 256 threads per block (enough for codebook_size=2024)
    constexpr int32_t kBlockSize = 256;
    const int32_t grid = num_codebooks; // one block per codebook

    // Shared memory: 2*(float + int32_t) per thread
    const std::size_t smem_bytes =
        static_cast<std::size_t>(kBlockSize) * (2 * sizeof(float) + 2 * sizeof(int32_t));

    segmented_argmax_kernel<<<grid, kBlockSize, smem_bytes, stream>>>(
        d_logits, codebook_size, audio_range, d_codes_out, d_full_argmax_out);
}

// ---------------------------------------------------------------------------
// Kernel: gather 8 codebook embeddings, average them, write to output.
// Launch with hidden_size threads. Each thread accumulates one hidden dim
// across all codebooks.
// ---------------------------------------------------------------------------

__global__ void gather_average_embed_kernel(const float* __restrict__ audio_embed_table,
                                            const int32_t* __restrict__ prev_codes,
                                            int32_t num_codebooks, int32_t codebook_size,
                                            int32_t hidden_size, float* __restrict__ output) {
    const int32_t h = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (h >= hidden_size)
        return;

    float sum = 0.0F;
    for (int32_t cb = 0; cb < num_codebooks; ++cb) {
        // Table layout: [num_codebooks, codebook_size, hidden_size]
        const int64_t offset = static_cast<int64_t>(cb) * codebook_size * hidden_size +
                               static_cast<int64_t>(prev_codes[cb]) * hidden_size + h;
        sum += audio_embed_table[offset];
    }
    output[h] = sum / static_cast<float>(num_codebooks);
}

void magpie_gather_average_embed_device(const float* d_audio_embed_table,
                                        const int32_t* d_prev_codes, int32_t num_codebooks,
                                        int32_t codebook_size, int32_t hidden_size, float* d_output,
                                        cudaStream_t stream) {
    constexpr int32_t kBlockSize = 256;
    const int32_t grid = (hidden_size + kBlockSize - 1) / kBlockSize;

    gather_average_embed_kernel<<<grid, kBlockSize, 0, stream>>>(
        d_audio_embed_table, d_prev_codes, num_codebooks, codebook_size, hidden_size, d_output);
}

// ---------------------------------------------------------------------------
// Kernel: scatter codes into accumulator, update prev_codes, check EOS.
// Single block of num_codebooks threads (8).
// ---------------------------------------------------------------------------

__global__ void scatter_codes_kernel(const int32_t* __restrict__ d_codes,
                                     int32_t* __restrict__ d_all_codes,
                                     int32_t* __restrict__ d_prev_codes,
                                     const int32_t* __restrict__ d_full_argmax,
                                     int32_t* __restrict__ d_eos_flag, int32_t frame_idx,
                                     int32_t num_codebooks, int32_t eos_token) {
    const int32_t cb = static_cast<int32_t>(threadIdx.x);
    if (cb >= num_codebooks)
        return;

    const int32_t code = d_codes[cb];

    // Write to accumulator: all_codes[frame_idx * num_codebooks + cb]
    d_all_codes[frame_idx * num_codebooks + cb] = code;

    // Update prev_codes for next iteration
    d_prev_codes[cb] = code;

    // Check EOS (any codebook's full_argmax == eos_token sets the flag)
    if (d_full_argmax[cb] == eos_token) {
        atomicOr(d_eos_flag, 1);
    }
}

void magpie_scatter_codes_device(const int32_t* d_codes, int32_t* d_all_codes,
                                 int32_t* d_prev_codes, const int32_t* d_full_argmax,
                                 int32_t* d_eos_flag, int32_t frame_idx, int32_t num_codebooks,
                                 int32_t eos_token, cudaStream_t stream) {
    scatter_codes_kernel<<<1, num_codebooks, 0, stream>>>(d_codes, d_all_codes, d_prev_codes,
                                                          d_full_argmax, d_eos_flag, frame_idx,
                                                          num_codebooks, eos_token);
}

// ---------------------------------------------------------------------------
// Kernel: top-k sampling per codebook.
// One block per codebook, 256 threads.
//   Phase 1: parallel reduction to find global max and full-range argmax
//   Phase 2: binary search on logit value to find top-k threshold
//   Phase 3: temperature softmax over elements above threshold
//   Phase 4: thread-0 sequential scan for multinomial sampling
// ---------------------------------------------------------------------------

// Exact replica of CPU sample_top_k + decode_magpie_frame_codes logic.
// One block per codebook. Thread 0 does all work for exact CPU parity —
// same partial-sort order, same softmax accumulation, same CDF scan.
// The other threads are idle but 8 blocks run in parallel (one per codebook).
// Total work per codebook: ~160K comparisons for selection sort + 80 exp() = ~0.16 ms.
__global__ void topk_sample_kernel(const float* __restrict__ d_logits, int32_t cb_size,
                                   int32_t audio_range, int32_t top_k, float temperature,
                                   int32_t eos_token, const float* __restrict__ d_rand_vals,
                                   int32_t* __restrict__ d_codes_out,
                                   int32_t* __restrict__ d_full_argmax_out,
                                   int32_t* __restrict__ d_eos_flag) {
    if (threadIdx.x != 0)
        return;

    const int32_t cb = static_cast<int32_t>(blockIdx.x);
    const float* logits = d_logits + cb * cb_size;

    // Use shared memory for the indices array: [cb_size] int32_t
    extern __shared__ int32_t s_indices[];

    // Initialize indices 0..cb_size-1
    for (int32_t i = 0; i < cb_size; ++i)
        s_indices[i] = i;

    // ---- Step 1: Full-range argmax for EOS detection ----
    int32_t full_argmax = 0;
    float full_max = logits[0];
    for (int32_t i = 1; i < cb_size; ++i) {
        if (logits[i] > full_max) {
            full_max = logits[i];
            full_argmax = i;
        }
    }
    d_full_argmax_out[cb] = full_argmax;
    if (full_argmax == eos_token)
        atomicOr(d_eos_flag, 1);

    // ---- Step 2: Selection sort for top-k (exact std::partial_sort equivalent) ----
    // Place the k largest elements at indices[0..k-1] in descending logit order.
    const int32_t k = (top_k < cb_size) ? top_k : cb_size;
    for (int32_t pos = 0; pos < k; ++pos) {
        int32_t best = pos;
        float best_val = logits[s_indices[pos]];
        for (int32_t j = pos + 1; j < cb_size; ++j) {
            float v = logits[s_indices[j]];
            if (v > best_val) {
                best_val = v;
                best = j;
            }
        }
        if (best != pos) {
            int32_t tmp = s_indices[pos];
            s_indices[pos] = s_indices[best];
            s_indices[best] = tmp;
        }
    }

    // ---- Step 3: Temperature softmax over top-k ----
    // Matches CPU: max_logit = logits[indices[0]], probs[i] = exp((logits[indices[i]] - max) /
    // temp)
    const float max_logit = logits[s_indices[0]];
    float sum = 0.0F;

    // Compute unnormalized probs in-place (reuse shared memory as float)
    float* s_probs = reinterpret_cast<float*>(s_indices + cb_size);
    for (int32_t i = 0; i < k; ++i) {
        s_probs[i] = expf((logits[s_indices[i]] - max_logit) / temperature);
        sum += s_probs[i];
    }
    // Normalize
    for (int32_t i = 0; i < k; ++i)
        s_probs[i] /= sum;

    // ---- Step 4: Multinomial sampling (exact CPU replica) ----
    const float r = d_rand_vals[cb];
    float cumulative = 0.0F;
    int32_t sampled_id = s_indices[k - 1]; // fallback: last top-k element

    for (int32_t i = 0; i < k; ++i) {
        cumulative += s_probs[i];
        if (r < cumulative) {
            sampled_id = s_indices[i];
            break;
        }
    }

    // ---- Step 5: Validate sampled token (exact CPU decode_magpie_frame_codes logic) ----
    if (sampled_id == eos_token) {
        // Sampled EOS — signal stop, use audio-range argmax as output code
        atomicOr(d_eos_flag, 1);
        // Fallback to audio-range argmax
        int32_t audio_best = 0;
        float audio_best_val = logits[0];
        for (int32_t i = 1; i < audio_range; ++i) {
            if (logits[i] > audio_best_val) {
                audio_best_val = logits[i];
                audio_best = i;
            }
        }
        sampled_id = audio_best;
    } else if (sampled_id < 0 || sampled_id >= audio_range) {
        // BOS or other special token — fallback to audio-range argmax
        int32_t audio_best = 0;
        float audio_best_val = logits[0];
        for (int32_t i = 1; i < audio_range; ++i) {
            if (logits[i] > audio_best_val) {
                audio_best_val = logits[i];
                audio_best = i;
            }
        }
        sampled_id = audio_best;
    }

    d_codes_out[cb] = sampled_id;
}

void magpie_topk_sample_device(const float* d_logits, int32_t num_codebooks, int32_t codebook_size,
                               int32_t audio_range, int32_t top_k, float temperature,
                               int32_t eos_token, const float* d_rand_vals, int32_t* d_codes_out,
                               int32_t* d_full_argmax_out, int32_t* d_eos_flag,
                               cudaStream_t stream) {
    // Shared memory: [codebook_size] int32 for indices + [top_k] float for probs
    const std::size_t smem_bytes = static_cast<std::size_t>(codebook_size) * sizeof(int32_t) +
                                   static_cast<std::size_t>(top_k) * sizeof(float);

    // One thread per block — exact CPU parity; parallelism is across codebooks
    topk_sample_kernel<<<num_codebooks, 1, smem_bytes, stream>>>(
        d_logits, codebook_size, audio_range, top_k, temperature, eos_token, d_rand_vals,
        d_codes_out, d_full_argmax_out, d_eos_flag);
}

// ---------------------------------------------------------------------------
// Kernel: CFG interpolation
// out[i] = uncond[i] + scale * (cond[i] - uncond[i])
// ---------------------------------------------------------------------------

__global__ void cfg_interpolate_kernel(const float* __restrict__ d_cond,
                                       const float* __restrict__ d_uncond,
                                       float* __restrict__ d_out, float cfg_scale,
                                       int32_t num_elements) {
    const int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= num_elements)
        return;
    d_out[i] = d_uncond[i] + cfg_scale * (d_cond[i] - d_uncond[i]);
}

void magpie_cfg_interpolate_device(const float* d_cond_logits, const float* d_uncond_logits,
                                   float* d_out_logits, float cfg_scale, int32_t num_elements,
                                   cudaStream_t stream) {
    constexpr int32_t kBlockSize = 256;
    const int32_t grid = (num_elements + kBlockSize - 1) / kBlockSize;

    cfg_interpolate_kernel<<<grid, kBlockSize, 0, stream>>>(d_cond_logits, d_uncond_logits,
                                                            d_out_logits, cfg_scale, num_elements);
}

} // namespace trtmc

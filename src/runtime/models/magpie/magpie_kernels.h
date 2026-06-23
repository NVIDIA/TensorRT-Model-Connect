#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace trtmc {

// GPU-side greedy argmax for multi-codebook logits.
// Finds argmax within [0, audio_range) and full [0, codebook_size) per codebook.
// d_logits:           [num_codebooks * codebook_size] float on device
// d_codes_out:        [num_codebooks] int32_t on device (argmax within audio range)
// d_full_argmax_out:  [num_codebooks] int32_t on device (argmax over full range)
void magpie_greedy_sample_device(const float* d_logits, int32_t num_codebooks,
                                 int32_t codebook_size, int32_t audio_range, int32_t* d_codes_out,
                                 int32_t* d_full_argmax_out, cudaStream_t stream);

// GPU-side gather + average of codebook embeddings.
// Gathers one embedding per codebook from prev_codes, averages them.
// d_audio_embed_table: [num_codebooks * codebook_size * hidden_size] float on device
// d_prev_codes:        [num_codebooks] int32_t on device
// d_output:            [hidden_size] float on device
void magpie_gather_average_embed_device(const float* d_audio_embed_table,
                                        const int32_t* d_prev_codes, int32_t num_codebooks,
                                        int32_t codebook_size, int32_t hidden_size, float* d_output,
                                        cudaStream_t stream);

// Scatter current frame's codes into an accumulator buffer and update prev_codes.
// d_codes:         [num_codebooks] int32_t — current frame argmax output
// d_all_codes:     [max_frames * num_codebooks] int32_t — accumulator (row-major)
// d_prev_codes:    [num_codebooks] int32_t — updated to d_codes for next iteration
// d_full_argmax:   [num_codebooks] int32_t — current frame full-range argmax
// d_eos_flag:      [1] int32_t — set to 1 if any codebook's full argmax == eos_token
void magpie_scatter_codes_device(const int32_t* d_codes, int32_t* d_all_codes,
                                 int32_t* d_prev_codes, const int32_t* d_full_argmax,
                                 int32_t* d_eos_flag, int32_t frame_idx, int32_t num_codebooks,
                                 int32_t eos_token, cudaStream_t stream);

// GPU-side top-k sampling for multi-codebook logits.
// For each codebook: finds top-k logits, applies temperature softmax,
// and samples from the distribution using the provided RNG seed.
// Also computes full-range argmax for EOS detection.
// d_logits:           [num_codebooks * codebook_size] float on device
// d_codes_out:        [num_codebooks] int32_t on device (sampled token within audio range)
// d_full_argmax_out:  [num_codebooks] int32_t on device (argmax over full range, for EOS)
// d_eos_flag:         [1] int32_t on device (set to 1 if any codebook argmax == eos_token)
// d_rand_vals: [num_codebooks] float in [0,1), generated on host with MT19937
void magpie_topk_sample_device(const float* d_logits, int32_t num_codebooks, int32_t codebook_size,
                               int32_t audio_range, int32_t top_k, float temperature,
                               int32_t eos_token, const float* d_rand_vals, int32_t* d_codes_out,
                               int32_t* d_full_argmax_out, int32_t* d_eos_flag,
                               cudaStream_t stream);

// CFG interpolation: out[i] = uncond[i] + scale * (cond[i] - uncond[i])
// Applied elementwise over num_elements logits on device.
void magpie_cfg_interpolate_device(const float* d_cond_logits, const float* d_uncond_logits,
                                   float* d_out_logits, float cfg_scale, int32_t num_elements,
                                   cudaStream_t stream);

} // namespace trtmc

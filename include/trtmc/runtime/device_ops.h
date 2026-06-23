#pragma once

// DeviceOps: standalone GPU utility kernels.
//
// These are reusable device-side building blocks for GPU-resident decode loops.
// They are independent of TRT engine execution (TrtModule handles that).
// Model pipelines choose whether to call these kernels from their owned loops.

#include <cstdint>
#include <cuda_runtime_api.h>

namespace trtmc {
namespace device_ops {

// Multi-codebook greedy argmax on device logits.
// d_logits: [num_codebooks * codebook_size] float on device
// d_codes: [num_codebooks] int32 — argmax within [0, audio_range)
// d_full_argmax: [num_codebooks] int32 — argmax over full [0, codebook_size)
void greedy_sample_codebooks(const float* d_logits, int32_t num_codebooks, int32_t codebook_size,
                             int32_t audio_range, int32_t* d_codes, int32_t* d_full_argmax,
                             cudaStream_t stream);

// Gather one embedding per codebook from table, average into output.
// d_embed_table: [num_entries * vocab_size * hidden_size] float on device
// d_token_ids: [num_entries] int32 on device
// d_output: [hidden_size] float on device
void gather_average_embeddings(const float* d_embed_table, const int32_t* d_token_ids,
                               int32_t num_entries, int32_t vocab_size, int32_t hidden_size,
                               float* d_output, cudaStream_t stream);

// Elementwise CFG interpolation: out = uncond + scale * (cond - uncond).
void cfg_interpolate(const float* d_cond, const float* d_uncond, float* d_out, float scale,
                     int32_t n, cudaStream_t stream);

// Scatter codes into accumulator, update prev_codes, check EOS.
// d_codes: [num_codebooks] int32 — current frame codes
// d_all_codes: [max_frames * num_codebooks] int32 — accumulator
// d_prev_codes: [num_codebooks] int32 — updated for next iteration
// d_full_argmax: [num_codebooks] int32 — full-range argmax for EOS check
// d_eos_flag: [1] int32 — set to 1 if any codebook hits eos_token
void scatter_codes_check_eos(const int32_t* d_codes, int32_t* d_all_codes, int32_t* d_prev_codes,
                             const int32_t* d_full_argmax, int32_t* d_eos_flag, int32_t frame,
                             int32_t num_codebooks, int32_t eos_token, cudaStream_t stream);

} // namespace device_ops
} // namespace trtmc

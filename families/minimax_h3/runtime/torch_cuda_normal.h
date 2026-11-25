/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector>

namespace trtmc::minimax_h3 {

struct VaeLatentNormalization {
    float mean[24];
    float std[24];
};

struct VaePixelNormalization {
    float mean[3];
    float std[3];
};

// Matches torch.randn(..., generator=torch.Generator().manual_seed(seed)) for
// contiguous float32 CPU tensors on AArch64. Calls are deliberately separate so
// video consumes the generator before audio, matching HF and Sol H3.
std::vector<float> torch_cuda_normal(std::size_t count, uint64_t seed, uint64_t offset = 0);
uint64_t torch_cuda_normal_consumed_offset(std::size_t count);

// Enqueue the MiniMax-H3 data-ward Euler update on `stream`, modifying the
// device-resident FP32 sample in place. Both pointers must be non-null (also for
// a zero count) and remain valid until the stream reaches this work; sigma must
// be positive. The function does not synchronize, so successful return only
// means CUDA accepted the launch. Invalid host arguments throw
// std::invalid_argument/std::overflow_error, while launch failures throw
// std::runtime_error. A zero count with valid pointers is a no-op.
void scheduler_step_cuda_async(float* sample, const float* velocity, std::size_t count,
                               float timestep, float sigma, float sigma_next, cudaStream_t stream);

// Extract one dynamic-canvas VAE tile batch directly from the denoiser's patched
// video rows. The kernel also applies the checkpoint latent mean/std transform,
// so no full latent unpatch is materialized on either host or device.
void extract_vae_tiles_cuda_async(const float* video_rows, float* latent_tiles, int32_t clip_index,
                                  int32_t clip_count, int32_t output_height, int32_t output_width,
                                  int32_t tile_rows, int32_t tile_columns,
                                  VaeLatentNormalization normalization, cudaStream_t stream);

// Spatially stitch one dynamic-canvas VAE output, assemble its 17 temporal
// frames (plus the final 5-frame tail), apply pixel normalization,
// clamp, and write frame-major RGB. `overlap` stores the previous clip's raw
// five-frame spatial overlap and is updated after it has been consumed.
void assemble_vae_clip_cuda_async(const float* decoded_tiles, float* overlap,
                                  float* frame_major_rgb, int32_t clip_index, int32_t clip_count,
                                  int32_t output_height, int32_t output_width, int32_t tile_rows,
                                  int32_t tile_columns, VaePixelNormalization normalization,
                                  cudaStream_t stream);

} // namespace trtmc::minimax_h3

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc::minimax_h3 {

struct ImageReferenceLatentShape {
    int32_t height;
    int32_t width;
};

struct ReferenceTargetLatentShape {
    int32_t frames;
    int32_t height;
    int32_t width;
    int32_t audio_latents; // Per waveform channel; the released model is stereo.
};

struct ImageReferenceLayout {
    // Preserve the released host layout's FP64 grid. The DiT's current FP32
    // position-input conversion is a separate engine boundary, not hidden here.
    std::vector<double> positions; // [sequence rows,3], T/H/W.
    std::vector<int32_t> token_tags;
    std::vector<int32_t> text_indices;
    std::vector<int32_t> audio_indices;
    std::vector<int32_t> video_indices;  // Image conditions first, generated last.
    std::vector<int32_t> packed_indices; // Gather concat(text,audio,video).
    std::vector<int32_t> target_video_indices;
    std::vector<int32_t> target_audio_indices;
    int32_t condition_video_rows{0};
};

// Image-only original Ref2VA layout. Reference dimensions are the actual VAE
// latent dimensions, not the target canvas or Qwen patch grid. max_rows is the
// loaded DiT profile cap: reject overflow, never truncate or resize references.
// Audio/video reference inputs must use their own encoder/layout path.
ImageReferenceLayout
make_image_reference_layout(const std::vector<int32_t>& text_tags,
                            const std::vector<ImageReferenceLatentShape>& images,
                            ReferenceTargetLatentShape target, int32_t max_rows);

struct ReferenceRowTimesteps {
    std::vector<float> timesteps; // Sorted, deduplicated original FP32 values.
    std::vector<int32_t> timestep_indices;
    std::vector<int32_t> adaln_indices; // timestep index * 3 + token tag.
};

// Text inherits the generated-video timestep, including vision-tagged Qwen
// rows. Image latents stay at max(video_timestep,0.999), not the audio time.
ReferenceRowTimesteps make_image_reference_timesteps(const ImageReferenceLayout& layout,
                                                     float video_timestep, float audio_timestep);

// Exact permutation [1,24,T,H,W] -> [T*H/2*W/2,96]; no normalization,
// resampling, noise draw or dtype conversion. Used independently per reference.
std::vector<float> pack_reference_video_rows(const std::vector<float>& latents, int32_t frames,
                                             int32_t height, int32_t width);

} // namespace trtmc::minimax_h3

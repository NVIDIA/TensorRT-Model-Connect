/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/image.h"

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::stable_video_diffusion {

// The initial image as contiguous HWC RGB float32 in [0, 1]. It must already
// have the bundle's video size, as the published usage resizes it before the
// call; UInt8 values are divided by 255 the way pil_to_numpy does.
std::vector<float> initial_image_pixels(const internal::ImageView& image, std::int32_t height,
                                        std::int32_t width);

// VaeImageProcessor.preprocess of an image already at the target size: CHW, 2x - 1.
std::vector<float> vae_image(const std::vector<float>& pixels, std::int32_t height,
                             std::int32_t width);

// StableVideoDiffusionPipeline._encode_image preprocessing: map to [-1, 1],
// Gaussian blur with reflect padding and an align-corners bicubic resize to
// size x size (_resize_with_antialiasing), map back to [0, 1], then the CLIP
// mean/std normalization. Returns CHW [3, size, size].
std::vector<float> clip_pixels(const std::vector<float>& pixels, std::int32_t height,
                               std::int32_t width, std::int32_t size,
                               const std::array<float, 3>& mean, const std::array<float, 3>& std);

} // namespace trtmc::stable_video_diffusion

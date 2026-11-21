/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/image.h"

#include <cstdint>
#include <vector>

namespace trtmc::wan2_2_ti2v {

// Upstream i2v's TF.to_tensor(image).sub_(0.5).div_(0.5): host HWC RGB (UInt8 or Float32) of
// exactly height x width to CHW values in [-1, 1].
std::vector<float> first_frame_pixels(const internal::ImageView& image, int32_t height,
                                      int32_t width);

// Upstream re-applies (1 - mask) * z + mask * latent after every scheduler step. With its 0/1
// mask that copies the encoded image z into latent frame 0 of every CTHW channel.
void keep_first_frame(std::vector<float>& latents, const std::vector<float>& first_frame,
                      int32_t latent_frames);

} // namespace trtmc::wan2_2_ti2v

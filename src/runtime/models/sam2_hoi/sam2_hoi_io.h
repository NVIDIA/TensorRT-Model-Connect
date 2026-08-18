/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::sam2_hoi {

inline constexpr int32_t kImageSize = 1024;
inline constexpr int32_t kLowResolutionMaskSize = 256;

namespace detail {

// Convert a value whose finite [0, 1] contract has already been validated.
// Keep the production float multiply, then widen before adding 0.5 so the
// truncation remains exactly equivalent to std::lround at every uint8 decision
// boundary. Adding 0.5 in float can prematurely round a value onto the next
// integer and is not equivalent for all valid VideoFrame inputs.
static inline uint8_t round_unit_float_to_u8(float value) {
    const float scaled = value * 255.0F;
    return static_cast<uint8_t>(static_cast<double>(scaled) + 0.5);
}

} // namespace detail

// Resize an RGB uint8 HWC image with Pillow's Image.resize(..., BICUBIC)
// contract. This lower-level seam is public so the fixed-point interpolation
// behavior can be verified without allocating a 1024x1024 tensor.
std::vector<uint8_t> resize_pillow_bicubic_rgb_u8(const uint8_t* rgb_hwc, int32_t source_height,
                                                  int32_t source_width, int32_t target_height,
                                                  int32_t target_width);

// Convert decoded RGB float HWC pixels in [0, 1] to a fixed [3, 1024, 1024]
// ImageNet-normalized tensor. The resize happens in uint8, as it does for a
// Pillow RGB image, before conversion to normalized float CHW.
std::vector<float> preprocess_image(const float* rgb_hwc, int32_t source_height,
                                    int32_t source_width);

// Fill only background (logit <= 0) 8-connected components whose area is at
// most max_area. Each mask is an independent contiguous [height, width] plane;
// a zero mask count is a valid no-op.
void fill_small_mask_holes(std::vector<float>& mask_logits, int32_t mask_count, int32_t height,
                           int32_t width, int32_t max_area = 8, float fill_value = 0.1F);

// Bilinearly resize [mask_count, 1, source_height, source_width] logits with
// align_corners=false and return uint8 [mask_count, 1, target_height,
// target_width] values using a strict logit > threshold decision. A zero mask
// count returns an empty vector.
std::vector<uint8_t> resize_and_threshold_masks(const float* low_res_logits, int32_t mask_count,
                                                int32_t source_height, int32_t source_width,
                                                int32_t target_height, int32_t target_width,
                                                float threshold = 0.01F);

// Write a deterministic NumPy v1.0 array with dtype uint8 and shape
// [mask_count, 1, height, width], including an empty first dimension. The error
// output is optional.
bool write_uint8_npy(const std::string& path, const std::vector<uint8_t>& masks, int32_t mask_count,
                     int32_t height, int32_t width, std::string* error = nullptr);

} // namespace trtmc::sam2_hoi

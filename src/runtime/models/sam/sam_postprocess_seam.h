/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/sam/sam_types.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace trtmc {

inline float sample_mask_bilinear(const std::vector<float>& mask, int32_t src_w, int32_t src_h,
                                  float x, float y) {
    if (src_w <= 0 || src_h <= 0 || mask.empty()) {
        return 0.0F;
    }

    const float clamped_x = std::clamp(x, 0.0F, static_cast<float>(src_w - 1));
    const float clamped_y = std::clamp(y, 0.0F, static_cast<float>(src_h - 1));
    const int32_t x0 = static_cast<int32_t>(std::floor(clamped_x));
    const int32_t y0 = static_cast<int32_t>(std::floor(clamped_y));
    const int32_t x1 = std::min(x0 + 1, src_w - 1);
    const int32_t y1 = std::min(y0 + 1, src_h - 1);
    const float wx = clamped_x - static_cast<float>(x0);
    const float wy = clamped_y - static_cast<float>(y0);

    const auto idx = [src_w](int32_t yy, int32_t xx) {
        return static_cast<std::size_t>(yy) * static_cast<std::size_t>(src_w) +
               static_cast<std::size_t>(xx);
    };

    const float v00 = mask[idx(y0, x0)];
    const float v01 = mask[idx(y0, x1)];
    const float v10 = mask[idx(y1, x0)];
    const float v11 = mask[idx(y1, x1)];
    const float top = v00 * (1.0F - wx) + v01 * wx;
    const float bottom = v10 * (1.0F - wx) + v11 * wx;
    return top * (1.0F - wy) + bottom * wy;
}

inline std::vector<float> resize_mask_bilinear(const std::vector<float>& mask, int32_t src_w,
                                               int32_t src_h, int32_t dst_w, int32_t dst_h) {
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return {};
    }
    if (src_w == dst_w && src_h == dst_h) {
        return mask;
    }

    std::vector<float> resized(static_cast<std::size_t>(dst_w) * static_cast<std::size_t>(dst_h),
                               0.0F);
    const float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    const float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);
    for (int32_t y = 0; y < dst_h; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5F) * scale_y - 0.5F;
        for (int32_t x = 0; x < dst_w; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5F) * scale_x - 0.5F;
            resized[static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_w) +
                    static_cast<std::size_t>(x)] =
                sample_mask_bilinear(mask, src_w, src_h, src_x, src_y);
        }
    }
    return resized;
}

inline bool has_valid_sam_postprocess_request(const SamResult& result, int32_t image_size,
                                              int32_t rescaled_w, int32_t rescaled_h,
                                              int32_t original_w, int32_t original_h) {
    return result.num_masks > 0 && result.mask_width > 0 && result.mask_height > 0 &&
           image_size > 0 && rescaled_w > 0 && rescaled_h > 0 && original_w > 0 && original_h > 0;
}

inline bool has_complete_sam_mask_payload(const SamResult& result) {
    const auto mask_size =
        static_cast<std::size_t>(result.mask_width) * static_cast<std::size_t>(result.mask_height);
    return result.masks.size() >= static_cast<std::size_t>(result.num_masks) * mask_size;
}

inline std::vector<float> extract_sam_mask(const SamResult& result, int32_t mask_index) {
    const auto mask_size =
        static_cast<std::size_t>(result.mask_width) * static_cast<std::size_t>(result.mask_height);
    const auto begin = result.masks.begin() + static_cast<std::ptrdiff_t>(mask_index) *
                                                  static_cast<std::ptrdiff_t>(mask_size);
    return std::vector<float>(begin, begin + static_cast<std::ptrdiff_t>(mask_size));
}

inline std::vector<float> crop_rescaled_sam_mask(const std::vector<float>& upsampled,
                                                 int32_t image_size, int32_t rescaled_w,
                                                 int32_t rescaled_h) {
    std::vector<float> cropped(
        static_cast<std::size_t>(rescaled_w) * static_cast<std::size_t>(rescaled_h), 0.0F);
    for (int32_t y = 0; y < rescaled_h; ++y) {
        const auto* src_row =
            upsampled.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(image_size);
        auto* dst_row =
            cropped.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(rescaled_w);
        std::copy_n(src_row, rescaled_w, dst_row);
    }
    return cropped;
}

inline std::vector<float> postprocess_single_sam_mask(const SamResult& result, int32_t mask_index,
                                                      int32_t image_size, int32_t rescaled_w,
                                                      int32_t rescaled_h, int32_t original_w,
                                                      int32_t original_h) {
    auto upsampled = resize_mask_bilinear(extract_sam_mask(result, mask_index), result.mask_width,
                                          result.mask_height, image_size, image_size);
    return resize_mask_bilinear(
        crop_rescaled_sam_mask(upsampled, image_size, rescaled_w, rescaled_h), rescaled_w,
        rescaled_h, original_w, original_h);
}

inline SamResult postprocess_sam_result(SamResult result, int32_t image_size, int32_t rescaled_w,
                                        int32_t rescaled_h, int32_t original_w,
                                        int32_t original_h) {
    if (!has_valid_sam_postprocess_request(result, image_size, rescaled_w, rescaled_h, original_w,
                                           original_h)) {
        return result;
    }
    if (!has_complete_sam_mask_payload(result)) {
        return result;
    }

    std::vector<float> processed;
    processed.reserve(static_cast<std::size_t>(result.num_masks) *
                      static_cast<std::size_t>(original_w) * static_cast<std::size_t>(original_h));

    for (int32_t mask_index = 0; mask_index < result.num_masks; ++mask_index) {
        auto restored = postprocess_single_sam_mask(result, mask_index, image_size, rescaled_w,
                                                    rescaled_h, original_w, original_h);
        processed.insert(processed.end(), restored.begin(), restored.end());
    }

    result.masks = std::move(processed);
    result.mask_width = original_w;
    result.mask_height = original_h;
    return result;
}

} // namespace trtmc

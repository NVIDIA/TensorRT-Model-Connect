/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/sam/runtime/sam_bilinear.h"
#include "families/sam/runtime/sam_types.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

// One axis of a mask resize from `src_len` to `dst_len` samples (the
// half-pixel-center mapping), limited to its first `count` destination indices.
inline SamBilinearAxis make_sam_mask_resize_axis(int32_t src_len, int32_t dst_len, int32_t count) {
    const float scale = static_cast<float>(src_len) / static_cast<float>(dst_len);
    return make_sam_bilinear_axis(src_len, count, [scale](int32_t i) {
        return (static_cast<float>(i) + 0.5F) * scale - 0.5F;
    });
}

// sample_mask_bilinear at every destination pixel, computed in separable
// passes (sam_bilinear.h).
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
    if (mask.empty()) {
        return resized;
    }
    sam_bilinear_resize_plane(mask.data(), src_w, make_sam_mask_resize_axis(src_w, dst_w, dst_w),
                              make_sam_mask_resize_axis(src_h, dst_h, dst_h), resized.data(),
                              static_cast<std::size_t>(dst_w));
    return resized;
}

inline bool has_valid_sam_postprocess_request(const SamResult& result, int32_t image_size,
                                              int32_t rescaled_w, int32_t rescaled_h,
                                              int32_t original_w, int32_t original_h) {
    // The rescaled image lies inside the image_size x image_size square the
    // decoder masks cover.
    return result.num_masks > 0 && result.mask_width > 0 && result.mask_height > 0 &&
           image_size > 0 && rescaled_w > 0 && rescaled_h > 0 && rescaled_w <= image_size &&
           rescaled_h <= image_size && original_w > 0 && original_h > 0;
}

inline bool has_complete_sam_mask_payload(const SamResult& result) {
    const auto mask_size =
        static_cast<std::size_t>(result.mask_width) * static_cast<std::size_t>(result.mask_height);
    return result.masks.size() >= static_cast<std::size_t>(result.num_masks) * mask_size;
}

// Restores each mask to the original image: the image_size x image_size
// upsample of the decoder mask, cropped to the rescaled image's top-left
// rescaled_w x rescaled_h region, then resized to original_w x original_h.
// Only the upsampled rows and columns the crop keeps are computed, and each
// mask is written straight into the result.
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

    const int32_t mask_w = result.mask_width;
    const int32_t mask_h = result.mask_height;
    const auto mask_area = static_cast<std::size_t>(mask_w) * static_cast<std::size_t>(mask_h);
    const auto crop_w = static_cast<std::size_t>(rescaled_w);
    const auto crop_area = crop_w * static_cast<std::size_t>(rescaled_h);
    const auto original_area =
        static_cast<std::size_t>(original_w) * static_cast<std::size_t>(original_h);

    // resize_mask_bilinear returns its input when the size does not change.
    const bool upsample_is_copy = mask_w == image_size && mask_h == image_size;
    const bool restore_is_copy = rescaled_w == original_w && rescaled_h == original_h;
    SamBilinearAxis upsample_x;
    SamBilinearAxis upsample_y;
    if (!upsample_is_copy) {
        upsample_x = make_sam_mask_resize_axis(mask_w, image_size, rescaled_w);
        upsample_y = make_sam_mask_resize_axis(mask_h, image_size, rescaled_h);
    }
    SamBilinearAxis restore_x;
    SamBilinearAxis restore_y;
    if (!restore_is_copy) {
        restore_x = make_sam_mask_resize_axis(rescaled_w, original_w, original_w);
        restore_y = make_sam_mask_resize_axis(rescaled_h, original_h, original_h);
    }

    std::vector<float> processed(static_cast<std::size_t>(result.num_masks) * original_area);
    std::vector<float> cropped(restore_is_copy ? 0 : crop_area);
    for (int32_t mask_index = 0; mask_index < result.num_masks; ++mask_index) {
        const float* mask = result.masks.data() + static_cast<std::size_t>(mask_index) * mask_area;
        float* restored = processed.data() + static_cast<std::size_t>(mask_index) * original_area;
        float* crop = restore_is_copy ? restored : cropped.data();
        if (upsample_is_copy) {
            for (int32_t y = 0; y < rescaled_h; ++y) {
                std::copy_n(mask + static_cast<std::size_t>(y) * static_cast<std::size_t>(mask_w),
                            crop_w, crop + static_cast<std::size_t>(y) * crop_w);
            }
        } else {
            sam_bilinear_resize_plane(mask, mask_w, upsample_x, upsample_y, crop, crop_w);
        }
        if (!restore_is_copy) {
            sam_bilinear_resize_plane(cropped.data(), rescaled_w, restore_x, restore_y, restored,
                                      static_cast<std::size_t>(original_w));
        }
    }

    result.masks = std::move(processed);
    result.mask_width = original_w;
    result.mask_height = original_h;
    return result;
}

} // namespace trtmc

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc {

struct ImageNormalizationParams {
    int32_t width{0};
    int32_t height{0};
    int32_t channels{3};
    float image_mean[3]{0.0F, 0.0F, 0.0F};
    float image_std[3]{1.0F, 1.0F, 1.0F};
};

enum class ImageTransformLayout { kSimpleChw, kMergeGroupChw };

struct ImageTransformParams {
    ImageTransformLayout layout{ImageTransformLayout::kSimpleChw};
    int32_t target_height{0};
    int32_t target_width{0};
    int32_t channels{3};
    int32_t patch_size{14};
    int32_t merge_size{2};
    int32_t temporal_patch_size{1};
};

inline bool normalize_hwc_u8_to_chw(const std::vector<unsigned char>& image_hwc,
                                    const ImageNormalizationParams& params,
                                    std::vector<float>& out_chw) {
    if (params.width <= 0 || params.height <= 0 || params.channels <= 0 || params.channels > 3) {
        return false;
    }

    const std::size_t pixel_count = static_cast<std::size_t>(params.width) * params.height;
    const std::size_t required_size = pixel_count * static_cast<std::size_t>(params.channels);
    if (image_hwc.size() < required_size) {
        return false;
    }

    out_chw.resize(required_size);
    for (int32_t c = 0; c < params.channels; ++c) {
        const float mean = params.image_mean[c];
        const float std = params.image_std[c];
        const float inv_std = (std > 1e-8F) ? (1.0F / std) : 1.0F;

        for (int32_t y = 0; y < params.height; ++y) {
            for (int32_t x = 0; x < params.width; ++x) {
                const std::size_t src_idx =
                    (static_cast<std::size_t>(y) * params.width + x) * params.channels + c;
                const float pixel = static_cast<float>(image_hwc[src_idx]) / 255.0F;

                const std::size_t dst_idx = static_cast<std::size_t>(c) * pixel_count +
                                            static_cast<std::size_t>(y) * params.width + x;
                out_chw[dst_idx] = (pixel - mean) * inv_std;
            }
        }
    }

    return true;
}

inline void copy_merge_patch_chw(const std::vector<float>& input_chw,
                                 const ImageTransformParams& params, std::size_t pixel_count,
                                 int32_t orig_h, int32_t orig_w, int32_t dst_h, int32_t dst_w,
                                 std::vector<float>& out_values) {
    for (int32_t c = 0; c < params.channels; ++c) {
        for (int32_t t = 0; t < params.temporal_patch_size; ++t) {
            const int32_t out_ch = c * params.temporal_patch_size + t;
            for (int32_t py = 0; py < params.patch_size; ++py) {
                for (int32_t px = 0; px < params.patch_size; ++px) {
                    const std::size_t src =
                        static_cast<std::size_t>(c) * pixel_count +
                        static_cast<std::size_t>(orig_h * params.patch_size + py) *
                            params.target_width +
                        (orig_w * params.patch_size + px);
                    const std::size_t dst =
                        static_cast<std::size_t>(out_ch) * pixel_count +
                        static_cast<std::size_t>(dst_h * params.patch_size + py) *
                            params.target_width +
                        (dst_w * params.patch_size + px);
                    out_values[dst] = input_chw[src];
                }
            }
        }
    }
}

inline void copy_merge_group_chw(const std::vector<float>& input_chw,
                                 const ImageTransformParams& params, std::size_t pixel_count,
                                 int32_t grid_w, int32_t merge_h, int32_t merge_w,
                                 std::vector<float>& out_values) {
    int32_t dst_patch_idx = 0;
    for (int32_t mh = 0; mh < merge_h; ++mh) {
        for (int32_t mw = 0; mw < merge_w; ++mw) {
            for (int32_t dh = 0; dh < params.merge_size; ++dh) {
                for (int32_t dw = 0; dw < params.merge_size; ++dw) {
                    const int32_t orig_h = mh * params.merge_size + dh;
                    const int32_t orig_w = mw * params.merge_size + dw;
                    const int32_t dst_h = dst_patch_idx / grid_w;
                    const int32_t dst_w = dst_patch_idx % grid_w;
                    copy_merge_patch_chw(input_chw, params, pixel_count, orig_h, orig_w, dst_h,
                                         dst_w, out_values);
                    ++dst_patch_idx;
                }
            }
        }
    }
}

inline bool transform_chw_layout(const std::vector<float>& input_chw,
                                 const ImageTransformParams& params, std::vector<float>& out_values,
                                 int32_t& out_channels) {
    out_channels = 0;
    if (params.target_height <= 0 || params.target_width <= 0 || params.channels <= 0) {
        return false;
    }

    const std::size_t pixel_count =
        static_cast<std::size_t>(params.target_height) * params.target_width;
    const std::size_t required_size = static_cast<std::size_t>(params.channels) * pixel_count;
    if (input_chw.size() < required_size) {
        return false;
    }

    if (params.layout == ImageTransformLayout::kSimpleChw) {
        out_values.assign(input_chw.begin(), input_chw.begin() + required_size);
        out_channels = params.channels;
        return true;
    }

    if (params.patch_size <= 0 || params.merge_size <= 0 || params.temporal_patch_size <= 0) {
        return false;
    }

    if (params.target_height % params.patch_size != 0 ||
        params.target_width % params.patch_size != 0) {
        return false;
    }
    const int32_t grid_h = params.target_height / params.patch_size;
    const int32_t grid_w = params.target_width / params.patch_size;
    if (grid_h % params.merge_size != 0 || grid_w % params.merge_size != 0) {
        return false;
    }
    const int32_t merge_h = grid_h / params.merge_size;
    const int32_t merge_w = grid_w / params.merge_size;
    const int32_t total_channels = params.channels * params.temporal_patch_size;

    out_values.assign(static_cast<std::size_t>(total_channels) * pixel_count, 0.0F);
    out_channels = total_channels;

    copy_merge_group_chw(input_chw, params, pixel_count, grid_w, merge_h, merge_w, out_values);

    return true;
}

} // namespace trtmc

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace trtmc {

struct SegformerLogitsShape {
    int32_t num_classes{0};
    int32_t output_h{0};
    int32_t output_w{0};
};

enum class SegformerPostprocessStatus {
    kOk = 0,
    kInvalidShape = 1,
    kLogitsSizeMismatch = 2,
};

inline bool get_segformer_logits_expected_size(const SegformerLogitsShape& shape,
                                               std::size_t& expected_size) {
    expected_size = 0;
    if (shape.num_classes <= 0 || shape.output_h <= 0 || shape.output_w <= 0) {
        return false;
    }

    const std::size_t classes = static_cast<std::size_t>(shape.num_classes);
    const std::size_t output_h = static_cast<std::size_t>(shape.output_h);
    const std::size_t output_w = static_cast<std::size_t>(shape.output_w);
    constexpr std::size_t kMaxSize = std::numeric_limits<std::size_t>::max();

    if (classes > (kMaxSize / output_h)) {
        return false;
    }
    const std::size_t classes_by_h = classes * output_h;
    if (classes_by_h > (kMaxSize / output_w)) {
        return false;
    }

    expected_size = classes_by_h * output_w;
    return true;
}

inline SegformerPostprocessStatus
compute_segformer_class_map_from_logits(const std::vector<float>& logits,
                                        const SegformerLogitsShape& shape,
                                        std::vector<int32_t>& class_map) {
    std::size_t expected_logits_size = 0;
    if (!get_segformer_logits_expected_size(shape, expected_logits_size)) {
        class_map.clear();
        return SegformerPostprocessStatus::kInvalidShape;
    }

    if (logits.size() != expected_logits_size) {
        class_map.clear();
        return SegformerPostprocessStatus::kLogitsSizeMismatch;
    }

    const int32_t num_classes = shape.num_classes;
    const int32_t output_h = shape.output_h;
    const int32_t output_w = shape.output_w;
    const std::size_t plane_size =
        static_cast<std::size_t>(output_h) * static_cast<std::size_t>(output_w);

    class_map.resize(plane_size);
    for (int32_t y = 0; y < output_h; ++y) {
        for (int32_t x = 0; x < output_w; ++x) {
            const std::size_t pixel_index =
                static_cast<std::size_t>(y) * static_cast<std::size_t>(output_w) +
                static_cast<std::size_t>(x);

            int32_t best_class = 0;
            float best_val = -1e30F;
            for (int32_t c = 0; c < num_classes; ++c) {
                const float val = logits[static_cast<std::size_t>(c) * plane_size + pixel_index];
                if (val > best_val) {
                    best_val = val;
                    best_class = c;
                }
            }

            class_map[pixel_index] = best_class;
        }
    }

    return SegformerPostprocessStatus::kOk;
}

inline SegformerPostprocessStatus
resize_segformer_logits_and_compute_class_map(const std::vector<float>& logits,
                                              const SegformerLogitsShape& shape, int32_t target_h,
                                              int32_t target_w, std::vector<int32_t>& class_map) {
    std::size_t expected_logits_size = 0;
    if (!get_segformer_logits_expected_size(shape, expected_logits_size) || target_h <= 0 ||
        target_w <= 0) {
        class_map.clear();
        return SegformerPostprocessStatus::kInvalidShape;
    }
    if (logits.size() != expected_logits_size) {
        class_map.clear();
        return SegformerPostprocessStatus::kLogitsSizeMismatch;
    }
    if (target_h == shape.output_h && target_w == shape.output_w)
        return compute_segformer_class_map_from_logits(logits, shape, class_map);

    const auto source_plane = static_cast<std::size_t>(shape.output_h) * shape.output_w;
    class_map.resize(static_cast<std::size_t>(target_h) * target_w);
    for (int32_t y = 0; y < target_h; ++y) {
        const float source_y = (static_cast<float>(y) + 0.5F) * static_cast<float>(shape.output_h) /
                                   static_cast<float>(target_h) -
                               0.5F;
        const float clamped_y = std::clamp(source_y, 0.0F, static_cast<float>(shape.output_h - 1));
        const int32_t y0 = static_cast<int32_t>(std::floor(clamped_y));
        const int32_t y1 = std::min(y0 + 1, shape.output_h - 1);
        const float y_weight = clamped_y - static_cast<float>(y0);
        for (int32_t x = 0; x < target_w; ++x) {
            const float source_x = (static_cast<float>(x) + 0.5F) *
                                       static_cast<float>(shape.output_w) /
                                       static_cast<float>(target_w) -
                                   0.5F;
            const float clamped_x =
                std::clamp(source_x, 0.0F, static_cast<float>(shape.output_w - 1));
            const int32_t x0 = static_cast<int32_t>(std::floor(clamped_x));
            const int32_t x1 = std::min(x0 + 1, shape.output_w - 1);
            const float x_weight = clamped_x - static_cast<float>(x0);

            int32_t best_class = 0;
            float best_value = -std::numeric_limits<float>::infinity();
            for (int32_t c = 0; c < shape.num_classes; ++c) {
                const auto base = static_cast<std::size_t>(c) * source_plane;
                const auto at = [&](int32_t yy, int32_t xx) {
                    return logits[base + static_cast<std::size_t>(yy) * shape.output_w + xx];
                };
                const float top = at(y0, x0) * (1.0F - x_weight) + at(y0, x1) * x_weight;
                const float bottom = at(y1, x0) * (1.0F - x_weight) + at(y1, x1) * x_weight;
                const float value = top * (1.0F - y_weight) + bottom * y_weight;
                if (value > best_value) {
                    best_value = value;
                    best_class = c;
                }
            }
            class_map[static_cast<std::size_t>(y) * target_w + x] = best_class;
        }
    }
    return SegformerPostprocessStatus::kOk;
}

} // namespace trtmc

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

namespace segformer_detail {

struct BilinearSpan {
    int32_t first{0};
    int32_t second{0};
    float lerp{0.0F};
};

inline std::vector<BilinearSpan> make_bilinear_spans(int32_t source_size, int32_t target_size) {
    std::vector<BilinearSpan> spans(static_cast<std::size_t>(target_size));
    const float scale = static_cast<float>(source_size) / static_cast<float>(target_size);
    for (int32_t target_index = 0; target_index < target_size; ++target_index) {
        const float source_index = (static_cast<float>(target_index) + 0.5F) * scale - 0.5F;
        const int32_t first_unclamped = static_cast<int32_t>(std::floor(source_index));
        spans[static_cast<std::size_t>(target_index)] = {
            std::clamp(first_unclamped, 0, source_size - 1),
            std::clamp(first_unclamped + 1, 0, source_size - 1),
            source_index - static_cast<float>(first_unclamped),
        };
    }
    return spans;
}

inline void resize_class_horizontally(const std::vector<float>& logits, int32_t class_index,
                                      int32_t source_h, int32_t source_w,
                                      const std::vector<BilinearSpan>& x_spans,
                                      std::vector<float>& horizontal) {
    const std::size_t source_plane_size =
        static_cast<std::size_t>(source_h) * static_cast<std::size_t>(source_w);
    const std::size_t target_width = x_spans.size();
    const std::size_t class_offset = static_cast<std::size_t>(class_index) * source_plane_size;
    for (int32_t source_y = 0; source_y < source_h; ++source_y) {
        const std::size_t source_row = class_offset + static_cast<std::size_t>(source_y) * source_w;
        const std::size_t horizontal_row = static_cast<std::size_t>(source_y) * target_width;
        for (std::size_t target_x = 0; target_x < target_width; ++target_x) {
            const auto& span = x_spans[target_x];
            horizontal[horizontal_row + target_x] =
                logits[source_row + static_cast<std::size_t>(span.first)] * (1.0F - span.lerp) +
                logits[source_row + static_cast<std::size_t>(span.second)] * span.lerp;
        }
    }
}

inline void update_class_map(const std::vector<float>& horizontal, int32_t class_index,
                             std::size_t target_width, const std::vector<BilinearSpan>& y_spans,
                             std::vector<float>& best_values, std::vector<int32_t>& class_map) {
    for (std::size_t target_y = 0; target_y < y_spans.size(); ++target_y) {
        const auto& span = y_spans[target_y];
        const std::size_t first_row = static_cast<std::size_t>(span.first) * target_width;
        const std::size_t second_row = static_cast<std::size_t>(span.second) * target_width;
        const std::size_t target_row = target_y * target_width;
        for (std::size_t target_x = 0; target_x < target_width; ++target_x) {
            const std::size_t target_index = target_row + target_x;
            const float value = horizontal[first_row + target_x] * (1.0F - span.lerp) +
                                horizontal[second_row + target_x] * span.lerp;
            if (value > best_values[target_index]) {
                best_values[target_index] = value;
                class_map[target_index] = class_index;
            }
        }
    }
}

} // namespace segformer_detail

inline SegformerPostprocessStatus
compute_segformer_class_map_from_logits(const std::vector<float>& logits,
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

    const int32_t num_classes = shape.num_classes;
    const std::size_t target_height = static_cast<std::size_t>(target_h);
    const std::size_t target_width = static_cast<std::size_t>(target_w);
    constexpr std::size_t kMaxSize = std::numeric_limits<std::size_t>::max();
    const std::size_t source_height = static_cast<std::size_t>(shape.output_h);
    if (target_height > (kMaxSize / target_width) || source_height > (kMaxSize / target_width)) {
        class_map.clear();
        return SegformerPostprocessStatus::kInvalidShape;
    }

    const int32_t source_h = shape.output_h;
    const int32_t source_w = shape.output_w;
    const auto x_spans = segformer_detail::make_bilinear_spans(source_w, target_w);
    const auto y_spans = segformer_detail::make_bilinear_spans(source_h, target_h);
    const std::size_t target_plane_size = target_height * target_width;
    std::vector<float> best_values(target_plane_size, -1e30F);
    std::vector<float> horizontal(source_height * target_width);
    class_map.assign(target_plane_size, 0);

    for (int32_t c = 0; c < num_classes; ++c) {
        segformer_detail::resize_class_horizontally(logits, c, source_h, source_w, x_spans,
                                                    horizontal);
        segformer_detail::update_class_map(horizontal, c, target_width, y_spans, best_values,
                                           class_map);
    }

    return SegformerPostprocessStatus::kOk;
}

inline SegformerPostprocessStatus
compute_segformer_class_map_from_logits(const std::vector<float>& logits,
                                        const SegformerLogitsShape& shape,
                                        std::vector<int32_t>& class_map) {
    return compute_segformer_class_map_from_logits(logits, shape, shape.output_h, shape.output_w,
                                                   class_map);
}

} // namespace trtmc

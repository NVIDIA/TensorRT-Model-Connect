/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/deformable_detr/runtime/box_decode.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace trtmc {

std::vector<DeformableDetrDetection>
decode_deformable_detr_boxes(const float* logits, const float* boxes, std::int32_t num_queries,
                             std::int32_t num_classes, std::int32_t image_height,
                             std::int32_t image_width, float threshold, std::int32_t top_k) {
    if (logits == nullptr || boxes == nullptr)
        throw std::invalid_argument("deformable_detr box decoding received null outputs");
    if (num_queries <= 0 || num_classes <= 0)
        throw std::invalid_argument("deformable_detr box decoding needs a positive output shape");
    if (top_k <= 0)
        throw std::invalid_argument("deformable_detr box decoding needs a positive top_k");

    const auto total = static_cast<std::size_t>(num_queries) * num_classes;
    std::vector<std::size_t> order(total);
    std::iota(order.begin(), order.end(), std::size_t{0});
    const auto keep = std::min<std::size_t>(total, static_cast<std::size_t>(top_k));
    // Ranking on the logit is equivalent to ranking on its sigmoid, which is
    // monotonic, so the transform is applied only to the survivors.
    std::partial_sort(
        order.begin(), order.begin() + static_cast<std::ptrdiff_t>(keep), order.end(),
        [logits](std::size_t left, std::size_t right) { return logits[left] > logits[right]; });

    const float width = static_cast<float>(image_width);
    const float height = static_cast<float>(image_height);
    std::vector<DeformableDetrDetection> out;
    out.reserve(keep);
    for (std::size_t rank = 0; rank < keep; ++rank) {
        const std::size_t flat = order[rank];
        const float score = 1.0F / (1.0F + std::exp(-logits[flat]));
        if (score <= threshold)
            break;
        const auto query = static_cast<std::size_t>(flat / static_cast<std::size_t>(num_classes));
        const auto label = static_cast<std::int32_t>(flat % static_cast<std::size_t>(num_classes));
        const float* box = boxes + query * 4U;
        const float cx = box[0] * width;
        const float cy = box[1] * height;
        const float half_w = box[2] * width * 0.5F;
        const float half_h = box[3] * height * 0.5F;
        out.push_back(DeformableDetrDetection{score, label, cx - half_w, cy - half_h, cx + half_w,
                                              cy + half_h});
    }
    return out;
}

} // namespace trtmc

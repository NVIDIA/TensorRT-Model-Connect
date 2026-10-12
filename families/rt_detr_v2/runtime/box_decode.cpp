/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/rt_detr_v2/runtime/box_decode.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace trtmc {

std::vector<RtDetrDetection> decode_rt_detr_boxes(const float* logits, const float* boxes,
                                                  int32_t num_queries, int32_t num_classes,
                                                  int32_t image_height, int32_t image_width,
                                                  float threshold) {
    if (logits == nullptr || boxes == nullptr)
        throw std::invalid_argument("rt_detr_v2 box decoding received null outputs");
    if (num_queries <= 0 || num_classes <= 0)
        throw std::invalid_argument("rt_detr_v2 box decoding needs a positive output shape");

    const auto total = static_cast<std::size_t>(num_queries) * num_classes;
    std::vector<std::size_t> order(total);
    std::iota(order.begin(), order.end(), std::size_t{0});
    // Keep as many detections as there are queries, matching num_top_queries.
    const auto keep = std::min<std::size_t>(total, static_cast<std::size_t>(num_queries));
    std::partial_sort(
        order.begin(), order.begin() + static_cast<std::ptrdiff_t>(keep), order.end(),
        [logits](std::size_t left, std::size_t right) { return logits[left] > logits[right]; });

    const float width = static_cast<float>(image_width);
    const float height = static_cast<float>(image_height);
    std::vector<RtDetrDetection> out;
    out.reserve(keep);
    for (std::size_t rank = 0; rank < keep; ++rank) {
        const std::size_t flat = order[rank];
        const float score = 1.0F / (1.0F + std::exp(-logits[flat]));
        if (score < threshold)
            break;
        const auto query = static_cast<std::size_t>(flat / static_cast<std::size_t>(num_classes));
        const auto label = static_cast<int32_t>(flat % static_cast<std::size_t>(num_classes));
        const float* box = boxes + query * 4U;
        const float cx = box[0] * width;
        const float cy = box[1] * height;
        const float half_w = box[2] * width * 0.5F;
        const float half_h = box[3] * height * 0.5F;
        out.push_back(
            RtDetrDetection{score, label, cx - half_w, cy - half_h, cx + half_w, cy + half_h});
    }
    return out;
}

} // namespace trtmc

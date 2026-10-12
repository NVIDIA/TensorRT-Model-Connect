/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct DeformableDetrDetection {
    float score{0.0F};
    std::int32_t label{0};
    float x_min{0.0F};
    float y_min{0.0F};
    float x_max{0.0F};
    float y_max{0.0F};
};

// Turn the head's raw outputs into absolute xyxy detections.
//
// Read from DeformableDetrImageProcessor::post_process_object_detection:
//
//   * selection is top-k over the flattened query-by-class score matrix, not a
//     per-query argmax. The two agree whenever each winning query has one
//     dominant class, so a single-object image cannot tell them apart;
//   * the cut is top_k = 100, NOT the query count. RT-DETR keeps one entry per
//     query (300); copying that number here would admit 200 extra detections;
//   * scores are sigmoid, not softmax;
//   * boxes are cxcywh in [0, 1] and are scaled by the ORIGINAL image size,
//     not the square the network ran on.
std::vector<DeformableDetrDetection>
decode_deformable_detr_boxes(const float* logits, const float* boxes, std::int32_t num_queries,
                             std::int32_t num_classes, std::int32_t image_height,
                             std::int32_t image_width, float threshold, std::int32_t top_k);

} // namespace trtmc

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct RtDetrDetection {
    float score{0.0F};
    int32_t label{0};
    float x_min{0.0F};
    float y_min{0.0F};
    float x_max{0.0F};
    float y_max{0.0F};
};

// Turn the head's raw outputs into absolute xyxy detections.
//
// The selection rule is top-k over the flattened query-by-class score matrix,
// not a per-query argmax. The two agree whenever every winning query has a
// single dominant class, which is why a one-object image cannot tell them
// apart; they differ when one query scores highly for two classes, where this
// rule emits that box twice under different labels. It was read from the
// reference post-processor rather than inferred from outputs.
//
// Scores are sigmoid, not softmax, and boxes are scaled by the ORIGINAL image
// size rather than the square the network ran on.
std::vector<RtDetrDetection> decode_rt_detr_boxes(const float* logits, const float* boxes,
                                                  int32_t num_queries, int32_t num_classes,
                                                  int32_t image_height, int32_t image_width,
                                                  float threshold);

} // namespace trtmc

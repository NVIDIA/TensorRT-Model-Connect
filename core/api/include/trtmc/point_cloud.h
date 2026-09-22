/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TRTMC_POINT_CLOUD_H
#define TRTMC_POINT_CLOUD_H

#include "trtmc/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Row-major [num_points, input_dim] float32 coordinates. */
    const float* points;
    uint64_t num_points;
    uint32_t input_dim;
} trtmc_points_to_semantic_segmentation_request_v1;

typedef struct {
    /* Per-point class labels in the input point order. */
    const int32_t* labels;
    uint64_t num_points;
    uint32_t num_classes;
    trtmc_i32_view class_ids;
    trtmc_strings_view class_names;
    /* Empty means unknown; class IDs remain model-local. */
    trtmc_string_view vocabulary_id;
    uint32_t has_ignore_label;
    int32_t ignore_label;
    uint32_t has_background_label;
    int32_t background_label;
    const float* class_scores; /* Optional [num_points, num_classes]. */
    uint64_t class_score_count;
    uint32_t score_kind;
} trtmc_points_to_semantic_segmentation_view_v1;

#define TRTMC_TASK_POINTS_TO_SEMANTIC_SEGMENTATION "points_to_semantic_segmentation"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_points_to_semantic_segmentation_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*,
                                          trtmc_points_to_semantic_segmentation_view_v1*,
                                          trtmc_error**);
} trtmc_points_to_semantic_segmentation_api_v1;

#ifdef __cplusplus
}
#endif

#endif

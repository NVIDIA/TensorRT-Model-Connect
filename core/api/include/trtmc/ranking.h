/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TRTMC_RANKING_H
#define TRTMC_RANKING_H
#include "trtmc/matrix.h"
#include "trtmc/scores.h"
#ifdef __cplusplus
extern "C" {
#endif

enum { TRTMC_RANKING_FLOAT32 = 1, TRTMC_RANKING_INT32 = 2, TRTMC_RANKING_INT64 = 3 };
/* Contiguous host tensor. Names are unique and nonempty, rank is positive,
 * extents are nonnegative and byte_size equals product(shape)*element_size.
 * All pointers borrow through synchronous run. Core validates storage only;
 * the family owns required names, axes, batch/profile, values and padding. */
typedef struct {
    trtmc_string_view name;
    uint32_t type;
    const int64_t* shape;
    uint64_t rank;
    const void* data;
    uint64_t byte_size;
} trtmc_ranking_feature_v1;
typedef struct {
    const trtmc_ranking_feature_v1* features;
    uint64_t feature_count;
} trtmc_structured_ranking_request_v1;
/* Rows and columns retain family-defined input row/candidate order, including
 * padding, with positive matrix dimensions; Core does not sort scores, select candidates or remove
 * padding. Nested storage borrows the owned result lifetime. */
typedef struct {
    trtmc_f32_matrix_view_v1 scores;
    uint32_t kind;
} trtmc_candidate_scores_view_v1;
typedef struct {
    uint32_t supported;
    uint64_t budget_bytes, reserved_bytes, used_bytes, peak_bytes;
    uint64_t active_sequences, allocation_failures;
} trtmc_ranking_cache_stats_v1;

#define TRTMC_TASK_STRUCTURED_RANKING "structured_ranking"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_structured_ranking_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_candidate_scores_view_v1*,
                                          trtmc_error**);
    /* Idle statistics are copied into out. Families without a cache report
     * supported=0; configuring such a model returns UNSUPPORTED. Configure
     * serializes with run. Both zero selects the family default; otherwise
     * specify one cap. Family code defines ranges and valid lifecycle states. */
    trtmc_status(TRTMC_CALL* cache_stats)(trtmc_model*, trtmc_ranking_cache_stats_v1*,
                                          trtmc_error**);
    trtmc_status(TRTMC_CALL* configure_cache)(trtmc_model*, uint64_t bytes, double memory_ratio,
                                              trtmc_error**);
} trtmc_structured_ranking_api_v1;
#ifdef __cplusplus
}
#endif
#endif /* TRTMC_RANKING_H */

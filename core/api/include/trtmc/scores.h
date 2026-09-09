/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TRTMC_SCORES_H
#define TRTMC_SCORES_H

#include "trtmc/types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { TRTMC_SCORE_LOGIT = 1, TRTMC_SCORE_PROBABILITY = 2, TRTMC_SCORE_UNBOUNDED = 3 };

typedef struct {
    const float* scores;
    uint64_t count;
    /* Empty labels denote vocabulary ordinals; otherwise one label per score.
     * Specific Tasks may require semantic labels, including language IDs. */
    trtmc_strings_view labels;
    uint32_t kind;
    trtmc_string_view vocabulary_id;
} trtmc_label_scores_view_v1;

#ifdef __cplusplus
}
#endif

#endif /* TRTMC_SCORES_H */

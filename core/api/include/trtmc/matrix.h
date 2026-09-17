/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TRTMC_MATRIX_H
#define TRTMC_MATRIX_H

#include "trtmc/types.h"

/* Contiguous row-major float32. count must equal rows * columns. Axis roles
 * are defined by the enclosing Task, never guessed from this storage view. */
typedef struct {
    const float* data;
    uint64_t count;
    uint64_t rows;
    uint64_t columns;
} trtmc_f32_matrix_view_v1;

#endif /* TRTMC_MATRIX_H */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "trtmc/matrix.h"
#include "trtmc/runtime/span.h"

#include <cstdint>

namespace trtmc {
struct FloatMatrixView {
    Span<const float> values;
    std::uint64_t rows{0};
    std::uint64_t columns{0};
    trtmc_f32_matrix_view_v1 c_view() const noexcept {
        return {values.data(), values.size(), rows, columns};
    }
};
} // namespace trtmc

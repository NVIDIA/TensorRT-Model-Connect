/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace trtmc {

inline bool pixart_should_use_gpu_matmul(int32_t rows, int32_t inner, int32_t columns) {
    if (rows <= 0 || inner <= 0 || columns <= 0)
        return false;
    constexpr int64_t kGpuOperationThreshold = 1'000'000;
    return static_cast<int64_t>(rows) * inner * columns > kGpuOperationThreshold;
}

} // namespace trtmc

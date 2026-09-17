/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/runtime/span.h"

#include <cstdint>
#include <vector>

namespace trtmc::internal {

// Contiguous row-major storage. The enclosing Task defines what each axis means.
struct FloatMatrixView {
    Span<const float> values;
    std::uint64_t rows{0};
    std::uint64_t columns{0};
};
struct FloatMatrix {
    std::vector<float> values;
    std::uint64_t rows{0};
    std::uint64_t columns{0};
};

} // namespace trtmc::internal

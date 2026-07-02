/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// locateanything-owned decoded image value type for model-local preprocessing.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc {
namespace runtime {
namespace adapters {
namespace io {

struct DecodedImage {
    std::vector<uint8_t> pixels;
    int32_t width{0};
    int32_t height{0};
    int32_t channels{0};

    [[nodiscard]] bool empty() const {
        return pixels.empty() || width <= 0 || height <= 0 || channels <= 0;
    }
};

} // namespace io
} // namespace adapters
} // namespace runtime
} // namespace trtmc

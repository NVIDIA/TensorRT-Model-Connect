/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace trtmc::laguna {

// Per-layer engine I/O names: cache_k_{i}, cache_v_{i}, present_k_{i}, present_v_{i}.
inline std::string layer_tensor_name(const char* stem, std::int32_t layer) {
    return std::string(stem) + "_" + std::to_string(layer);
}

} // namespace trtmc::laguna

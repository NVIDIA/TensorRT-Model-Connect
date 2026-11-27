/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <vector>

namespace trtmc::qwen3_next {

// Preserve the official CPU router's partition and sorting behavior for ties.
std::vector<std::int32_t> select_experts(const float* probabilities, std::int32_t num_experts,
                                         std::int32_t top_k);

} // namespace trtmc::qwen3_next

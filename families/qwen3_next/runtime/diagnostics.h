/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::qwen3_next {
struct LogitTrace {
    std::vector<std::int32_t> prompt_ids;
    std::vector<std::int32_t> generated_ids;
    std::vector<std::vector<float>> rows;
    std::uint64_t routing_retries{0};
};

// Family-local diagnostic contract used by the exact-checkpoint oracle.
class Diagnostics {
  public:
    virtual ~Diagnostics() = default;
    virtual LogitTrace trace_logits(const std::string& prompt, std::int32_t steps) = 0;
};
} // namespace trtmc::qwen3_next

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace trtmc::wan_t2v {

enum class ParallelMode {
    Single,
    Tensor,
    Context,
};

struct ParallelRuntimeConfig {
    ParallelMode mode{ParallelMode::Single};
    std::int32_t size{1};

    bool distributed() const { return mode != ParallelMode::Single; }
};

ParallelRuntimeConfig parse_parallel_runtime_config(const std::string& json);
std::string denoiser_section_name(const ParallelRuntimeConfig& config, std::int32_t rank);

} // namespace trtmc::wan_t2v

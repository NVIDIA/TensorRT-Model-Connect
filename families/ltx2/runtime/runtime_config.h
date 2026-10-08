/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace trtmc::ltx2 {

enum class ParallelMode {
    Single,
    Context,
};

struct ParallelRuntimeConfig {
    ParallelMode mode{ParallelMode::Single};
    std::int32_t size{1};

    bool distributed() const { return mode != ParallelMode::Single; }
};

// Bundles built before context parallelism carry no parallel keys and run on
// one device. Context-parallel bundles share one rank-dynamic denoiser plan.
ParallelRuntimeConfig parse_parallel_runtime_config(const std::string& json);

} // namespace trtmc::ltx2

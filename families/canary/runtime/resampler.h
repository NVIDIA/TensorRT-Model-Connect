/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

namespace canary {
class RangeRunner;
} // namespace canary

// A range of a shared resampling pass holds at least this many outputs (but
// for the last, which may hold three fewer: ranges are cut between groups of
// four outputs), so a short signal stays on the calling thread.
constexpr std::int32_t kCanaryResamplerMinOutputsPerRange = 2048;

// With a runner, the outputs are filtered in ranges that the runner's threads
// take (canary_parallel.h). Every output is computed by the same statements
// however the outputs are cut and whichever thread takes a range, so the
// result does not depend on `runner` or on `min_outputs_per_range`.
std::vector<float>
resample_linear(const float* samples, std::int32_t num_samples, std::int32_t source_rate,
                std::int32_t target_rate, canary::RangeRunner* runner = nullptr,
                std::int32_t min_outputs_per_range = kCanaryResamplerMinOutputsPerRange);
std::vector<float>
resample_linear_range(const float* samples, std::int32_t num_samples, std::int32_t source_rate,
                      std::int32_t target_rate, std::int32_t output_start,
                      std::int32_t output_count, canary::RangeRunner* runner = nullptr,
                      std::int32_t min_outputs_per_range = kCanaryResamplerMinOutputsPerRange);

} // namespace trtmc

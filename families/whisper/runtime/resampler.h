/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

std::vector<float> resample_linear(const float* samples, std::int32_t num_samples,
                                   std::int32_t source_rate, std::int32_t target_rate);
std::vector<float> resample_linear_range(const float* samples, std::int32_t num_samples,
                                         std::int32_t source_rate, std::int32_t target_rate,
                                         std::int32_t output_start, std::int32_t output_count);

} // namespace trtmc

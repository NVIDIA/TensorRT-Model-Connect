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

// resample_linear_range for one pair of rates, for a caller that resamples a
// growing input range by range. The tap weights depend on the two rates only:
// the first call that needs them builds them and later calls reuse them. Every
// call returns what resample_linear_range returns for the same arguments. One
// object is not safe for concurrent calls: a call may build the table.
class LinearResampler {
  public:
    LinearResampler(std::int32_t source_rate, std::int32_t target_rate)
        : source_rate_(source_rate), target_rate_(target_rate) {}

    std::vector<float> range(const float* samples, std::int32_t num_samples,
                             std::int32_t output_start, std::int32_t output_count);

  private:
    std::int32_t source_rate_;
    std::int32_t target_rate_;
    // One row of tap weights per output phase, and each row's sum in tap
    // order. Empty until a call needs them.
    std::vector<double> phase_weights_;
    std::vector<double> phase_weight_sums_;
};

} // namespace trtmc

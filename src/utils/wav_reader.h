/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct WavData {
    std::vector<float> samples; // mono float32 [-1, 1]
    int32_t sample_rate{0};
};

// Compatibility wrapper for the canonical trtmc::io::read_wav implementation.
// Returns source-rate mono float32 samples by averaging every interleaved channel.
// This function never resamples; the receiving model pipeline owns that conversion.
WavData read_wav(const std::string& path);

// Resample audio using linear interpolation.
// Returns resampled samples at target_rate.
std::vector<float> resample_linear(const float* samples, int32_t n_samples, int32_t source_rate,
                                   int32_t target_rate);

// Resample a contiguous range of output indices without recomputing the
// preceding prefix. Values are identical to slicing resample_linear() for the
// same complete input.
std::vector<float> resample_linear_range(const float* samples, int32_t n_samples,
                                         int32_t source_rate, int32_t target_rate,
                                         int32_t output_start, int32_t output_count);

} // namespace trtmc

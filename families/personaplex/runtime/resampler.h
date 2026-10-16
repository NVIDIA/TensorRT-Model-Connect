/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

// Resample mono audio as the Native reference does (librosa soxr "HQ" band
// edges); the output has ceil(num_samples * target_rate / source_rate) samples.
std::vector<float> resample_audio(const float* samples, std::int32_t num_samples,
                                  std::int32_t source_rate, std::int32_t target_rate);

} // namespace trtmc

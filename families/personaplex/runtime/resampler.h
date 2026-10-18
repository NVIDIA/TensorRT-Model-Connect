/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

// Match the Native audio loader's libsoxr HQ resampling and librosa output length.
// Non-identity resampling requires the system libsoxr.so.0 runtime.
std::vector<float> resample_audio(const float* samples, std::int32_t num_samples,
                                  std::int32_t source_rate, std::int32_t target_rate);

} // namespace trtmc

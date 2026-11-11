/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <vector>
namespace trtmc::cohere_asr {
std::vector<float> normal_noise(std::size_t count);
std::vector<float> features(const float* samples, std::size_t count,
                            const std::vector<float>& window, const std::vector<float>& filters);
} // namespace trtmc::cohere_asr

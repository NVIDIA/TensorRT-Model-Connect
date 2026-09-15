/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "trtmc/task.h"

#include <nlohmann/json.hpp>
#include <vector>

namespace trtmc::cosyvoice3 {
struct ReferenceFeatures {
    std::vector<float> speaker, tokens, mel;
    int speaker_frames{}, token_frames{}, mel_frames{};
};
ReferenceFeatures reference_features(const AudioReference&, const nlohmann::json& coefficients);
std::vector<float> resample_reference(const std::vector<float>& samples, int source_rate,
                                      int target_rate);
} // namespace trtmc::cosyvoice3

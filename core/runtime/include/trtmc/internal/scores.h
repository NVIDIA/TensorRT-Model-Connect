/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::internal {

enum class ScoreKind : std::uint32_t { Logit = 1, Probability = 2, Unbounded = 3 };

struct LabelScoresResult {
    std::vector<float> scores;
    // Empty labels denote vocabulary ordinals. Tasks requiring semantic labels
    // (for example language identification) must return one label per score.
    std::vector<std::string> labels;
    ScoreKind kind{ScoreKind::Unbounded};
    std::string vocabulary_id;
};

} // namespace trtmc::internal

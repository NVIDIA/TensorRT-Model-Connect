/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <vector>

namespace trtmc::nemotron_omni {

struct AudioFeatures {
    std::vector<float> values; // [actual frames, 128], excluding STFT's final pad frame
    std::size_t frames{0};
};

// Input is mono, 16 kHz PCM. Filter layout is [128, 257]. No clip truncation.
AudioFeatures audio_features(const std::vector<float>& samples, const std::vector<float>& filters,
                             std::size_t max_frames);

struct AudioPositions {
    std::vector<float> values; // [2 * tokens - 1, hidden]
    std::vector<int> indices;  // [heads, tokens, tokens]
};
AudioPositions audio_positions(int tokens, int hidden, int heads);

} // namespace trtmc::nemotron_omni

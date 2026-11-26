/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/moss_transcribe_diarize/runtime/runtime_config.h"

#include <cstdint>
#include <vector>

namespace trtmc::moss_transcribe_diarize {

// One fixed-length window of the input audio.
struct AudioChunk {
    std::int64_t first_sample{0};
    std::int64_t samples{0};
    std::int32_t audio_tokens{0};
};

// Windows of `chunk_samples`; the last may be shorter. A window of n samples
// yields (n - 1) / samples_per_token + 1 audio tokens.
std::vector<AudioChunk> plan_chunks(std::int64_t total_samples, std::int32_t chunk_samples,
                                    std::int32_t samples_per_token);

// The audio span: `audio_tokens` placeholders with the decimal second markers
// written between them, exactly as the reference processor expands them.
std::vector<std::int32_t> audio_span(std::int32_t audio_tokens, const TimeMarkerConfig& markers,
                                     std::int32_t audio_token_id);

// Prompt ids plus, per position, the audio row that replaces it (-1 for text).
struct PromptPlan {
    std::vector<std::int32_t> input_ids;
    std::vector<std::int32_t> audio_index;
    std::int32_t audio_tokens{0};
};

PromptPlan build_prompt(const RuntimeConfig& config, std::int32_t audio_tokens);

} // namespace trtmc::moss_transcribe_diarize

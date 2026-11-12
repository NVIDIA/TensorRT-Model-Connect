/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/moss_transcribe_diarize/runtime/prompt_plan.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace trtmc::moss_transcribe_diarize {

std::vector<AudioChunk> plan_chunks(std::int64_t total_samples, std::int32_t chunk_samples,
                                    std::int32_t samples_per_token) {
    if (total_samples <= 0)
        throw std::invalid_argument("audio must contain at least one sample");
    if (chunk_samples <= 0 || samples_per_token <= 0)
        throw std::invalid_argument("invalid chunk geometry");
    std::vector<AudioChunk> chunks;
    for (std::int64_t start = 0; start < total_samples; start += chunk_samples) {
        AudioChunk chunk;
        chunk.first_sample = start;
        chunk.samples = std::min<std::int64_t>(chunk_samples, total_samples - start);
        chunk.audio_tokens = static_cast<std::int32_t>((chunk.samples - 1) / samples_per_token + 1);
        chunks.push_back(chunk);
    }
    return chunks;
}

std::vector<std::int32_t> audio_span(std::int32_t audio_tokens, const TimeMarkerConfig& markers,
                                     std::int32_t audio_token_id) {
    if (audio_tokens <= 0)
        return {};
    // int(audio_tokens_per_second * every_seconds), truncated as Python's int().
    const auto per_marker =
        static_cast<std::int64_t>(markers.audio_tokens_per_second * markers.every_seconds);
    if (!markers.enabled || markers.every_seconds <= 0 || per_marker <= 0)
        return std::vector<std::int32_t>(static_cast<std::size_t>(audio_tokens), audio_token_id);

    // range(every, int(duration) + 1, every) with duration = tokens / tokens_per_second.
    const auto whole_seconds = static_cast<std::int64_t>(static_cast<double>(audio_tokens) /
                                                         markers.audio_tokens_per_second);
    std::vector<std::int32_t> span;
    span.reserve(static_cast<std::size_t>(audio_tokens) + 8);
    std::int64_t consumed = 0;
    for (std::int64_t second = markers.every_seconds; second <= whole_seconds;
         second += markers.every_seconds) {
        const std::int64_t position = (second / markers.every_seconds) * per_marker;
        if (position > consumed) {
            span.insert(span.end(), static_cast<std::size_t>(position - consumed), audio_token_id);
            consumed = position;
        }
        for (const char digit : std::to_string(second))
            span.push_back(markers.digit_token_ids[static_cast<std::size_t>(digit - '0')]);
    }
    if (audio_tokens > consumed)
        span.insert(span.end(), static_cast<std::size_t>(audio_tokens - consumed), audio_token_id);
    return span;
}

PromptPlan build_prompt(const RuntimeConfig& config, std::int32_t audio_tokens) {
    const auto span = audio_span(audio_tokens, config.markers, config.audio_token_id);
    PromptPlan plan;
    plan.audio_tokens = audio_tokens;
    plan.input_ids.reserve(config.prompt_prefix_ids.size() + span.size() +
                           config.prompt_suffix_ids.size());
    plan.input_ids.insert(plan.input_ids.end(), config.prompt_prefix_ids.begin(),
                          config.prompt_prefix_ids.end());
    plan.input_ids.insert(plan.input_ids.end(), span.begin(), span.end());
    plan.input_ids.insert(plan.input_ids.end(), config.prompt_suffix_ids.begin(),
                          config.prompt_suffix_ids.end());
    // Audio rows fill the placeholders in order, as masked_scatter does. The
    // fixed prompt parts must not contain the placeholder themselves.
    plan.audio_index.assign(plan.input_ids.size(), -1);
    std::int32_t row = 0;
    for (std::size_t i = 0; i < plan.input_ids.size(); ++i) {
        if (plan.input_ids[i] == config.audio_token_id)
            plan.audio_index[i] = row++;
    }
    if (row != audio_tokens)
        throw std::runtime_error("prompt placeholders do not match the audio token count");
    return plan;
}

} // namespace trtmc::moss_transcribe_diarize

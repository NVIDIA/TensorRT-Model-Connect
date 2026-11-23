/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vibevoice_asr/runtime/host_plan.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace trtmc::vibevoice_asr {

std::vector<EncoderChunk> plan_encoder_chunks(std::int64_t total_frames, std::int64_t chunk_frames,
                                              std::int64_t context_frames) {
    if (total_frames <= 0 || chunk_frames <= 0 || context_frames < 0)
        throw std::invalid_argument("VibeVoice-ASR encoder chunk plan needs positive sizes");
    std::vector<EncoderChunk> chunks;
    for (std::int64_t first = 0; first < total_frames; first += chunk_frames)
        chunks.push_back(
            {first, std::min(chunk_frames, total_frames - first), std::min(context_frames, first)});
    return chunks;
}

std::int32_t argmax(const float* logits, std::int32_t count) {
    if (logits == nullptr || count <= 0)
        throw std::invalid_argument("VibeVoice-ASR argmax needs logits");
    std::int32_t best = 0;
    for (std::int32_t index = 1; index < count; ++index) {
        if (logits[index] > logits[best])
            best = index;
    }
    return best;
}

bool generation_complete(std::int64_t prompt_tokens, std::int64_t generated_tokens,
                         std::int64_t max_new_tokens, std::int64_t max_sequence_length) {
    return generated_tokens >= max_new_tokens ||
           prompt_tokens + generated_tokens >= max_sequence_length;
}

void check_output_budget(std::int64_t prompt_tokens, std::int64_t max_output_tokens,
                         std::int64_t max_sequence_length) {
    if (max_output_tokens > max_sequence_length - prompt_tokens)
        throw std::invalid_argument(
            "VibeVoice-ASR prompt of " + std::to_string(prompt_tokens) +
            " tokens plus max_output_tokens " + std::to_string(max_output_tokens) +
            " exceeds the bundle's max_sequence_length of " + std::to_string(max_sequence_length) +
            "; use less audio, fewer output tokens or a larger max_sequence_length");
}

} // namespace trtmc::vibevoice_asr

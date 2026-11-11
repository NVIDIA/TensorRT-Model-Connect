/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/vibevoice_asr/runtime/host_plan.h"

#include <algorithm>
#include <stdexcept>

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

} // namespace trtmc::vibevoice_asr

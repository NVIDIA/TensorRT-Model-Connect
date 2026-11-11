/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <vector>

namespace trtmc::vibevoice_asr {

// Encoder chunks over [0, total_frames): each computes `frames` output frames
// starting at `first_frame` from audio that also covers `context_frames`
// earlier frames, the causal receptive field the chunk's first frame needs.
struct EncoderChunk {
    std::int64_t first_frame{0};
    std::int64_t frames{0};
    std::int64_t context_frames{0};
};
std::vector<EncoderChunk> plan_encoder_chunks(std::int64_t total_frames, std::int64_t chunk_frames,
                                              std::int64_t context_frames);

// Lowest index of the largest value, the tie rule of torch.argmax.
std::int32_t argmax(const float* logits, std::int32_t count);

} // namespace trtmc::vibevoice_asr

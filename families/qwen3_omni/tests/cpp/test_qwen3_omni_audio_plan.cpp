/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen3_omni/runtime/audio_plan.h"

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    assert(trtmc::qwen3_omni_round_bfloat16(1.00390625F) == 1.0F);
    assert(trtmc::qwen3_omni_round_bfloat16(1.01171875F) == 1.015625F);

    std::vector<std::int32_t> frame_major(2 * 16);
    for (std::int32_t frame = 0; frame < 2; ++frame) {
        for (std::int32_t codebook = 0; codebook < 16; ++codebook)
            frame_major[static_cast<std::size_t>(frame) * 16 + codebook] = frame * 100 + codebook;
    }
    const auto input = trtmc::qwen3_omni_code2wav_input(frame_major, 16, 32);
    assert(input.size() == 16U * 32U);
    for (std::int32_t codebook = 0; codebook < 16; ++codebook) {
        assert(input[static_cast<std::size_t>(codebook) * 32] == codebook);
        assert(input[static_cast<std::size_t>(codebook) * 32 + 1] == 100 + codebook);
        assert(input[static_cast<std::size_t>(codebook) * 32 + 2] == 0);
    }
    assert(trtmc::qwen3_omni_output_samples(2, 1920, 555, 10000) == 3285);
    assert(trtmc::qwen3_omni_output_samples(0, 1920, 555, 10000) == 0);
    assert(trtmc::qwen3_omni_embedding_section_size(3072U * 1024U * sizeof(float), 1, 3072, 1024));
    assert(trtmc::qwen3_omni_embedding_section_size(15U * 2048U * 1024U * sizeof(float), 15, 2048,
                                                    1024));

    std::vector<float> positive_logits(2151, -100.0F);
    positive_logits[7] = 10.0F;
    positive_logits[8] = 9.6F;
    assert(trtmc::qwen3_omni_talker_argmax(positive_logits, 2048, 2150, {}) == 7);
    assert(trtmc::qwen3_omni_talker_argmax(positive_logits, 2048, 2150, {7}) == 8);

    std::vector<float> negative_logits(2151, -100.0F);
    negative_logits[7] = -9.6F;
    negative_logits[8] = -10.0F;
    assert(trtmc::qwen3_omni_talker_argmax(negative_logits, 2048, 2150, {}) == 7);
    assert(trtmc::qwen3_omni_talker_argmax(negative_logits, 2048, 2150, {7}) == 8);

    assert(trtmc::qwen3_omni_audio_frame_limit(32) == 31);
    assert(trtmc::qwen3_omni_audio_frame_limit(1) == 0);
    return 0;
}

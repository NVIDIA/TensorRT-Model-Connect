/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace trtmc {

/// Configuration for the Qwen3-Omni multimodal pipeline.
struct OmniConfig {
    int32_t sample_rate{24000};

    int32_t thinker_hidden_size{0};
    int32_t thinker_vocab_size{0};
    int32_t thinker_num_layers{0};
    int32_t thinker_num_heads{0};
    int32_t thinker_eos_token_id{151645};
    int32_t num_experts{8};
    int32_t num_experts_per_tok{2};

    int32_t audio_embed_dim{1280};
    int32_t audio_num_mel{128};
    int32_t audio_num_layers{0};
    int32_t audio_num_frames{1500};

    int32_t talker_hidden_size{0};
    int32_t talker_num_layers{0};
    int32_t talker_n_codebooks{16};
    int32_t talker_codebook_size{2048};

    int32_t code2wav_upsample_factor{1920};
    int32_t code2wav_output_delay{555};
    int32_t code2wav_max_frames{32};

    bool greedy{false};
    float temperature{0.7F};
    int32_t top_k{50};
};

} // namespace trtmc

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

/// Configuration for the SpeechToSpeech (PersonaPlex) pipeline.
struct SpeechConfig {
    int32_t sample_rate{24000};
    int32_t num_codebooks{8};
    int32_t codebook_size{2048};
    float frame_rate{12.5F};
    int32_t mimi_max_frames{512};

    int32_t temporal_hidden_size{0};
    int32_t temporal_num_layers{0};

    int32_t depth_hidden_size{0};
    int32_t depth_num_layers{6};
    int32_t depth_num_heads{0};
    int32_t depth_num_kv_heads{0};
    int32_t depth_max_cache_length{16};

    std::vector<float> depth_projection;
    int32_t temporal_hidden_for_proj{0};

    std::vector<float> audio_embeddings;
    int32_t audio_vocab_size{2049};

    std::vector<float> temporal_text_embedding;
    int32_t temporal_text_vocab{0};
    int32_t text_padding_id{3};

    std::vector<float> depth_text_embedding;
    int32_t depth_text_vocab{0};

    int32_t mimi_decode_codebooks{8};

    std::vector<float> depth_audio_embeddings;
    int32_t num_depformer_emb{0};

    std::vector<int32_t> delays;
    int32_t text_initial_token_id{32000};
    int32_t audio_initial_token_id{2048};

    float depth_temperature{0.0F};
    int32_t depth_top_k{0};

    int32_t text_eos_token_id{-1};

    std::string system_prompt;
    std::vector<int32_t> text_prompt_ids;

    std::string hf_python;
};

} // namespace trtmc

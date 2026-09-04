/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>

namespace trtmc {

struct BarkConfig {
    int32_t sample_rate{24000};
    int32_t hidden_size{768};

    // Semantic model constants
    int32_t semantic_input_vocab{129600};
    int32_t semantic_output_vocab{10048};
    int32_t text_encoding_offset{10048};
    int32_t semantic_pad_token{10000};
    int32_t semantic_infer_token{129599};
    int32_t semantic_vocab_size{10000}; // valid semantic token range [0, 10000)
    int32_t text_pad_token{129595};     // HF text_pad_token for masked positions

    // Coarse model constants
    int32_t coarse_input_vocab{12096};
    int32_t coarse_semantic_pad_token{12048};
    int32_t coarse_infer_token{12050};
    int32_t n_coarse_codebooks{2};
    int32_t codebook_size{1024};
    int32_t coarse_rate_hz{75};
    float semantic_rate_hz{49.9F};
    int32_t max_coarse_history{630};
    int32_t max_coarse_input_length{256};
    int32_t sliding_window_len{60};

    // Codec (EnCodec) engine config
    int32_t codec_seq_length{0};        // max frames the codec engine was compiled for
    int32_t codec_upsample_factor{320}; // total upsample ratio (8*5*4*2)
    int32_t codec_n_codebooks{8};       // number of codebooks in codec engine input

    // Fine model config
    int32_t fine_hidden_size{768};
    int32_t fine_n_lm_heads{7};
    int32_t fine_codebook_size{1056}; // vocab per codebook (fine model)
    int32_t fine_seq_length{0};       // 0 = no fine engine

    // Sampling parameters
    float semantic_temperature{0.7F};
    float coarse_temperature{0.7F};
    float fine_temperature{0.5F};
    int32_t top_k{50};
    float min_eos_p{0.0F}; // 0 = disabled (matching HF bark-small default)
    bool greedy{false};    // if true, use argmax instead of sampling

    // audio.bark.* namespace replaces TRTMC_BARK_{DUMP,GREEDY,SEED}.
    // bark_plugin populates these from ctx.runtime_config.
    std::string dump_path{}; // empty -> no token dump
    int64_t seed{-1};        // -1 -> use default-constructed RNG state
};

} // namespace trtmc

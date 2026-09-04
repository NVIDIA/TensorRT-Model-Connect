/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct CanaryConfig {
    int32_t num_mel_bins{80};
    int32_t max_source_positions{1500};
    int32_t max_target_positions{448};
    int32_t encoder_layers{0};
    int32_t decoder_layers{0};
    int32_t decoder_attention_heads{0};
    int32_t eot_token_id{50257}; // <|endoftext|>

    int32_t mel_length{0}; // expected mel input length; 0 = auto (max_source_positions * 2)

    // Exact decoder prompt derived from the checkpoint tokenizer at build time.
    std::vector<int32_t> decoder_start_token_ids;

    // Canary prompt metadata is derived from the checkpoint tokenizer while
    // building the bundle. The positions refer to decoder_start_token_ids.
    std::vector<std::string> supported_languages;
    std::vector<int32_t> language_token_ids;
    int32_t source_language_position{4};
    int32_t target_language_position{5};
    int32_t punctuation_position{6};
    int32_t timestamp_position{8};
    int32_t punctuation_token_id{-1};
    int32_t no_punctuation_token_id{-1};
    int32_t timestamp_token_id{-1};
    int32_t no_timestamp_token_id{-1};
    bool translation_requires_english{true};
};

} // namespace trtmc

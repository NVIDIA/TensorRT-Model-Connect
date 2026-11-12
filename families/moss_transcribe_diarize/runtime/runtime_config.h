/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::moss_transcribe_diarize {

// Numeric second markers the reference processor writes into the audio span.
struct TimeMarkerConfig {
    bool enabled{true};
    double audio_tokens_per_second{0.0};
    std::int32_t every_seconds{0};
    std::array<std::int32_t, 10> digit_token_ids{};
};

// The bundle's runtime.json, validated once at load.
struct RuntimeConfig {
    std::string precision;
    std::int32_t sample_rate{0};
    std::int32_t chunk_samples{0};
    std::int32_t n_fft{0};
    std::int32_t hop_length{0};
    std::int32_t num_mel_bins{0};
    std::int32_t mel_frames{0};
    std::int32_t tokens_per_chunk{0};
    std::int32_t samples_per_token{0};
    std::int32_t hidden_size{0};
    std::int32_t num_layers{0};
    std::int32_t num_kv_heads{0};
    std::int32_t head_dim{0};
    std::int32_t vocab_size{0};
    std::int32_t max_sequence_length{0};
    std::int32_t prefill_chunk_tokens{0};
    std::int32_t audio_token_id{0};
    TimeMarkerConfig markers;
    std::vector<std::int32_t> prompt_prefix_ids;
    std::vector<std::int32_t> prompt_suffix_ids;
    std::vector<std::int32_t> eos_token_ids;
    std::int32_t default_max_new_tokens{0};
};

RuntimeConfig parse_runtime_config(const std::vector<char>& json_bytes);

} // namespace trtmc::moss_transcribe_diarize

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtmc {

struct RnntConfig {
    int32_t sample_rate{16000};
    int32_t num_mel_bins{128};
    int32_t mel_n_fft{512};
    int32_t mel_win_length{400};
    int32_t mel_hop_length{160};
    int32_t mel_chunk_length{30};
    float mel_preemph{0.97F};
    int32_t mel_length{3000};
    int32_t encoder_hidden_size{0};
    int32_t pred_hidden_size{0};
    int32_t pred_num_layers{1};
    int32_t vocab_size{0}; // excludes RNNT blank
    int32_t blank_id{0};
    int32_t max_symbols_per_step{10};
    int32_t encoder_seq_len{0};
    int32_t encoder_layers{0};
    int32_t att_context_left{70};
    int32_t att_context_right{13};
    int32_t subsampling_factor{8};
    int32_t streaming_cache_left{70};
    int32_t streaming_time_cache{8};
    int32_t streaming_pre_encode_cache{9};
    int32_t streaming_drop_pre_encoded{2};
    bool causal_downsampling{false};

    // Multilingual / prompt-kernel variant fields.
    bool has_prompt_kernel{false};
    int32_t num_prompts{0};
    std::unordered_map<std::string, int32_t> prompt_dictionary;

    // Per-checkpoint list of supported att_context_right values (e.g.,
    // {13,6,1,0} for en-0.6b, {13,6,3,0} for the 3.5 multilingual).
    std::vector<int32_t> supported_right_contexts;
};

struct RnntStreamingSchedule {
    int32_t att_context_left{70};
    int32_t att_context_right{13};
    int32_t subsampling_factor{8};
    int32_t encoder_frame_ms{80};
    int32_t chunk_ms{1120};
    int32_t chunk_samples{17920};
    int32_t first_chunk_mel_frames{105};
    int32_t next_chunk_mel_frames{112};
    int32_t first_shift_mel_frames{105};
    int32_t next_shift_mel_frames{112};
    int32_t first_pre_encode_cache_mel_frames{0};
    int32_t next_pre_encode_cache_mel_frames{9};
    int32_t valid_encoder_frames{14};
    int32_t drop_extra_pre_encoded{2};
};

inline bool is_supported_nemotron_att_context(int32_t left, int32_t right, int32_t supported_left,
                                              const std::vector<int32_t>& supported_right) {
    if (left != supported_left)
        return false;
    return std::find(supported_right.begin(), supported_right.end(), right) !=
           supported_right.end();
}

inline RnntStreamingSchedule make_nemotron_streaming_schedule(int32_t att_context_left,
                                                              int32_t att_context_right,
                                                              int32_t sample_rate = 16000,
                                                              int32_t mel_hop_length = 160,
                                                              int32_t subsampling_factor = 8) {
    if (att_context_left <= 0)
        throw std::invalid_argument("Nemotron RNNT streaming requires positive att_context_left");
    if (att_context_right < 0)
        throw std::invalid_argument(
            "Nemotron RNNT streaming requires non-negative att_context_right");
    if (sample_rate <= 0 || mel_hop_length <= 0 || subsampling_factor <= 0)
        throw std::invalid_argument("RNN-T streaming schedule requires positive "
                                    "sample_rate, mel_hop_length, and subsampling_factor");

    RnntStreamingSchedule s;
    s.att_context_left = att_context_left;
    s.att_context_right = att_context_right;
    s.subsampling_factor = subsampling_factor;
    s.encoder_frame_ms = 80;
    s.valid_encoder_frames = att_context_right + 1;
    s.chunk_ms = s.valid_encoder_frames * s.encoder_frame_ms;
    s.chunk_samples = sample_rate * s.chunk_ms / 1000;

    // Matches NeMo CacheAwareStreamingAudioBuffer for this checkpoint:
    // sampling_frames=[1,8], pre_encode_cache_size=[0,9].
    s.first_chunk_mel_frames = 1 + subsampling_factor * att_context_right;
    s.next_chunk_mel_frames = subsampling_factor * s.valid_encoder_frames;
    s.first_shift_mel_frames = s.first_chunk_mel_frames;
    s.next_shift_mel_frames = s.next_chunk_mel_frames;
    s.first_pre_encode_cache_mel_frames = 0;
    s.next_pre_encode_cache_mel_frames = 9;
    s.drop_extra_pre_encoded = 2;
    return s;
}

struct RnntGreedyDecision {
    bool emit_token{false};
    bool advance_frame{false};
};

inline RnntGreedyDecision make_rnnt_greedy_decision(int32_t token_id, int32_t blank_id,
                                                    int32_t symbols_this_frame,
                                                    int32_t max_symbols_per_step) {
    if (token_id == blank_id)
        return {false, true};
    if (max_symbols_per_step > 0 && symbols_this_frame + 1 >= max_symbols_per_step)
        return {true, true};
    return {true, false};
}

} // namespace trtmc

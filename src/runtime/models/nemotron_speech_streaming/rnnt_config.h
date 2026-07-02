/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <stdexcept>

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

inline bool is_supported_nemotron_att_context(int32_t left, int32_t right) {
    return left == 70 && (right == 0 || right == 1 || right == 6 || right == 13);
}

inline RnntStreamingSchedule make_nemotron_streaming_schedule(int32_t att_context_left,
                                                              int32_t att_context_right,
                                                              int32_t sample_rate = 16000,
                                                              int32_t mel_hop_length = 160,
                                                              int32_t subsampling_factor = 8) {
    if (!is_supported_nemotron_att_context(att_context_left, att_context_right))
        throw std::invalid_argument("RNN-T streaming supports att_context_size "
                                    "[70,0], [70,1], [70,6], or [70,13]");
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

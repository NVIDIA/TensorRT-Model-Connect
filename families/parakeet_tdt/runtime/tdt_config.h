/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtmc {

struct TdtConfig {
    int32_t sample_rate{16000};
    int32_t num_mel_bins{128};
    int32_t mel_n_fft{512};
    int32_t mel_win_length{400};
    int32_t mel_hop_length{160};
    int32_t mel_chunk_length{30};
    float mel_preemph{0.97F};
    int32_t mel_length{3000};
    int32_t encoder_hidden_size{1024};
    int32_t pred_hidden_size{640};
    int32_t pred_num_layers{2};
    int32_t vocab_size{8192};
    int32_t blank_id{8192};
    int32_t max_symbols_per_step{10};
    int32_t encoder_seq_len{375};
    int32_t encoder_layers{24};
    int32_t subsampling_factor{8};
    std::vector<int32_t> duration_values{0, 1, 2, 3, 4};

    // Retained as inert fields so the family-private offline pipeline shares
    // one constructor shape with the repository's speech pipeline machinery.
    int32_t att_context_left{-1};
    int32_t att_context_right{-1};
    int32_t streaming_cache_left{0};
    int32_t streaming_time_cache{0};
    int32_t streaming_pre_encode_cache{0};
    int32_t streaming_drop_pre_encoded{0};
    bool causal_downsampling{false};
    bool has_prompt_kernel{false};
    int32_t num_prompts{0};
    std::unordered_map<std::string, int32_t> prompt_dictionary;
    std::vector<int32_t> supported_right_contexts;
};

struct TdtGreedyDecision {
    bool emit_token{false};
    int32_t frame_advance{0};
};

// Streaming is deliberately not advertised for v3, but these private fields
// keep the inherited speech-session implementation buildable and fail closed
// unless a bundle contains matching streaming encoder plans.
struct TdtStreamingSchedule {
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

inline TdtStreamingSchedule make_tdt_streaming_schedule(int32_t left, int32_t right,
                                                        int32_t sample_rate = 16000,
                                                        int32_t hop = 160,
                                                        int32_t subsampling = 8) {
    if (left <= 0 || right < 0 || sample_rate <= 0 || hop <= 0 || subsampling <= 0)
        throw std::invalid_argument("invalid private TDT streaming schedule");
    TdtStreamingSchedule out;
    out.att_context_left = left;
    out.att_context_right = right;
    out.subsampling_factor = subsampling;
    out.encoder_frame_ms = hop * subsampling * 1000 / sample_rate;
    out.valid_encoder_frames = right + 1;
    out.chunk_ms = out.valid_encoder_frames * out.encoder_frame_ms;
    out.chunk_samples = sample_rate * out.chunk_ms / 1000;
    out.first_chunk_mel_frames = 1 + subsampling * right;
    out.next_chunk_mel_frames = subsampling * out.valid_encoder_frames;
    out.first_shift_mel_frames = out.first_chunk_mel_frames;
    out.next_shift_mel_frames = out.next_chunk_mel_frames;
    out.next_pre_encode_cache_mel_frames = subsampling + 1;
    out.drop_extra_pre_encoded = out.next_pre_encode_cache_mel_frames - (subsampling - 1);
    return out;
}

inline void validate_tdt_mel_geometry(const TdtConfig& config, int32_t filterbank_freq_bins,
                                      int32_t filterbank_mel_bins) {
    if (config.sample_rate <= 0 || config.mel_hop_length <= 0 || config.mel_n_fft <= 0 ||
        config.mel_chunk_length <= 0 || config.mel_length <= 0 || config.num_mel_bins <= 0) {
        throw std::invalid_argument("invalid TDT mel dimensions in config");
    }
    if (filterbank_freq_bins != config.mel_n_fft / 2 + 1 ||
        filterbank_mel_bins != config.num_mel_bins) {
        throw std::invalid_argument("TDT mel filterbank dimensions do not match config");
    }
}

inline TdtGreedyDecision make_tdt_greedy_decision(int32_t token_id, int32_t duration_index,
                                                  const std::vector<int32_t>& duration_values,
                                                  int32_t blank_id) {
    if (duration_index < 0 || duration_index >= static_cast<int32_t>(duration_values.size()))
        throw std::out_of_range("TDT duration index is outside the configured duration table");
    int32_t advance = duration_values[static_cast<std::size_t>(duration_index)];
    if (advance < 0)
        throw std::invalid_argument("TDT durations must be non-negative");
    const bool emit = token_id != blank_id;
    if (advance == 0 && !emit)
        advance = 1;
    return {emit, advance};
}

} // namespace trtmc

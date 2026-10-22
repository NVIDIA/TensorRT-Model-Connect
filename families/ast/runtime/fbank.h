/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc::ast {

// The Kaldi filter-bank frontend AST's reference feature extractor calls, with
// every constant resolved at bundle time. The analysis window and the mel
// filters are built by the family's Python side and carried in the bundle, so
// the mel scale is spelled out once rather than twice.
struct FbankConfig {
    std::int32_t sample_rate{16000};
    std::int32_t frame_length{400};
    std::int32_t frame_shift{160};
    std::int32_t n_fft{512};
    std::int32_t num_mel_bins{128};
    std::int32_t max_length{1024};
    float mel_floor{1.1920928955078125e-07F};
    float preemphasis{0.97F};
    float feature_mean{0.0F};
    float feature_scale{1.0F};
    std::vector<float> window;      // [frame_length]
    std::vector<float> mel_filters; // [num_mel_bins, n_fft / 2 + 1], row-major
};

// Frames the waveform the way Kaldi's snip_edges does: whole frames only, with
// no centring and no padding at either end.
std::int64_t compute_ast_frame_count(std::int64_t num_samples, std::int32_t frame_length,
                                     std::int32_t frame_shift);

// One frame, in Kaldi's order: remove the DC offset, pre-emphasise against the
// frame's own replicated first sample, then window. Writes frame_length values.
void compute_ast_frame(const float* samples, const FbankConfig& config, float* frame);

// Power spectrum of a real signal zero-padded to config.n_fft, which Kaldi has
// already rounded up to a power of two. Returns n_fft / 2 + 1 values.
std::vector<float> compute_ast_power_spectrum(const float* frame, const FbankConfig& config);

// The normalised spectrogram the engine takes, [max_length, num_mel_bins]
// row-major. Real frames are padded with the raw filter-bank's zero and the
// whole thing is normalised afterwards, which is why the padding does not come
// out as zero.
std::vector<float> compute_ast_features(const float* samples, std::int64_t num_samples,
                                        const FbankConfig& config);

// Mixes interleaved channels down to mono, which is what the reference feeds
// the feature extractor. A single channel is copied through unchanged.
std::vector<float> mix_ast_mono(const float* samples, std::int64_t count, std::int32_t channels);

} // namespace trtmc::ast

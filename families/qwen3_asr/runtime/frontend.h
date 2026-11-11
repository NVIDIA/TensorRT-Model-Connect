/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Qwen3-ASR audio front end: the Qwen3ASRFeatureExtractor log-mel spectrogram and the encoder's
// chunk / window layout.
//
// Features: clips shorter than min_samples are zero-padded (the padding counts as audio); the
// waveform is reflect-padded by n_fft / 2 on both sides; a periodic Hann window of n_fft samples
// slides by hop_length; the power spectrum goes through the Slaney filterbank; values become
// log10(max(x, 1e-10)), are floored at the clip's maximum minus 8 and mapped to (x + 4) / 4. The
// last STFT frame is dropped, so a clip of n samples has n / hop_length frames.
//
// Layout: the mel axis is cut into chunks of chunk_frames frames, zero-padded; chunks are grouped
// into attention windows of window_chunks chunks. Only the last chunk can be partial, so the valid
// encoder tokens are the first audio_tokens rows of the chunk-major token grid.

#include <cstdint>
#include <vector>

namespace trtmc::qwen3_asr {

struct MelFilterbank {
    std::vector<float> weights; // [n_freq, n_mel], row-major
    std::int32_t n_freq{0};
    std::int32_t n_mel{0};
};

struct FrontendConfig {
    std::int32_t sample_rate{0};
    std::int32_t n_fft{0};
    std::int32_t hop_length{0};
    std::int64_t min_samples{0};
    std::int32_t chunk_frames{0};
    std::int32_t tokens_per_chunk{0};
    std::int32_t window_chunks{0};
};

struct LogMel {
    std::vector<float> values; // [n_mel, frames], row-major
    std::int32_t n_mel{0};
    std::int32_t frames{0};
};

struct EncoderBatch {
    std::vector<float> features;            // [windows, window_chunks, n_mel, chunk_frames]
    std::vector<std::int32_t> chunk_tokens; // [windows, window_chunks]
    std::int32_t windows{0};
    std::int32_t audio_tokens{0};
};

// Tokens three (kernel 3, stride 2, padding 1) convolutions keep from `frames` frames.
std::int32_t post_cnn_length(std::int32_t frames);

// Audio tokens of a mono clip of `samples` samples, computed from its length alone; equals
// encoder_batch(log_mel(...)).audio_tokens.
std::int64_t audio_token_count(std::int64_t samples, const FrontendConfig& config);

LogMel log_mel(const std::vector<float>& mono, const FrontendConfig& config,
               const MelFilterbank& filters);

EncoderBatch encoder_batch(const LogMel& mel, const FrontendConfig& config);

namespace detail {
// |DFT|^2 of one real frame, bins 0 .. n / 2; exposed for the front-end unit test.
std::vector<double> power_spectrum(const std::vector<float>& frame);
} // namespace detail

} // namespace trtmc::qwen3_asr

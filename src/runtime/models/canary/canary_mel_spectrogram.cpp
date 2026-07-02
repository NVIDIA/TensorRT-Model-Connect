/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/canary/canary_mel_spectrogram.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace trtmc {
namespace canary {

namespace {

std::vector<float> make_hann_window(int32_t length) {
    std::vector<float> window(length);
    const double pi2 = 2.0 * 3.14159265358979323846;
    for (int32_t i = 0; i < length; ++i) {
        window[i] = static_cast<float>(
            0.5 * (1.0 - std::cos(pi2 * static_cast<double>(i) / static_cast<double>(length))));
    }
    return window;
}

void rfft_power_direct(const float* x, int32_t n, int32_t n_out, float* power_out) {
    const double pi2 = 2.0 * 3.14159265358979323846;
    for (int32_t k = 0; k < n_out; ++k) {
        double re = 0.0;
        double im = 0.0;
        const double w = pi2 * static_cast<double>(k) / static_cast<double>(n);
        for (int32_t t = 0; t < n; ++t) {
            const double angle = w * static_cast<double>(t);
            re += static_cast<double>(x[t]) * std::cos(angle);
            im -= static_cast<double>(x[t]) * std::sin(angle);
        }
        power_out[k] = static_cast<float>(re * re + im * im);
    }
}

std::vector<float> build_center_padded_audio(const float* samples, int32_t n_samples,
                                             int32_t chunk_length_s, int32_t sample_rate,
                                             int32_t n_fft) {
    const int32_t audio_length = chunk_length_s * sample_rate;
    std::vector<float> audio_padded(audio_length, 0.0F);
    const int32_t copy_len = std::min(n_samples, audio_length);
    if (copy_len > 0) {
        std::memcpy(audio_padded.data(), samples, copy_len * sizeof(float));
    }

    const int32_t pad_size = n_fft / 2;
    const int32_t padded_length = pad_size + audio_length + pad_size;
    std::vector<float> padded(padded_length, 0.0F);
    std::memcpy(padded.data() + pad_size, audio_padded.data(), audio_length * sizeof(float));
    return padded;
}

std::vector<float> compute_power_spectrogram(const std::vector<float>& padded,
                                             const std::vector<float>& window, int32_t n_fft,
                                             int32_t hop_length, int32_t n_freq_bins,
                                             int32_t& n_frames_raw) {
    n_frames_raw = 1 + (static_cast<int32_t>(padded.size()) - n_fft) / hop_length;
    std::vector<float> power(static_cast<std::size_t>(n_freq_bins) * n_frames_raw, 0.0F);
    std::vector<float> windowed(n_fft);
    std::vector<float> frame_power(n_freq_bins);

    for (int32_t t = 0; t < n_frames_raw; ++t) {
        const int32_t start = t * hop_length;
        for (int32_t i = 0; i < n_fft; ++i) {
            windowed[static_cast<std::size_t>(i)] =
                padded[static_cast<std::size_t>(start + i)] * window[static_cast<std::size_t>(i)];
        }

        rfft_power_direct(windowed.data(), n_fft, n_freq_bins, frame_power.data());

        for (int32_t f = 0; f < n_freq_bins; ++f) {
            power[static_cast<std::size_t>(f) * n_frames_raw + t] = frame_power[f];
        }
    }
    return power;
}

std::vector<float> project_power_to_mel(const std::vector<float>& power, const float* mel_filters,
                                        int32_t n_freq_bins, int32_t n_mel_bins,
                                        int32_t n_frames_raw) {
    std::vector<float> mel_spec(static_cast<std::size_t>(n_mel_bins) * n_frames_raw, 0.0F);

    for (int32_t t = 0; t < n_frames_raw; ++t) {
        for (int32_t f = 0; f < n_freq_bins; ++f) {
            const float p = power[static_cast<std::size_t>(f) * n_frames_raw + t];
            if (p == 0.0F) {
                continue;
            }
            for (int32_t m = 0; m < n_mel_bins; ++m) {
                mel_spec[static_cast<std::size_t>(m) * n_frames_raw + t] +=
                    p * mel_filters[static_cast<std::size_t>(f) * n_mel_bins + m];
            }
        }
    }
    return mel_spec;
}

void normalize_log_mel_inplace(std::vector<float>& mel_spec) {
    float global_max = -1e10F;
    for (float& value : mel_spec) {
        value = std::log10(std::max(value, 1e-10F));
        if (value > global_max) {
            global_max = value;
        }
    }

    const float floor = global_max - 8.0F;
    for (float& value : mel_spec) {
        value = std::max(value, floor);
        value = (value + 4.0F) / 4.0F;
    }
}

std::vector<float> trim_last_frame(std::vector<float> mel_spec, int32_t n_mel_bins,
                                   int32_t n_frames_raw, int32_t& n_frames_out) {
    n_frames_out = n_frames_raw;
    if (n_frames_raw <= 1) {
        return mel_spec;
    }

    n_frames_out = n_frames_raw - 1;
    std::vector<float> trimmed(static_cast<std::size_t>(n_mel_bins) * n_frames_out);
    for (int32_t m = 0; m < n_mel_bins; ++m) {
        std::memcpy(trimmed.data() + static_cast<std::size_t>(m) * n_frames_out,
                    mel_spec.data() + static_cast<std::size_t>(m) * n_frames_raw,
                    n_frames_out * sizeof(float));
    }
    return trimmed;
}

} // namespace

MelResult extract_mel_spectrogram(const float* samples, int32_t n_samples, const float* mel_filters,
                                  int32_t n_freq_bins, int32_t n_mel_bins, int32_t n_fft,
                                  int32_t hop_length, int32_t chunk_length_s, int32_t sample_rate) {
    const int32_t expected_freq_bins = n_fft / 2 + 1;
    const int32_t freq_bins = n_freq_bins == expected_freq_bins ? n_freq_bins : expected_freq_bins;
    const std::vector<float> padded =
        build_center_padded_audio(samples, n_samples, chunk_length_s, sample_rate, n_fft);

    int32_t n_frames_raw = 0;
    const std::vector<float> power = compute_power_spectrogram(
        padded, make_hann_window(n_fft), n_fft, hop_length, freq_bins, n_frames_raw);
    std::vector<float> mel_spec =
        project_power_to_mel(power, mel_filters, freq_bins, n_mel_bins, n_frames_raw);
    normalize_log_mel_inplace(mel_spec);

    int32_t n_frames_out = 0;
    mel_spec = trim_last_frame(std::move(mel_spec), n_mel_bins, n_frames_raw, n_frames_out);

    // Frames covering the real audio (before chunk padding). The audio is
    // padded to chunk_length_s before framing, so n_frames_out is always the
    // full chunk length; valid_frames lets the encoder mask out the padded
    // tail. Mirrors the chunk framing math (out = audio_samples / hop_length).
    const int32_t chunk_samples = chunk_length_s * sample_rate;
    const int32_t valid_audio = std::min(std::max(n_samples, 0), chunk_samples);
    const int32_t valid_frames = std::min(n_frames_out, valid_audio / hop_length);

    MelResult result;
    result.data = std::move(mel_spec);
    result.n_mels = n_mel_bins;
    result.n_frames = n_frames_out;
    result.valid_frames = valid_frames;
    return result;
}

} // namespace canary
} // namespace trtmc

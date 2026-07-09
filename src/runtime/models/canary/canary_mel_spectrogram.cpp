/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/canary/canary_mel_spectrogram.h"

#include <algorithm>
#include <cmath>
#include <complex>
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

bool is_power_of_two(int32_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

void rfft_power_radix2(const float* x, int32_t n, int32_t n_out, float* power_out,
                       std::vector<std::complex<double>>& workspace) {
    workspace.resize(static_cast<std::size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        workspace[static_cast<std::size_t>(i)] =
            std::complex<double>(static_cast<double>(x[i]), 0.0);
    }

    for (int32_t i = 1, j = 0; i < n; ++i) {
        int32_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(workspace[static_cast<std::size_t>(i)],
                      workspace[static_cast<std::size_t>(j)]);
        }
    }

    const double pi2 = 2.0 * 3.14159265358979323846;
    for (int32_t length = 2; length <= n; length <<= 1) {
        const double angle = -pi2 / static_cast<double>(length);
        const std::complex<double> step(std::cos(angle), std::sin(angle));
        const int32_t half = length / 2;
        for (int32_t offset = 0; offset < n; offset += length) {
            std::complex<double> twiddle(1.0, 0.0);
            for (int32_t i = 0; i < half; ++i) {
                const auto even = workspace[static_cast<std::size_t>(offset + i)];
                const auto odd = workspace[static_cast<std::size_t>(offset + i + half)] * twiddle;
                workspace[static_cast<std::size_t>(offset + i)] = even + odd;
                workspace[static_cast<std::size_t>(offset + i + half)] = even - odd;
                twiddle *= step;
            }
        }
    }

    const int32_t bins = std::min(n_out, n / 2 + 1);
    for (int32_t k = 0; k < bins; ++k) {
        power_out[k] = static_cast<float>(std::norm(workspace[static_cast<std::size_t>(k)]));
    }
    std::fill(power_out + bins, power_out + n_out, 0.0F);
}

void rfft_power_impl(const float* x, int32_t n, int32_t n_out, float* power_out,
                     std::vector<std::complex<double>>& workspace) {
    if (is_power_of_two(n)) {
        rfft_power_radix2(x, n, n_out, power_out, workspace);
        return;
    }
    rfft_power_direct(x, n, n_out, power_out);
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

std::vector<float> compute_mel_spectrogram(const std::vector<float>& padded,
                                           const std::vector<float>& window,
                                           const float* mel_filters, int32_t n_fft,
                                           int32_t hop_length, int32_t n_freq_bins,
                                           int32_t n_mel_bins, int32_t frames_to_compute,
                                           int32_t& n_frames_raw) {
    n_frames_raw = 1 + (static_cast<int32_t>(padded.size()) - n_fft) / hop_length;
    std::vector<float> mel_spec(static_cast<std::size_t>(n_mel_bins) * n_frames_raw, 0.0F);
    std::vector<float> windowed(n_fft);
    std::vector<float> frame_power(n_freq_bins);
    std::vector<float> frame_mel(n_mel_bins);
    std::vector<std::complex<double>> fft_workspace;

    const int32_t computed_frames = std::clamp(frames_to_compute, 0, n_frames_raw);
    for (int32_t t = 0; t < computed_frames; ++t) {
        const int32_t start = t * hop_length;
        for (int32_t i = 0; i < n_fft; ++i) {
            windowed[static_cast<std::size_t>(i)] =
                padded[static_cast<std::size_t>(start + i)] * window[static_cast<std::size_t>(i)];
        }

        rfft_power_impl(windowed.data(), n_fft, n_freq_bins, frame_power.data(), fft_workspace);

        std::fill(frame_mel.begin(), frame_mel.end(), 0.0F);
        for (int32_t f = 0; f < n_freq_bins; ++f) {
            const float p = frame_power[static_cast<std::size_t>(f)];
            if (p == 0.0F) {
                continue;
            }
            const float* filter_row = mel_filters + static_cast<std::size_t>(f) * n_mel_bins;
            for (int32_t m = 0; m < n_mel_bins; ++m) {
                frame_mel[static_cast<std::size_t>(m)] +=
                    p * filter_row[static_cast<std::size_t>(m)];
            }
        }
        for (int32_t m = 0; m < n_mel_bins; ++m) {
            mel_spec[static_cast<std::size_t>(m) * n_frames_raw + t] =
                frame_mel[static_cast<std::size_t>(m)];
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

namespace detail {

std::vector<float> rfft_power(const std::vector<float>& input) {
    const int32_t n = static_cast<int32_t>(input.size());
    const int32_t n_out = n / 2 + 1;
    std::vector<float> power(static_cast<std::size_t>(n_out), 0.0F);
    std::vector<std::complex<double>> workspace;
    if (n > 0) {
        rfft_power_impl(input.data(), n, n_out, power.data(), workspace);
    }
    return power;
}

} // namespace detail

MelResult extract_mel_spectrogram(const float* samples, int32_t n_samples, const float* mel_filters,
                                  int32_t n_freq_bins, int32_t n_mel_bins, int32_t n_fft,
                                  int32_t hop_length, int32_t chunk_length_s, int32_t sample_rate) {
    const int32_t expected_freq_bins = n_fft / 2 + 1;
    const int32_t freq_bins = n_freq_bins == expected_freq_bins ? n_freq_bins : expected_freq_bins;
    const int32_t chunk_samples = chunk_length_s * sample_rate;
    const int32_t valid_audio = std::min(std::max(n_samples, 0), chunk_samples);
    const int32_t pad_size = n_fft / 2;

    // The encoder still receives the full chunk-shaped mel tensor, but only
    // frames whose windows can overlap real audio need an FFT and mel
    // projection. The untouched tail remains zero-power and is normalized to
    // the same floor value as the previous full padded computation.
    const int32_t frames_to_compute =
        valid_audio > 0 ? 1 + (pad_size + valid_audio - 1) / hop_length : 0;
    const std::vector<float> padded =
        build_center_padded_audio(samples, n_samples, chunk_length_s, sample_rate, n_fft);

    int32_t n_frames_raw = 0;
    std::vector<float> mel_spec =
        compute_mel_spectrogram(padded, make_hann_window(n_fft), mel_filters, n_fft, hop_length,
                                freq_bins, n_mel_bins, frames_to_compute, n_frames_raw);
    normalize_log_mel_inplace(mel_spec);

    int32_t n_frames_out = 0;
    mel_spec = trim_last_frame(std::move(mel_spec), n_mel_bins, n_frames_raw, n_frames_out);

    // Frames covering the real audio (before chunk padding). The audio is
    // padded to chunk_length_s before framing, so n_frames_out is always the
    // full chunk length; valid_frames lets the encoder mask out the padded
    // tail. Mirrors the chunk framing math (out = audio_samples / hop_length).
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

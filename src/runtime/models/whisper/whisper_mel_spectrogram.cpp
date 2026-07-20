/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/whisper/whisper_mel_spectrogram.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <vector>

namespace trtmc {
namespace whisper {

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

bool is_power_of_two(int32_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

void fft_radix2_inplace(std::vector<std::complex<double>>& values, bool inverse) {
    const std::size_t size = values.size();
    for (std::size_t i = 1, j = 0; i < size; ++i) {
        std::size_t bit = size >> 1;
        for (; (j & bit) != 0; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(values[i], values[j]);
        }
    }

    constexpr double kPi2 = 2.0 * 3.14159265358979323846;
    for (std::size_t length = 2; length <= size; length <<= 1) {
        const double angle = (inverse ? kPi2 : -kPi2) / static_cast<double>(length);
        const std::complex<double> step(std::cos(angle), std::sin(angle));
        const std::size_t half = length / 2;
        for (std::size_t offset = 0; offset < size; offset += length) {
            std::complex<double> twiddle(1.0, 0.0);
            for (std::size_t i = 0; i < half; ++i) {
                const auto even = values[offset + i];
                const auto odd = values[offset + i + half] * twiddle;
                values[offset + i] = even + odd;
                values[offset + i + half] = even - odd;
                twiddle *= step;
            }
        }
    }

    if (inverse) {
        const double scale = 1.0 / static_cast<double>(size);
        for (auto& value : values) {
            value *= scale;
        }
    }
}

class RfftPowerPlan {
  public:
    explicit RfftPowerPlan(int32_t n) : n_(n) {
        if (n_ <= 0 || is_power_of_two(n_)) {
            return;
        }

        fft_size_ = 1;
        while (fft_size_ < static_cast<std::size_t>(2 * n_ - 1)) {
            fft_size_ <<= 1;
        }

        chirp_.resize(static_cast<std::size_t>(n_));
        kernel_fft_.assign(fft_size_, {0.0, 0.0});
        constexpr double kPi = 3.14159265358979323846;
        for (int32_t i = 0; i < n_; ++i) {
            const double index = static_cast<double>(i);
            const double angle = kPi * index * index / static_cast<double>(n_);
            chirp_[static_cast<std::size_t>(i)] = {std::cos(angle), -std::sin(angle)};
            const std::complex<double> kernel(std::cos(angle), std::sin(angle));
            kernel_fft_[static_cast<std::size_t>(i)] = kernel;
            if (i != 0) {
                kernel_fft_[fft_size_ - static_cast<std::size_t>(i)] = kernel;
            }
        }
        fft_radix2_inplace(kernel_fft_, false);
    }

    void execute(const float* input, int32_t n_out, float* power_out) {
        const int32_t bins = std::min(n_out, n_ / 2 + 1);
        if (n_ <= 0) {
            std::fill(power_out, power_out + n_out, 0.0F);
            return;
        }

        if (is_power_of_two(n_)) {
            workspace_.resize(static_cast<std::size_t>(n_));
            for (int32_t i = 0; i < n_; ++i) {
                workspace_[static_cast<std::size_t>(i)] = {static_cast<double>(input[i]), 0.0};
            }
            fft_radix2_inplace(workspace_, false);
            write_power(bins, n_out, power_out);
            return;
        }

        workspace_.assign(fft_size_, {0.0, 0.0});
        for (int32_t i = 0; i < n_; ++i) {
            workspace_[static_cast<std::size_t>(i)] =
                static_cast<double>(input[i]) * chirp_[static_cast<std::size_t>(i)];
        }
        fft_radix2_inplace(workspace_, false);
        for (std::size_t i = 0; i < fft_size_; ++i) {
            workspace_[i] *= kernel_fft_[i];
        }
        fft_radix2_inplace(workspace_, true);
        for (int32_t k = 0; k < bins; ++k) {
            const auto value =
                workspace_[static_cast<std::size_t>(k)] * chirp_[static_cast<std::size_t>(k)];
            power_out[k] = static_cast<float>(std::norm(value));
        }
        std::fill(power_out + bins, power_out + n_out, 0.0F);
    }

  private:
    void write_power(int32_t bins, int32_t n_out, float* power_out) const {
        for (int32_t k = 0; k < bins; ++k) {
            power_out[k] = static_cast<float>(std::norm(workspace_[static_cast<std::size_t>(k)]));
        }
        std::fill(power_out + bins, power_out + n_out, 0.0F);
    }

    int32_t n_{0};
    std::size_t fft_size_{0};
    std::vector<std::complex<double>> chirp_;
    std::vector<std::complex<double>> kernel_fft_;
    std::vector<std::complex<double>> workspace_;
};

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
    RfftPowerPlan fft_plan(n_fft);

    const int32_t computed_frames = std::clamp(frames_to_compute, 0, n_frames_raw);
    for (int32_t t = 0; t < computed_frames; ++t) {
        const int32_t start = t * hop_length;
        for (int32_t i = 0; i < n_fft; ++i) {
            windowed[static_cast<std::size_t>(i)] =
                padded[static_cast<std::size_t>(start + i)] * window[static_cast<std::size_t>(i)];
        }

        fft_plan.execute(windowed.data(), n_freq_bins, frame_power.data());

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
    if (n > 0) {
        RfftPowerPlan plan(n);
        plan.execute(input.data(), n_out, power.data());
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
    const int32_t frames_to_compute =
        valid_audio > 0 && hop_length > 0 ? 1 + (pad_size + valid_audio - 1) / hop_length : 0;
    const std::vector<float> padded =
        build_center_padded_audio(samples, n_samples, chunk_length_s, sample_rate, n_fft);

    int32_t n_frames_raw = 0;
    std::vector<float> mel_spec =
        compute_mel_spectrogram(padded, make_hann_window(n_fft), mel_filters, n_fft, hop_length,
                                freq_bins, n_mel_bins, frames_to_compute, n_frames_raw);
    normalize_log_mel_inplace(mel_spec);

    int32_t n_frames_out = 0;
    mel_spec = trim_last_frame(std::move(mel_spec), n_mel_bins, n_frames_raw, n_frames_out);

    MelResult result;
    result.data = std::move(mel_spec);
    result.n_mels = n_mel_bins;
    result.n_frames = n_frames_out;
    return result;
}

} // namespace whisper
} // namespace trtmc

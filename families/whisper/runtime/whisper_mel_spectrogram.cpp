/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/whisper/runtime/whisper_mel_spectrogram.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <utility>
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

// In-place radix-2 transform of one power-of-two size in one direction, without
// the inverse transform's 1/size scale. The bit-reversed order and every
// stage's twiddle factors are tabulated once per instance: stage `length` uses
// w_i = step^i (i < length / 2), produced by the recurrence w_0 = 1,
// w_{i+1} = w_i * step, the same complex operations in the same order as an
// inline per-butterfly update. The values are then the inline update's bit for
// bit wherever the compiler fuses the same multiply-adds in both shapes: with
// the Release flags it fuses none on x86-64, and the same ones on aarch64
// (GCC 13).
class Radix2Transform {
  public:
    Radix2Transform(std::size_t size, bool inverse) : size_(size) {
        bit_reversed_.resize(size_, 0);
        for (std::size_t i = 1, j = 0; i < size_; ++i) {
            std::size_t bit = size_ >> 1;
            for (; (j & bit) != 0; bit >>= 1) {
                j ^= bit;
            }
            j ^= bit;
            bit_reversed_[i] = j;
        }
        // The real and imaginary parts go to separate tables: stored side by
        // side as std::complex, the aarch64 build (GCC 13) fuses multiply-adds
        // differently and no longer matches the inline update bit for bit.
        twiddles_real_.reserve(size_ > 1 ? size_ - 1 : 0);
        twiddles_imag_.reserve(size_ > 1 ? size_ - 1 : 0);
        constexpr double kPi2 = 2.0 * 3.14159265358979323846;
        for (std::size_t length = 2; length <= size_; length <<= 1) {
            const double angle = (inverse ? kPi2 : -kPi2) / static_cast<double>(length);
            const std::complex<double> step(std::cos(angle), std::sin(angle));
            std::complex<double> twiddle(1.0, 0.0);
            for (std::size_t i = 0; i < length / 2; ++i) {
                twiddles_real_.push_back(twiddle.real());
                twiddles_imag_.push_back(twiddle.imag());
                twiddle *= step;
            }
        }
    }

    // Where natural-order element i sits once the values are bit-reversed
    // (the permutation is its own inverse).
    std::size_t bit_reversed(std::size_t i) const { return bit_reversed_[i]; }

    // The whole transform of values in natural order.
    void operator()(std::complex<double>* values) const {
        for (std::size_t i = 1; i < size_; ++i) {
            const std::size_t j = bit_reversed_[i];
            if (i < j) {
                std::swap(values[i], values[j]);
            }
        }
        butterflies(values);
    }

    // The butterfly stages alone, for values already in bit-reversed order.
    void butterflies(std::complex<double>* values) const {
        const double* stage_real = twiddles_real_.data();
        const double* stage_imag = twiddles_imag_.data();
        for (std::size_t length = 2; length <= size_; length <<= 1) {
            const std::size_t half = length / 2;
            for (std::size_t offset = 0; offset < size_; offset += length) {
                for (std::size_t i = 0; i < half; ++i) {
                    const std::complex<double> twiddle(stage_real[i], stage_imag[i]);
                    const auto even = values[offset + i];
                    const auto odd = values[offset + i + half] * twiddle;
                    values[offset + i] = even + odd;
                    values[offset + i + half] = even - odd;
                }
            }
            stage_real += half;
            stage_imag += half;
        }
    }

  private:
    std::size_t size_;
    std::vector<std::size_t> bit_reversed_;
    std::vector<double> twiddles_real_;
    std::vector<double> twiddles_imag_;
};

// Size of the radix-2 transforms behind a frame of length n: n itself when it
// is a power of two, else the next power of two that holds the 2n - 1 values
// of Bluestein's algorithm.
std::size_t transform_size(int32_t n) {
    if (n <= 0) {
        return 0;
    }
    if (is_power_of_two(n)) {
        return static_cast<std::size_t>(n);
    }
    std::size_t size = 1;
    while (size < static_cast<std::size_t>(2 * n - 1)) {
        size <<= 1;
    }
    return size;
}

class RfftPowerPlan {
  public:
    explicit RfftPowerPlan(int32_t n)
        : n_(n), fft_size_(transform_size(n)), forward_(fft_size_, false),
          inverse_(is_power_of_two(n) ? 0 : fft_size_, true), workspace_(fft_size_) {
        if (n_ <= 0 || is_power_of_two(n_)) {
            return;
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
        forward_(kernel_fft_.data());
    }

    void execute(const float* input, int32_t n_out, float* power_out) {
        const int32_t bins = std::min(n_out, n_ / 2 + 1);
        if (n_ <= 0) {
            std::fill(power_out, power_out + n_out, 0.0F);
            return;
        }

        if (is_power_of_two(n_)) {
            // Loaded in bit-reversed order, the frame needs the butterfly
            // stages alone.
            for (int32_t i = 0; i < n_; ++i) {
                workspace_[forward_.bit_reversed(static_cast<std::size_t>(i))] = {
                    static_cast<double>(input[i]), 0.0};
            }
            forward_.butterflies(workspace_.data());
            write_power(bins, n_out, power_out);
            return;
        }

        std::fill(workspace_.begin(), workspace_.end(), std::complex<double>(0.0, 0.0));
        for (int32_t i = 0; i < n_; ++i) {
            workspace_[forward_.bit_reversed(static_cast<std::size_t>(i))] =
                static_cast<double>(input[i]) * chirp_[static_cast<std::size_t>(i)];
        }
        forward_.butterflies(workspace_.data());
        for (std::size_t i = 0; i < fft_size_; ++i) {
            workspace_[i] *= kernel_fft_[i];
        }
        inverse_(workspace_.data());
        // Only the first `bins` values of the inverse transform are read, so
        // only they take its 1/size scale.
        const double scale = 1.0 / static_cast<double>(fft_size_);
        for (int32_t k = 0; k < bins; ++k) {
            workspace_[static_cast<std::size_t>(k)] *= scale;
        }
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

    int32_t n_;
    std::size_t fft_size_;
    Radix2Transform forward_;
    // Bluestein's second transform; empty for a power-of-two length.
    Radix2Transform inverse_;
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
    // STFT centers the fixed-length waveform with reflection, excluding each edge sample.
    for (int32_t i = 0; i < pad_size; ++i) {
        padded[i] = audio_padded[pad_size - i];
        padded[pad_size + audio_length + i] = audio_padded[audio_length - 2 - i];
    }
    return padded;
}

// The half-open column span [first, last) of one filterbank row outside which
// every coefficient is zero (an all-zero row is an empty span).
std::pair<int32_t, int32_t> nonzero_filter_columns(const float* row, int32_t n_mel_bins) {
    int32_t first = 0;
    while (first < n_mel_bins && row[first] == 0.0F) {
        ++first;
    }
    int32_t last = n_mel_bins;
    while (last > first && row[last - 1] == 0.0F) {
        --last;
    }
    return {first, last};
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
    // Each row's nonzero column span, found when a frame first needs the row
    // (first == -1 until then).
    std::vector<std::pair<int32_t, int32_t>> filter_columns(static_cast<std::size_t>(n_freq_bins),
                                                            {-1, -1});

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
            // A finite power adds exactly +0 through a zero coefficient, so
            // only the row's nonzero column span changes frame_mel. A
            // non-finite power keeps the full row (inf * 0 is NaN).
            auto& columns = filter_columns[static_cast<std::size_t>(f)];
            if (columns.first < 0) {
                columns = nonzero_filter_columns(filter_row, n_mel_bins);
            }
            const auto [first, last] = columns;
            const bool finite = std::isfinite(p);
            const int32_t column_end = finite ? last : n_mel_bins;
            for (int32_t m = finite ? first : 0; m < column_end; ++m) {
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

// Drops every row's last frame in place: row m moves from m * n_frames_raw
// down to m * n_frames_out, never past a row still to be moved.
void trim_last_frame(std::vector<float>& mel_spec, int32_t n_mel_bins, int32_t n_frames_raw,
                     int32_t& n_frames_out) {
    n_frames_out = n_frames_raw;
    if (n_frames_raw <= 1) {
        return;
    }

    n_frames_out = n_frames_raw - 1;
    for (int32_t m = 1; m < n_mel_bins; ++m) {
        std::memmove(mel_spec.data() + static_cast<std::size_t>(m) * n_frames_out,
                     mel_spec.data() + static_cast<std::size_t>(m) * n_frames_raw,
                     n_frames_out * sizeof(float));
    }
    mel_spec.resize(static_cast<std::size_t>(n_mel_bins) * n_frames_out);
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
    int32_t n_frames_out = 0;
    trim_last_frame(mel_spec, n_mel_bins, n_frames_raw, n_frames_out);
    normalize_log_mel_inplace(mel_spec);

    MelResult result;
    result.data = std::move(mel_spec);
    result.n_mels = n_mel_bins;
    result.n_frames = n_frames_out;
    return result;
}

} // namespace whisper
} // namespace trtmc

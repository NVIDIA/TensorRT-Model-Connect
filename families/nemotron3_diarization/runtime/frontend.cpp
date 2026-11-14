/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron3_diarization/runtime/frontend.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::nemotron3_diarization {
namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(std::string("nemotron3_diarization front end: ") + message);
}

bool is_power_of_two(std::int32_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

} // namespace

LogMelFrontEnd::LogMelFrontEnd(FrontEndConfig config) : config_(std::move(config)) {
    const auto& c = config_;
    require(c.sample_rate > 0 && c.hop_length > 0 && c.num_mel_bins > 0, "geometry is unusable");
    require(is_power_of_two(c.n_fft) && c.win_length > 1 && c.win_length <= c.n_fft,
            "n_fft must be a power of two no shorter than the window");
    require(c.window.size() == static_cast<std::size_t>(c.win_length),
            "window length does not match win_length");
    const auto bins = static_cast<std::size_t>(c.n_fft / 2 + 1);
    require(c.mel_filters.size() == static_cast<std::size_t>(c.num_mel_bins) * bins,
            "mel filter bank does not match the spectrum");
    require(std::isfinite(c.preemphasis) && c.log_guard > 0.0F, "constants are unusable");

    // torch.stft centres a window shorter than the transform inside it.
    const auto n_fft = static_cast<std::size_t>(c.n_fft);
    const auto offset = static_cast<std::size_t>((c.n_fft - c.win_length) / 2);
    padded_window_.assign(n_fft, 0.0F);
    for (std::size_t i = 0; i < c.window.size(); ++i)
        padded_window_[offset + i] = c.window[i];

    std::size_t bits = 0;
    while ((std::size_t{1} << bits) < n_fft)
        ++bits;
    bit_reversed_.resize(n_fft);
    for (std::size_t i = 0; i < n_fft; ++i) {
        std::size_t reversed = 0;
        for (std::size_t bit = 0; bit < bits; ++bit)
            reversed |= ((i >> bit) & 1U) << (bits - 1 - bit);
        bit_reversed_[i] = reversed;
    }
    twiddle_.resize(n_fft / 2);
    for (std::size_t k = 0; k < twiddle_.size(); ++k) {
        const double angle = -2.0 * M_PI * static_cast<double>(k) / static_cast<double>(n_fft);
        twiddle_[k] = {std::cos(angle), std::sin(angle)};
    }

    // The triangles are sparse; multiplying only their nonzero span gives the
    // same sums, since every skipped product is an exact zero.
    filter_begin_.assign(static_cast<std::size_t>(c.num_mel_bins), 0);
    filter_end_.assign(static_cast<std::size_t>(c.num_mel_bins), 0);
    for (std::size_t mel = 0; mel < filter_begin_.size(); ++mel) {
        const float* row = c.mel_filters.data() + mel * bins;
        std::size_t begin = bins, end = 0;
        for (std::size_t bin = 0; bin < bins; ++bin) {
            if (row[bin] != 0.0F) {
                begin = std::min(begin, bin);
                end = bin + 1;
            }
        }
        filter_begin_[mel] = begin < end ? begin : 0;
        filter_end_[mel] = end;
    }
}

void LogMelFrontEnd::power_spectrum(const std::vector<float>& padded, std::int64_t start,
                                    std::vector<std::complex<double>>& buffer,
                                    std::vector<float>& power) const {
    const std::size_t n = buffer.size();
    for (std::size_t i = 0; i < n; ++i) {
        const float sample = padded[static_cast<std::size_t>(start) + i] * padded_window_[i];
        buffer[bit_reversed_[i]] = {static_cast<double>(sample), 0.0};
    }
    for (std::size_t length = 2; length <= n; length <<= 1) {
        const std::size_t half = length / 2, stride = n / length;
        for (std::size_t base = 0; base < n; base += length) {
            for (std::size_t k = 0; k < half; ++k) {
                const auto even = buffer[base + k];
                const auto odd = buffer[base + k + half] * twiddle_[k * stride];
                buffer[base + k] = even + odd;
                buffer[base + k + half] = even - odd;
            }
        }
    }
    // The reference holds the transform as complex64, takes |X| with sqrt of
    // the float32 sum of squares, and squares that again; each step rounds.
    for (std::size_t bin = 0; bin < power.size(); ++bin) {
        const auto real = static_cast<float>(buffer[bin].real());
        const auto imaginary = static_cast<float>(buffer[bin].imag());
        const float sum = real * real + imaginary * imaginary;
        const float magnitude = std::sqrt(sum);
        power[bin] = magnitude * magnitude;
    }
}

LogMel LogMelFrontEnd::compute(const float* mono, std::int64_t count) const {
    const auto& c = config_;
    require(mono != nullptr || count == 0, "audio pointer is null");
    LogMel result;
    result.valid_frames = count / c.hop_length;
    result.frames = 1 + count / c.hop_length;
    const auto mels = static_cast<std::size_t>(c.num_mel_bins);
    result.values.assign(static_cast<std::size_t>(result.frames) * mels, 0.0F);

    // Pre-emphasis first, then the centring pad of n_fft / 2 zeros at both ends.
    const auto pad = static_cast<std::size_t>(c.n_fft / 2);
    std::vector<float> padded(static_cast<std::size_t>(count) + 2 * pad, 0.0F);
    for (std::int64_t i = 0; i < count; ++i) {
        float value = mono[i];
        if (i > 0) {
            const float previous = c.preemphasis * mono[i - 1];
            value = mono[i] - previous;
        }
        padded[pad + static_cast<std::size_t>(i)] = value;
    }

    std::vector<std::complex<double>> buffer(static_cast<std::size_t>(c.n_fft));
    std::vector<float> power(static_cast<std::size_t>(c.n_fft / 2 + 1));
    const std::size_t bins = power.size();
    // Frames past the valid count are padding: the reference zeroes them.
    for (std::int64_t frame = 0; frame < result.valid_frames; ++frame) {
        power_spectrum(padded, frame * c.hop_length, buffer, power);
        float* out = result.values.data() + static_cast<std::size_t>(frame) * mels;
        for (std::size_t mel = 0; mel < mels; ++mel) {
            const float* row = c.mel_filters.data() + mel * bins;
            double total = 0.0;
            for (std::size_t bin = filter_begin_[mel]; bin < filter_end_[mel]; ++bin)
                total += static_cast<double>(row[bin]) * static_cast<double>(power[bin]);
            out[mel] = std::log(static_cast<float>(total) + c.log_guard);
        }
    }
    return result;
}

std::vector<float> mix_to_mono(const float* samples, std::size_t count, std::uint32_t channels) {
    if (channels <= 1)
        return std::vector<float>(samples, samples + count);
    const std::size_t frames = count / channels;
    std::vector<float> mono(frames);
    for (std::size_t frame = 0; frame < frames; ++frame) {
        double total = 0.0;
        for (std::uint32_t channel = 0; channel < channels; ++channel)
            total += samples[frame * channels + channel];
        mono[frame] = static_cast<float>(total / channels);
    }
    return mono;
}

} // namespace trtmc::nemotron3_diarization

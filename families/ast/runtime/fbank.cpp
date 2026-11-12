/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/ast/runtime/fbank.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace trtmc::ast {
namespace {

bool is_power_of_two(std::int32_t value) {
    return value > 0 && (value & (value - 1)) == 0;
}

void fft_radix2_inplace(std::vector<std::complex<double>>& values) {
    const std::size_t n = values.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; (j & bit) != 0; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(values[i], values[j]);
    }
    for (std::size_t length = 2; length <= n; length <<= 1) {
        const double angle = -2.0 * M_PI / static_cast<double>(length);
        const std::complex<double> step(std::cos(angle), std::sin(angle));
        for (std::size_t start = 0; start < n; start += length) {
            std::complex<double> factor(1.0, 0.0);
            for (std::size_t offset = 0; offset < length / 2; ++offset) {
                const auto even = values[start + offset];
                const auto odd = values[start + offset + length / 2] * factor;
                values[start + offset] = even + odd;
                values[start + offset + length / 2] = even - odd;
                factor *= step;
            }
        }
    }
}

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(std::string("ast filter bank: ") + message);
}

void validate(const FbankConfig& config) {
    require(config.frame_length > 1 && config.frame_shift > 0, "frame geometry is unusable");
    require(config.num_mel_bins > 0 && config.max_length > 0, "spectrogram shape is unusable");
    require(is_power_of_two(config.n_fft) && config.n_fft >= config.frame_length,
            "n_fft must be a power of two no smaller than the frame");
    require(config.window.size() == static_cast<std::size_t>(config.frame_length),
            "window length does not match the frame");
    require(config.mel_filters.size() == static_cast<std::size_t>(config.num_mel_bins) *
                                             static_cast<std::size_t>(config.n_fft / 2 + 1),
            "mel filter bank does not match the spectrum");
    require(config.mel_floor > 0.0F, "mel floor must be positive");
    require(std::isfinite(config.feature_scale) && config.feature_scale != 0.0F,
            "feature scale is unusable");
}

} // namespace

std::int64_t compute_ast_frame_count(std::int64_t num_samples, std::int32_t frame_length,
                                     std::int32_t frame_shift) {
    // Kaldi's snip_edges: only frames that fit entirely inside the waveform.
    // A clip shorter than one frame yields none at all.
    if (frame_length <= 0 || frame_shift <= 0 || num_samples < frame_length)
        return 0;
    return 1 + (num_samples - frame_length) / frame_shift;
}

void compute_ast_frame(const float* samples, const FbankConfig& config, float* frame) {
    const auto length = static_cast<std::size_t>(config.frame_length);

    double total = 0.0;
    for (std::size_t i = 0; i < length; ++i)
        total += samples[i];
    const auto offset = static_cast<float>(total / static_cast<double>(length));

    // Pre-emphasis reads the sample before each one. Kaldi replicates the
    // frame's own first sample rather than reaching back into the waveform, so
    // the first difference is always zero.
    float previous = samples[0] - offset;
    for (std::size_t i = 0; i < length; ++i) {
        const float centred = samples[i] - offset;
        frame[i] = (centred - config.preemphasis * previous) * config.window[i];
        previous = centred;
    }
}

std::vector<float> compute_ast_power_spectrum(const float* frame, const FbankConfig& config) {
    const auto size = static_cast<std::size_t>(config.n_fft);
    std::vector<std::complex<double>> buffer(size, {0.0, 0.0});
    for (std::size_t i = 0; i < static_cast<std::size_t>(config.frame_length); ++i)
        buffer[i] = {static_cast<double>(frame[i]), 0.0};
    fft_radix2_inplace(buffer);

    std::vector<float> power(size / 2 + 1);
    for (std::size_t bin = 0; bin < power.size(); ++bin) {
        const double real = buffer[bin].real();
        const double imaginary = buffer[bin].imag();
        power[bin] = static_cast<float>(real * real + imaginary * imaginary);
    }
    return power;
}

std::vector<float> mix_ast_mono(const float* samples, std::int64_t count, std::int32_t channels) {
    if (channels <= 1)
        return std::vector<float>(samples, samples + count);
    const std::int64_t frames = count / channels;
    std::vector<float> mono(static_cast<std::size_t>(frames));
    for (std::int64_t frame = 0; frame < frames; ++frame) {
        double total = 0.0;
        for (std::int32_t channel = 0; channel < channels; ++channel)
            total += samples[frame * channels + channel];
        mono[static_cast<std::size_t>(frame)] =
            static_cast<float>(total / static_cast<double>(channels));
    }
    return mono;
}

std::vector<float> compute_ast_features(const float* samples, std::int64_t num_samples,
                                        const FbankConfig& config) {
    validate(config);
    const std::int64_t available =
        compute_ast_frame_count(num_samples, config.frame_length, config.frame_shift);
    require(available > 0, "the clip is shorter than one analysis frame");

    const auto mels = static_cast<std::size_t>(config.num_mel_bins);
    const auto bins = static_cast<std::size_t>(config.n_fft / 2 + 1);
    // Frames past max_length are dropped rather than folded in: the reference
    // truncates, so a long clip is scored on its opening window.
    const auto used =
        static_cast<std::size_t>(std::min<std::int64_t>(available, config.max_length));

    std::vector<float> features(static_cast<std::size_t>(config.max_length) * mels, 0.0F);
    std::vector<float> frame(static_cast<std::size_t>(config.frame_length));
    for (std::size_t index = 0; index < used; ++index) {
        compute_ast_frame(samples + index * static_cast<std::size_t>(config.frame_shift), config,
                          frame.data());
        const auto power = compute_ast_power_spectrum(frame.data(), config);
        float* row = features.data() + index * mels;
        for (std::size_t mel = 0; mel < mels; ++mel) {
            const float* filter = config.mel_filters.data() + mel * bins;
            double energy = 0.0;
            for (std::size_t bin = 0; bin < bins; ++bin)
                energy += static_cast<double>(filter[bin]) * static_cast<double>(power[bin]);
            row[mel] = std::log(std::max(static_cast<float>(energy), config.mel_floor));
        }
    }
    // The padded rows are still zero here, and the normalisation that follows
    // is what turns them into the constant the reference produces.
    for (auto& value : features)
        value = (value - config.feature_mean) * config.feature_scale;
    return features;
}

} // namespace trtmc::ast

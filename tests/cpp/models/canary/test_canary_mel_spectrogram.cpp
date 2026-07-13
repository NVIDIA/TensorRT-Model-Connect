/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/canary/canary_mel_spectrogram.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

std::vector<float> make_identity_filterbank(int32_t n_freq_bins, int32_t n_mel_bins) {
    std::vector<float> fb(static_cast<std::size_t>(n_freq_bins) * n_mel_bins, 0.0F);
    const int32_t mapped = std::min(n_freq_bins, n_mel_bins);
    for (int32_t i = 0; i < mapped; ++i) {
        fb[static_cast<std::size_t>(i) * n_mel_bins + i] = 1.0F;
    }
    return fb;
}

std::vector<float> reference_rfft_power(const std::vector<float>& input) {
    const int32_t n = static_cast<int32_t>(input.size());
    const int32_t n_out = n / 2 + 1;
    const double pi2 = 2.0 * 3.14159265358979323846;
    std::vector<float> power(static_cast<std::size_t>(n_out), 0.0F);
    for (int32_t k = 0; k < n_out; ++k) {
        double real = 0.0;
        double imag = 0.0;
        for (int32_t t = 0; t < n; ++t) {
            const double angle =
                pi2 * static_cast<double>(k) * static_cast<double>(t) / static_cast<double>(n);
            real += static_cast<double>(input[static_cast<std::size_t>(t)]) * std::cos(angle);
            imag -= static_cast<double>(input[static_cast<std::size_t>(t)]) * std::sin(angle);
        }
        power[static_cast<std::size_t>(k)] = static_cast<float>(real * real + imag * imag);
    }
    return power;
}

void check_fft_matches_direct(const std::vector<float>& input, const char* test_name) {
    const auto actual = trtmc::canary::detail::rfft_power(input);
    const auto reference = reference_rfft_power(input);
    bool values_match = actual.size() == reference.size();
    for (std::size_t i = 0; values_match && i < actual.size(); ++i) {
        const float tolerance = 1e-5F * std::max(1.0F, std::abs(reference[i]));
        values_match = std::abs(actual[i] - reference[i]) <= tolerance;
    }
    check(values_match, test_name);
}

void test_canary_fft_matches_direct_dft() {
    check_fft_matches_direct({0.25F, -0.5F, 0.75F, 1.0F, -0.25F, 0.125F, 0.5F, -0.75F},
                             "canary radix-2 FFT power matches direct DFT");
    check_fft_matches_direct({0.1F, 0.2F, -0.3F, 0.4F, 0.5F, -0.6F, 0.7F, 0.8F, -0.9F, 1.0F},
                             "canary non-radix-2 fallback matches direct DFT");

    std::vector<float> canary_input(512);
    for (std::size_t i = 0; i < canary_input.size(); ++i) {
        canary_input[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.17) +
                                             0.25 * std::cos(static_cast<double>(i) * 0.07));
    }
    check_fft_matches_direct(canary_input, "canary 512-point FFT power matches direct DFT");
}

void test_canary_shape_and_empty_audio() {
    const int32_t sample_rate = 16000;
    const int32_t n_fft = 512;
    const int32_t win_length = 400;
    const int32_t hop_length = 160;
    const int32_t chunk_length_s = 30;
    const int32_t n_freq_bins = n_fft / 2 + 1;
    const int32_t n_mel_bins = 80;
    auto fb = make_identity_filterbank(n_freq_bins, n_mel_bins);

    const auto mel = trtmc::canary::extract_mel_spectrogram(
        nullptr, 0, fb.data(), n_freq_bins, n_mel_bins, n_fft, win_length, hop_length,
        chunk_length_s, sample_rate, 0.97F, true);

    check(mel.n_mels == n_mel_bins, "canary mel keeps mel bin count");
    check(mel.n_frames == 3000, "canary mel frame count matches 30s HF window");
    check(static_cast<int32_t>(mel.data.size()) == n_mel_bins * mel.n_frames,
          "canary mel data size matches shape");
    check(mel.valid_frames == 0, "canary empty audio has no valid feature frames");
    check(std::all_of(mel.data.begin(), mel.data.end(), [](float value) { return value == 0.0F; }),
          "canary empty audio uses NeMo's zero pad value");
}

trtmc::canary::MelResult extract_test_mel(const std::vector<float>& audio) {
    const int32_t sample_rate = 32;
    const int32_t n_fft = 8;
    const int32_t win_length = 6;
    const int32_t hop_length = 2;
    const int32_t chunk_length_s = 1;
    const int32_t n_freq_bins = n_fft / 2 + 1;
    const int32_t n_mel_bins = n_freq_bins;
    auto fb = make_identity_filterbank(n_freq_bins, n_mel_bins);
    return trtmc::canary::extract_mel_spectrogram(
        audio.data(), static_cast<int32_t>(audio.size()), fb.data(), n_freq_bins, n_mel_bins, n_fft,
        win_length, hop_length, chunk_length_s, sample_rate, 0.97F, true);
}

void test_canary_low_volume_uses_per_feature_normalization() {
    std::vector<float> audio(24);
    for (std::size_t i = 0; i < audio.size(); ++i) {
        audio[i] = static_cast<float>(0.35 * std::sin(static_cast<double>(i) * 0.47) +
                                      0.2 * std::cos(static_cast<double>(i) * 0.19));
    }
    for (float& sample : audio)
        sample *= 0.12F;

    const auto quiet = extract_test_mel(audio);
    bool centered = true;
    bool has_normalized_variation = false;
    for (int32_t m = 0; m < quiet.n_mels; ++m) {
        const std::size_t base = static_cast<std::size_t>(m) * quiet.n_frames;
        double mean = 0.0;
        double square_sum = 0.0;
        for (int32_t t = 0; t < quiet.valid_frames; ++t) {
            mean += quiet.data[base + t];
            square_sum += static_cast<double>(quiet.data[base + t]) * quiet.data[base + t];
        }
        mean /= quiet.valid_frames;
        centered = centered && std::abs(mean) < 1e-5;
        has_normalized_variation = has_normalized_variation || square_sum > 1.0;
    }
    check(centered, "canary low-volume features use NeMo per-feature centering");
    check(has_normalized_variation,
          "canary low-volume features retain normalized speech variation");
}

void test_canary_silence_padding_uses_valid_frame_statistics() {
    std::vector<float> audio(24, 0.0F);
    for (std::size_t i = 6; i < 18; ++i) {
        audio[i] = static_cast<float>(0.4 * std::sin(static_cast<double>(i - 6) * 0.61));
    }
    const auto mel = extract_test_mel(audio);
    check(mel.valid_frames == static_cast<int32_t>(audio.size()) / 2,
          "canary silence-padded audio preserves its valid duration");

    bool normalized = true;
    bool padded_tail_zero = true;
    for (int32_t m = 0; m < mel.n_mels; ++m) {
        const std::size_t base = static_cast<std::size_t>(m) * mel.n_frames;
        double mean = 0.0;
        for (int32_t t = 0; t < mel.valid_frames; ++t)
            mean += mel.data[base + t];
        mean /= mel.valid_frames;
        normalized = normalized && std::abs(mean) < 1e-5;
        for (int32_t t = mel.valid_frames; t < mel.n_frames; ++t)
            padded_tail_zero = padded_tail_zero && mel.data[base + t] == 0.0F;
    }
    check(normalized, "canary silence-padded audio is normalized over valid frames only");
    check(padded_tail_zero, "canary chunk padding uses NeMo's zero feature value");
}

} // namespace

int main() {
    test_canary_fft_matches_direct_dft();
    test_canary_shape_and_empty_audio();
    test_canary_low_volume_uses_per_feature_normalization();
    test_canary_silence_padding_uses_valid_frame_statistics();
    return failures;
}

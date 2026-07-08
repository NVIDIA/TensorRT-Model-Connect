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
    check_fft_matches_direct(
        {0.25F, -0.5F, 0.75F, 1.0F, -0.25F, 0.125F, 0.5F, -0.75F},
        "canary radix-2 FFT power matches direct DFT");
    check_fft_matches_direct({0.1F, 0.2F, -0.3F, 0.4F, 0.5F, -0.6F, 0.7F, 0.8F, -0.9F, 1.0F},
                             "canary non-radix-2 fallback matches direct DFT");

    std::vector<float> canary_input(512);
    for (std::size_t i = 0; i < canary_input.size(); ++i) {
        canary_input[i] = static_cast<float>(
            std::sin(static_cast<double>(i) * 0.17) + 0.25 * std::cos(static_cast<double>(i) * 0.07));
    }
    check_fft_matches_direct(canary_input, "canary 512-point FFT power matches direct DFT");
}

void test_canary_shape_and_empty_audio() {
    const int32_t sample_rate = 16000;
    const int32_t n_fft = 400;
    const int32_t hop_length = 160;
    const int32_t chunk_length_s = 30;
    const int32_t n_freq_bins = n_fft / 2 + 1;
    const int32_t n_mel_bins = 80;
    auto fb = make_identity_filterbank(n_freq_bins, n_mel_bins);

    const auto mel =
        trtmc::canary::extract_mel_spectrogram(nullptr, 0, fb.data(), n_freq_bins, n_mel_bins,
                                               n_fft, hop_length, chunk_length_s, sample_rate);

    check(mel.n_mels == n_mel_bins, "canary mel keeps mel bin count");
    check(mel.n_frames == 3000, "canary mel frame count matches 30s HF window");
    check(static_cast<int32_t>(mel.data.size()) == n_mel_bins * mel.n_frames,
          "canary mel data size matches shape");
    check(std::all_of(mel.data.begin(), mel.data.end(),
                      [](float value) { return value == -1.5F; }),
          "canary empty audio keeps the normalized zero-power value");
}

void test_canary_short_audio_matches_full_zero_padded_reference() {
    const int32_t sample_rate = 32;
    const int32_t n_fft = 8;
    const int32_t hop_length = 2;
    const int32_t chunk_length_s = 1;
    const int32_t n_freq_bins = n_fft / 2 + 1;
    const int32_t n_mel_bins = n_freq_bins;
    auto fb = make_identity_filterbank(n_freq_bins, n_mel_bins);

    std::vector<float> short_audio = {0.25F, -0.5F, 0.75F, 1.0F, -0.25F, 0.125F,
                                      0.5F,  -0.75F, 0.2F, 0.4F,  -0.1F};
    std::vector<float> full_audio(static_cast<std::size_t>(sample_rate * chunk_length_s), 0.0F);
    std::copy(short_audio.begin(), short_audio.end(), full_audio.begin());

    const auto optimized = trtmc::canary::extract_mel_spectrogram(
        short_audio.data(), static_cast<int32_t>(short_audio.size()), fb.data(), n_freq_bins,
        n_mel_bins, n_fft, hop_length, chunk_length_s, sample_rate);
    const auto reference = trtmc::canary::extract_mel_spectrogram(
        full_audio.data(), static_cast<int32_t>(full_audio.size()), fb.data(), n_freq_bins,
        n_mel_bins, n_fft, hop_length, chunk_length_s, sample_rate);

    check(optimized.n_frames == reference.n_frames,
          "canary optimized mel preserves the full chunk shape");
    check(optimized.valid_frames == static_cast<int32_t>(short_audio.size()) / hop_length,
          "canary optimized mel reports short audio frames");
    check(reference.valid_frames == reference.n_frames,
          "canary full zero-padded reference computes the full chunk");
    check(optimized.data.size() == reference.data.size(),
          "canary optimized mel preserves the full data size");

    bool values_match = optimized.data.size() == reference.data.size();
    for (std::size_t i = 0; values_match && i < optimized.data.size(); ++i) {
        values_match = std::abs(optimized.data[i] - reference.data[i]) <= 1e-6F;
    }
    check(values_match, "canary optimized mel matches full zero-padded values");
}

} // namespace

int main() {
    test_canary_fft_matches_direct_dft();
    test_canary_shape_and_empty_audio();
    test_canary_short_audio_matches_full_zero_padded_reference();
    return failures;
}

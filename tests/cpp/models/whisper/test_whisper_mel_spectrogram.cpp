/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/whisper/whisper_mel_spectrogram.h"

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

bool vectors_close(const std::vector<float>& actual, const std::vector<float>& expected,
                   float relative_tolerance, float absolute_tolerance) {
    if (actual.size() != expected.size()) {
        return false;
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float tolerance =
            absolute_tolerance +
            relative_tolerance * std::abs(expected[static_cast<std::size_t>(i)]);
        if (std::abs(actual[i] - expected[i]) > tolerance) {
            std::cerr << "mismatch at " << i << ": actual=" << actual[i]
                      << " expected=" << expected[i] << " tolerance=" << tolerance << '\n';
            return false;
        }
    }
    return true;
}

void check_fft_matches_direct(const std::vector<float>& input, const char* test_name) {
    const auto actual = trtmc::whisper::detail::rfft_power(input);
    const auto reference = reference_rfft_power(input);
    check(vectors_close(actual, reference, 1e-5F, 1e-6F), test_name);
}

void test_whisper_fft_matches_direct_dft() {
    check_fft_matches_direct({0.25F, -0.5F, 0.75F, 1.0F, -0.25F, 0.125F, 0.5F, -0.75F},
                             "whisper radix-2 FFT power matches direct DFT");

    std::vector<float> whisper_input(400);
    for (std::size_t i = 0; i < whisper_input.size(); ++i) {
        whisper_input[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.17) +
                                              0.25 * std::cos(static_cast<double>(i) * 0.07));
    }
    check_fft_matches_direct(whisper_input, "whisper 400-point FFT power matches direct DFT");
}

std::vector<float> make_hf_5_2_filterbank() {
    // WhisperFeatureExtractor 5.2.0 with feature_size=4,
    // sampling_rate=16000, n_fft=32. Layout is [frequency, mel].
    return {
        0.0F,           0.0F,           0.0F,           0.0F,           0.00133960015F,
        0.0F,           0.0F,           0.0F,           0.00060509576F, 0.00073521355F,
        0.0F,           0.0F,           0.0F,           0.00088614809F, 0.00016090245F,
        0.0F,           0.0F,           0.00033586150F, 0.00046726403F, 0.0F,
        0.0F,           0.0F,           0.00059016780F, 0.00003439805F, 0.0F,
        0.0F,           0.00042571520F, 0.00012267498F, 0.0F,           0.0F,
        0.00026126259F, 0.00021095191F, 0.0F,           0.0F,           0.00009680999F,
        0.00029922883F, 0.0F,           0.0F,           0.0F,           0.00033170475F,
        0.0F,           0.0F,           0.0F,           0.00028431836F, 0.0F,
        0.0F,           0.0F,           0.00023693196F, 0.0F,           0.0F,
        0.0F,           0.00018954557F, 0.0F,           0.0F,           0.0F,
        0.00014215918F, 0.0F,           0.0F,           0.0F,           0.00009477279F,
        0.0F,           0.0F,           0.0F,           0.00004738639F, 0.0F,
        0.0F,           0.0F,           0.0F,
    };
}

trtmc::whisper::MelResult extract_hf_fixture_mel(const std::vector<float>& audio) {
    const auto filterbank = make_hf_5_2_filterbank();
    return trtmc::whisper::extract_mel_spectrogram(audio.data(), static_cast<int32_t>(audio.size()),
                                                   filterbank.data(), 17, 4, 32, 8, 1, 16000);
}

void check_hf_5_2_fixture(const std::vector<float>& audio, const std::vector<int32_t>& indices,
                          const std::vector<float>& expected, const char* test_name) {
    const auto mel = extract_hf_fixture_mel(audio);
    check(mel.n_mels == 4 && mel.n_frames == 2000, "Whisper HF fixture shape matches");
    std::vector<float> selected;
    selected.reserve(static_cast<std::size_t>(mel.n_mels) * indices.size());
    for (int32_t m = 0; m < mel.n_mels; ++m) {
        for (const int32_t index : indices) {
            selected.push_back(mel.data[static_cast<std::size_t>(m) * mel.n_frames + index]);
        }
    }
    check(vectors_close(selected, expected, 1e-5F, 2e-5F), test_name);
}

void test_whisper_matches_pinned_hf_frontend() {
    std::vector<float> short_audio(64, 0.0F);
    for (int32_t i = 0; i < 32; ++i) {
        short_audio[static_cast<std::size_t>(i + 16)] =
            static_cast<float>(0.35 * std::sin(static_cast<double>(i) * 0.47) +
                               0.2 * std::cos(static_cast<double>(i) * 0.19));
    }
    const std::vector<int32_t> short_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1999};
    const std::vector<float> short_expected = {
        -1.47796965F, 0.19769770F,  0.49146098F,  0.52203035F,  0.37673461F,  0.47700906F,
        0.44652843F,  0.16935682F,  -1.47796965F, -1.47796965F, -1.47796965F, -1.47796965F,
        -1.47796965F, -1.47796965F, 0.14345938F,  0.44907600F,  0.51333833F,  0.45681787F,
        0.50315177F,  0.45515430F,  0.14392549F,  -1.47796965F, -1.47796965F, -1.47796965F,
        -1.47796965F, -1.47796965F, -1.47796965F, -0.00149941F, 0.21142197F,  0.23270231F,
        0.23056060F,  0.27122152F,  0.29486173F,  0.06656158F,  -1.47796965F, -1.47796965F,
        -1.47796965F, -1.47796965F, -1.47796965F, -1.47796965F, -0.27518058F, -0.17686820F,
        -0.36955905F, -0.69783723F, -0.04921615F, 0.08560586F,  -0.08074880F, -1.47796965F,
        -1.47796965F, -1.47796965F, -1.47796965F, -1.47796965F,
    };
    check_hf_5_2_fixture(short_audio, short_indices, short_expected,
                         "Whisper short audio matches Transformers 5.2.0");

    const std::vector<float> padded_audio(64, 0.0F);
    const std::vector<int32_t> padded_indices = {0, 1, 2, 999, 1999};
    check_hf_5_2_fixture(padded_audio, padded_indices, std::vector<float>(20, -1.5F),
                         "Whisper zero-padded audio matches Transformers 5.2.0");

    std::vector<float> boundary_audio(16000, 0.0F);
    for (int32_t i = 0; i < 15968; ++i) {
        boundary_audio[static_cast<std::size_t>(i + 16)] =
            static_cast<float>(0.35 * std::sin(static_cast<double>(i) * 0.047) +
                               0.2 * std::cos(static_cast<double>(i) * 0.019));
    }
    const std::vector<int32_t> boundary_indices = {0,    1,    2,    3,    10,  999,
                                                   1995, 1996, 1997, 1998, 1999};
    const std::vector<float> boundary_expected = {
        -1.39574528F, 0.09532887F,  0.41899896F,  0.51257664F,  0.37318176F,  0.35541523F,
        0.53302789F,  0.49865597F,  0.43587410F,  0.29114467F,  -0.11682808F, -1.39574528F,
        0.02567625F,  0.19092834F,  -0.06663859F, -0.05304563F, -0.15682101F, -0.23419797F,
        -0.14918065F, -0.05524480F, 0.07153308F,  -0.17537153F, -1.39574528F, -0.11697865F,
        -0.00449359F, -0.16378736F, -0.49035490F, -0.60813010F, -0.72875285F, -0.60238636F,
        -0.29121733F, -0.17834580F, -0.31444991F, -1.39574528F, -0.25144815F, -0.12285542F,
        -0.29862618F, -0.92274916F, -1.04521441F, -1.18764424F, -1.04027009F, -0.43942106F,
        -0.30394208F, -0.47242820F,
    };
    check_hf_5_2_fixture(boundary_audio, boundary_indices, boundary_expected,
                         "Whisper chunk-boundary audio matches Transformers 5.2.0");
}

void test_whisper_skipped_tail_matches_full_zero_padding() {
    std::vector<float> short_audio(64, 0.0F);
    for (std::size_t i = 0; i < short_audio.size(); ++i) {
        short_audio[i] = static_cast<float>(0.3 * std::sin(static_cast<double>(i) * 0.41));
    }
    std::vector<float> explicitly_padded(16000, 0.0F);
    std::copy(short_audio.begin(), short_audio.end(), explicitly_padded.begin());

    const auto short_mel = extract_hf_fixture_mel(short_audio);
    const auto full_mel = extract_hf_fixture_mel(explicitly_padded);
    check(vectors_close(short_mel.data, full_mel.data, 1e-6F, 1e-6F),
          "Whisper skipped zero tail preserves full-padding features");
}

void test_whisper_shape_and_energy() {
    const int32_t sample_rate = 16000;
    const int32_t n_fft = 400;
    const int32_t hop_length = 160;
    const int32_t chunk_length_s = 30;
    const int32_t n_freq_bins = n_fft / 2 + 1;
    const int32_t n_mel_bins = 80;
    auto fb = make_identity_filterbank(n_freq_bins, n_mel_bins);

    std::vector<float> sine(sample_rate);
    const double pi2 = 2.0 * 3.14159265358979323846;
    for (int32_t i = 0; i < sample_rate; ++i) {
        sine[static_cast<std::size_t>(i)] =
            static_cast<float>(std::sin(pi2 * 440.0 * static_cast<double>(i) / sample_rate));
    }

    const auto mel = trtmc::whisper::extract_mel_spectrogram(
        sine.data(), static_cast<int32_t>(sine.size()), fb.data(), n_freq_bins, n_mel_bins, n_fft,
        hop_length, chunk_length_s, sample_rate);

    check(mel.n_mels == n_mel_bins, "whisper mel keeps mel bin count");
    check(mel.n_frames == 3000, "whisper mel frame count matches 30s HF window");
    check(static_cast<int32_t>(mel.data.size()) == n_mel_bins * mel.n_frames,
          "whisper mel data size matches shape");

    float energy_target = 0.0F;
    float energy_quiet = 0.0F;
    const int32_t check_frames = std::min(mel.n_frames, 50);
    for (int32_t t = 0; t < check_frames; ++t) {
        energy_target += mel.data[static_cast<std::size_t>(11) * mel.n_frames + t];
        energy_quiet += mel.data[static_cast<std::size_t>(50) * mel.n_frames + t];
    }
    check(energy_target > energy_quiet, "whisper mel keeps 440Hz energy concentrated");
}

} // namespace

int main() {
    test_whisper_fft_matches_direct_dft();
    test_whisper_matches_pinned_hf_frontend();
    test_whisper_skipped_tail_matches_full_zero_padding();
    test_whisper_shape_and_energy();
    return failures;
}

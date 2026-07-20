/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for RNNT-owned audio feature extraction helpers.

#include "runtime/models/nemotron_speech_streaming/audio_helpers.h"
#include "utils/wav_reader.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void check_close(float actual, float expected, float tolerance, const char* name) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

std::vector<float> make_identity_filterbank(int32_t n_freq_bins) {
    std::vector<float> filters(static_cast<std::size_t>(n_freq_bins) * n_freq_bins, 0.0F);
    for (int32_t i = 0; i < n_freq_bins; ++i) {
        filters[static_cast<std::size_t>(i) * n_freq_bins + i] = 1.0F;
    }
    return filters;
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

void check_vectors_close(const std::vector<float>& actual, const std::vector<float>& expected,
                         float tolerance, const char* name) {
    check(actual.size() == expected.size(), name);
    for (std::size_t i = 0; i < std::min(actual.size(), expected.size()); ++i) {
        const float scaled_tolerance = tolerance * std::max(1.0F, std::abs(expected[i]));
        check_close(actual[i], expected[i], scaled_tolerance, name);
    }
}

void test_rnnt_fft_matches_direct_dft() {
    std::vector<float> input(512);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(std::sin(static_cast<double>(i) * 0.17) +
                                      0.25 * std::cos(static_cast<double>(i) * 0.07));
    }
    check_vectors_close(trtmc::rnnt::detail::rfft_power(input), reference_rfft_power(input), 1e-5F,
                        "rnnt 512-point FFT matches direct DFT");
}

trtmc::rnnt::MelSpectrogramOptions make_incremental_options() {
    trtmc::rnnt::MelSpectrogramOptions options;
    options.n_fft = 8;
    options.win_length = 6;
    options.hop_length = 2;
    options.chunk_length_s = 1;
    options.sample_rate = 32;
    options.symmetric_window = true;
    options.center_window_in_fft = true;
    options.preemphasis = 0.97F;
    options.log_scale = trtmc::rnnt::MelLogScale::kNaturalLog;
    return options;
}

std::vector<float> make_test_audio(int32_t count) {
    std::vector<float> audio(static_cast<std::size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
        audio[static_cast<std::size_t>(i)] =
            static_cast<float>(0.35 * std::sin(static_cast<double>(i) * 0.47) +
                               0.2 * std::cos(static_cast<double>(i) * 0.19));
    }
    return audio;
}

std::vector<float> materialize_incremental(const trtmc::rnnt::IncrementalMelSpectrogram& state) {
    std::vector<float> output;
    output.reserve(static_cast<std::size_t>(state.n_mels()) * state.frame_count());
    for (int32_t mel = 0; mel < state.n_mels(); ++mel) {
        for (int32_t frame = 0; frame < state.frame_count(); ++frame) {
            output.push_back(state.value(mel, frame));
        }
    }
    return output;
}

std::vector<float> run_resampled_chunks(const std::vector<float>& source_audio,
                                        const std::vector<int32_t>& chunk_sizes,
                                        trtmc::rnnt::IncrementalMelStats& stats) {
    const auto options = make_incremental_options();
    const auto filters = make_identity_filterbank(5);
    trtmc::rnnt::IncrementalMelSpectrogram state(filters.data(), 5, 5, options, 96);
    int32_t accepted = 0;
    for (const int32_t chunk_size : chunk_sizes) {
        const int32_t take =
            std::min(chunk_size, static_cast<int32_t>(source_audio.size()) - accepted);
        state.accept_audio(source_audio.data() + accepted, take);
        accepted += take;
        const bool final = accepted == static_cast<int32_t>(source_audio.size());
        if (final) {
            state.ensure_frames(16, true);
            break;
        }

        constexpr int32_t kResampleHalfTaps = 16;
        const int32_t stable_target = std::max(0, accepted - kResampleHalfTaps) / 3;
        const int32_t stable_frames =
            stable_target >= options.n_fft / 2
                ? 1 + (stable_target - options.n_fft / 2) / options.hop_length
                : 0;
        state.ensure_frames(stable_frames, false);
    }
    stats = state.stats();
    return materialize_incremental(state);
}

void test_incremental_mel_processes_each_frame_once() {
    const auto options = make_incremental_options();
    const auto filters = make_identity_filterbank(5);
    const auto audio = make_test_audio(32);
    const auto offline = trtmc::rnnt::extract_configured_mel_spectrogram(
        audio.data(), static_cast<int32_t>(audio.size()), filters.data(), 5, 5, options);

    trtmc::rnnt::IncrementalMelSpectrogram state(filters.data(), 5, 5, options, 32);
    state.accept_audio(audio.data(), 8);
    state.ensure_frames(2, false);
    state.accept_audio(audio.data() + 8, 8);
    state.ensure_frames(6, false);
    state.accept_audio(audio.data() + 16, 16);
    state.ensure_frames(16, true);

    check_vectors_close(materialize_incremental(state), offline.data, 1e-5F,
                        "incremental mel matches one-shot features");
    const auto stats = state.stats();
    check(stats.accepted_source_samples == 32, "incremental mel accepts each source sample once");
    check(stats.generated_resampled_samples == 32,
          "incremental mel materializes each identity-rate sample once");
    check(stats.computed_mel_frames == 16, "incremental mel computes each frame once");
}

void test_incremental_resample_matches_one_shot() {
    const auto options = make_incremental_options();
    const auto filters = make_identity_filterbank(5);
    const auto source_audio = make_test_audio(96);
    const auto resampled = trtmc::resample_linear(source_audio.data(), 96, 96, 32);
    const auto offline = trtmc::rnnt::extract_configured_mel_spectrogram(
        resampled.data(), static_cast<int32_t>(resampled.size()), filters.data(), 5, 5, options);

    trtmc::rnnt::IncrementalMelSpectrogram state(filters.data(), 5, 5, options, 96);
    state.accept_audio(source_audio.data(), 72);
    state.ensure_frames(6, false);
    state.accept_audio(source_audio.data() + 72, 24);
    state.ensure_frames(16, true);

    check_vectors_close(materialize_incremental(state), offline.data, 1e-5F,
                        "incremental resampling matches one-shot features");
    const auto stats = state.stats();
    check(stats.accepted_source_samples == 96,
          "incremental resampling accepts each source sample once");
    check(stats.generated_resampled_samples == 32,
          "incremental resampling generates each target sample once");
    check(stats.computed_mel_frames == 16, "incremental resampling computes each mel frame once");
}

void test_incremental_resample_is_chunk_size_independent() {
    const auto source_audio = make_test_audio(96);
    trtmc::rnnt::IncrementalMelStats coarse_stats;
    trtmc::rnnt::IncrementalMelStats irregular_stats;
    const auto coarse = run_resampled_chunks(source_audio, {24, 24, 24, 24}, coarse_stats);
    const auto irregular = run_resampled_chunks(source_audio, {17, 31, 11, 37}, irregular_stats);

    check_vectors_close(irregular, coarse, 0.0F,
                        "incremental features are independent of source chunk sizes");
    check(coarse_stats.accepted_source_samples == 96 &&
              irregular_stats.accepted_source_samples == 96,
          "chunk schedules accept each source sample once");
    check(coarse_stats.generated_resampled_samples == 32 &&
              irregular_stats.generated_resampled_samples == 32,
          "chunk schedules generate each resampled sample once");
    check(coarse_stats.computed_mel_frames == 16 && irregular_stats.computed_mel_frames == 16,
          "chunk schedules compute each mel frame once");
}

void test_rnnt_mel_matches_configured_owned_options() {
    const std::vector<float> samples{0.0F, 0.1F, -0.2F, 0.4F, -0.1F, 0.0F, 0.2F, -0.3F};
    const int32_t n_fft = 4;
    const int32_t n_freq_bins = 3;
    const int32_t n_mel_bins = 2;
    const std::vector<float> mel_filters{
        0.5F, 0.1F, 0.2F, 0.7F, 0.3F, 0.4F,
    };

    const auto rnnt = trtmc::rnnt::extract_rnnt_mel_spectrogram(
        samples.data(), static_cast<int32_t>(samples.size()), mel_filters.data(), n_freq_bins,
        n_mel_bins, n_fft, /*win_length=*/3, /*hop_length=*/2, /*chunk_length_s=*/1,
        /*sample_rate=*/8, /*preemphasis=*/0.97F);

    trtmc::rnnt::MelSpectrogramOptions options;
    options.n_fft = n_fft;
    options.win_length = 3;
    options.hop_length = 2;
    options.chunk_length_s = 1;
    options.sample_rate = 8;
    options.symmetric_window = true;
    options.center_window_in_fft = true;
    options.preemphasis = 0.97F;
    options.log_scale = trtmc::rnnt::MelLogScale::kNaturalLog;
    const auto configured = trtmc::rnnt::extract_configured_mel_spectrogram(
        samples.data(), static_cast<int32_t>(samples.size()), mel_filters.data(), n_freq_bins,
        n_mel_bins, options);

    check(rnnt.n_mels == configured.n_mels, "rnnt mel keeps configured mel count");
    check(rnnt.n_frames == configured.n_frames, "rnnt mel keeps configured frame count");
    check(rnnt.data.size() == configured.data.size(), "rnnt mel keeps configured data size");
    for (std::size_t i = 0; i < rnnt.data.size(); ++i) {
        check_close(rnnt.data[i], configured.data[i], 1e-6F, "rnnt mel value matches configured");
    }
}

} // namespace

int main() {
    test_rnnt_fft_matches_direct_dft();
    test_rnnt_mel_matches_configured_owned_options();
    test_incremental_mel_processes_each_frame_once();
    test_incremental_resample_matches_one_shot();
    test_incremental_resample_is_chunk_size_independent();

    if (g_failures != 0) {
        std::cerr << g_failures << " rnnt audio helper test(s) failed\n";
        return 1;
    }
    return 0;
}

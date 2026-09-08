/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron_voicechat/runtime/audio_helpers.h"
#include "families/nemotron_voicechat/runtime/pipeline.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace voicechat = trtmc::nemotron_voicechat;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void test_fixed_first_and_steady_contract() {
    const auto first = voicechat::make_streaming_mel_step(true, 0, 8, false);
    check(first.history_frames == 0 && first.requested_new_frames == 1 &&
              first.valid_new_frames == 1 && first.engine_frames == 1,
          "first 80 ms input consumes the one-frame first-step plan");
    const auto steady = voicechat::make_streaming_mel_step(false, 1, 16, false);
    check(steady.history_frames == 9 && steady.requested_new_frames == 8 &&
              steady.valid_new_frames == 8 && steady.engine_frames == 17,
          "steady input is nine cached plus eight new mel frames");

    const auto rollover_cold_start = voicechat::make_streaming_mel_step(false, 0, 9, false);
    check(rollover_cold_start.history_frames == 9 &&
              rollover_cold_start.requested_new_frames == 8 &&
              rollover_cold_start.valid_new_frames == 8 && rollover_cold_start.engine_frames == 17,
          "a rolled context cold-starts the resident steady plan with masked history");
}

void test_model_card_partial_tail() {
    // center=True produces floor(N / hop) + 1 frames: the 249734-sample
    // model-card input has 1561. After first + 194 steady chunks the next
    // index is 1553, leaving a complete eight-frame final step whose right
    // STFT boundary is supplied by reflect padding.
    const auto tail = voicechat::make_streaming_mel_step(false, 1553, 1561, true);
    check(tail.valid_new_frames == 8 && tail.engine_frames == 17,
          "model-card final audio tail consumes the centered reflection frame");
    for (int32_t remaining = 1; remaining <= 7; ++remaining) {
        const auto boundary = voicechat::make_streaming_mel_step(false, 100, 100 + remaining, true);
        check(boundary.valid_new_frames == remaining && boundary.engine_frames == 17,
              "all final partial mel counts preserve the fixed engine shape");
    }
}

void test_long_session_policy() {
    voicechat::Config config;
    config.tts_max_cache_length = 512;
    check(voicechat::streaming_frontend_capacity_seconds(config) == 10487,
          "frontend capacity follows logical TTS positions, not the rolling physical cache");
}

void test_resampled_stream_crosses_physical_tts_cache_boundary() {
    voicechat::Config config;
    config.tts_max_cache_length = 512;

    // A scaled 3:1 resampling stress path with the live 80-ms/eight-mel-frame
    // cadence. Production resamples to native rate before this frontend, but
    // the old physical-cache-derived cap failed in either path: at 42 seconds,
    // the 526th chunk requested centered-STFT samples beyond the clamp.
    trtmc::voicechat_audio::MelSpectrogramOptions options;
    options.n_fft = 4;
    options.win_length = 4;
    options.hop_length = 2;
    options.chunk_length_s = voicechat::streaming_frontend_capacity_seconds(config);
    options.sample_rate = 200;
    options.center_window_in_fft = true;
    options.log_scale = trtmc::voicechat_audio::MelLogScale::kNaturalLog;
    const std::array<float, 3> filterbank = {1.0F, 0.0F, 0.0F};
    const std::array<float, 4> exact_window = {1.0F, 1.0F, 1.0F, 1.0F};
    const std::array<float, 48> source_frame{};
    trtmc::voicechat_audio::IncrementalMelSpectrogram mel(
        filterbank.data(), 3, 1, options, 600, exact_window.data(),
        static_cast<int32_t>(exact_window.size()));

    bool completed = false;
    try {
        constexpr int32_t kInputFrames = 526;
        for (int32_t input_frame = 0; input_frame < kInputFrames; ++input_frame) {
            mel.accept_audio(source_frame.data(), static_cast<int32_t>(source_frame.size()));
            const int32_t requested_mel_frames = input_frame == 0 ? 1 : 1 + 8 * input_frame;
            mel.ensure_frames(requested_mel_frames, false);
        }
        completed = true;
    } catch (const std::runtime_error& error) {
        std::cerr << "long resampled stream failed: " << error.what() << '\n';
    }
    check(completed, "resampled frontend remains live beyond 524 80-ms input frames");
    if (completed)
        check(mel.frame_count() == 4201,
              "long resampled frontend materializes every requested mel frame");
}

void test_checkpoint_window_and_reflect_boundary() {
    check(trtmc::voicechat_audio::detail::reflect_index(-2, 4) == 2 &&
              trtmc::voicechat_audio::detail::reflect_index(-1, 4) == 1 &&
              trtmc::voicechat_audio::detail::reflect_index(4, 4) == 2 &&
              trtmc::voicechat_audio::detail::reflect_index(5, 4) == 1,
          "centered STFT reflection matches torch reflect padding on both boundaries");

    trtmc::voicechat_audio::MelSpectrogramOptions options;
    options.n_fft = 4;
    options.win_length = 4;
    options.hop_length = 2;
    options.chunk_length_s = 1;
    options.sample_rate = 8;
    options.center_window_in_fft = true;
    options.log_scale = trtmc::voicechat_audio::MelLogScale::kNaturalLog;
    const std::array<float, 3> filterbank = {1.0F, 0.0F, 0.0F};
    const std::array<float, 4> exact_window = {1.0F, 1.0F, 1.0F, 1.0F};

    bool rejected = false;
    try {
        trtmc::voicechat_audio::IncrementalMelSpectrogram invalid(filterbank.data(), 3, 1, options,
                                                                  8, exact_window.data(), 3);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "checkpoint mel window length is validated exactly");

    trtmc::voicechat_audio::IncrementalMelSpectrogram mel(filterbank.data(), 3, 1, options, 8,
                                                          exact_window.data(), 4);
    const std::array<float, 4> signal = {1.0F, 2.0F, 3.0F, 4.0F};
    mel.accept_audio(signal.data(), static_cast<int32_t>(signal.size()));
    mel.ensure_frames(3, true);
    check(std::abs(mel.value(0, 0) - std::log(64.0F)) < 1.0e-5F &&
              std::abs(mel.value(0, 1) - std::log(100.0F)) < 1.0e-5F &&
              std::abs(mel.value(0, 2) - std::log(144.0F)) < 1.0e-5F,
          "incremental centered STFT matches left and right reflect-padding oracles");

    options.chunk_length_s = 2;
    trtmc::voicechat_audio::IncrementalMelSpectrogram exact_boundary(
        filterbank.data(), 3, 1, options, 8, exact_window.data(), 4);
    const std::array<float, 10> boundary_signal = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
                                                   6.0F, 7.0F, 8.0F, 9.0F, 10.0F};
    exact_boundary.accept_audio(boundary_signal.data(),
                                static_cast<int32_t>(boundary_signal.size()));
    exact_boundary.ensure_frames(5, false);
    check(exact_boundary.frame_count() == 5,
          "non-final centered frame accepts an exact right sample boundary");
}

void test_equal_rate_stream_rebase_is_sample_exact() {
    trtmc::voicechat_audio::MelSpectrogramOptions options;
    options.n_fft = 8;
    options.win_length = 8;
    options.hop_length = 2;
    options.chunk_length_s = 4;
    options.sample_rate = 64;
    options.center_window_in_fft = true;
    options.preemphasis = 0.73F;
    options.log_scale = trtmc::voicechat_audio::MelLogScale::kNaturalLog;
    const std::array<float, 10> filterbank = {0.8F, 0.2F, 0.5F, 0.5F, 0.3F,
                                              0.7F, 0.6F, 0.4F, 0.9F, 0.1F};
    const std::array<float, 8> exact_window = {0.2F, 0.5F, 0.8F, 1.0F, 1.0F, 0.8F, 0.5F, 0.2F};
    auto make_mel = [&] {
        return trtmc::voicechat_audio::IncrementalMelSpectrogram(
            filterbank.data(), 5, 2, options, options.sample_rate, exact_window.data(),
            static_cast<int32_t>(exact_window.size()));
    };
    auto baseline = make_mel();
    auto rebased = make_mel();
    std::vector<float> signal(112);
    for (std::size_t index = 0; index < signal.size(); ++index)
        signal[index] = static_cast<float>((static_cast<int32_t>(index * 7U) % 23) - 11) * 0.031F;

    constexpr int32_t kInitialSamples = 64;
    constexpr int32_t kInitialNextFrame = 25;
    constexpr int32_t kHistoryFrames = 3;
    baseline.accept_audio(signal.data(), kInitialSamples);
    rebased.accept_audio(signal.data(), kInitialSamples);
    baseline.ensure_frames(kInitialNextFrame, false);
    rebased.ensure_frames(kInitialNextFrame, false);
    int32_t baseline_next = kInitialNextFrame;
    int32_t rebased_next = rebased.rebase_streaming(kInitialNextFrame, kHistoryFrames);
    check(rebased_next == 6,
          "mel rebase retains history plus centered-STFT/preemphasis guard frames");

    auto compare_chunk = [&](int32_t baseline_start, int32_t rebased_start, int32_t frames,
                             const char* message) {
        bool equal = true;
        for (int32_t bin = 0; bin < 2; ++bin) {
            for (int32_t frame = 0; frame < frames; ++frame) {
                const float expected = baseline.value(bin, baseline_start + frame);
                const float actual = rebased.value(bin, rebased_start + frame);
                equal = equal && std::memcmp(&expected, &actual, sizeof(float)) == 0;
            }
        }
        check(equal, message);
    };

    for (int32_t chunk = 0; chunk < 2; ++chunk) {
        const int32_t offset = kInitialSamples + chunk * 16;
        baseline.accept_audio(signal.data() + offset, 16);
        rebased.accept_audio(signal.data() + offset, 16);
        baseline.ensure_frames(baseline_next + 8, false);
        rebased.ensure_frames(rebased_next + 8, false);
        compare_chunk(baseline_next - kHistoryFrames, rebased_next - kHistoryFrames, 11,
                      "rebased mel chunks are bitwise equal across the rollover boundary");
        baseline_next += 8;
        rebased_next += 8;
    }

    rebased_next = rebased.rebase_streaming(rebased_next, kHistoryFrames);
    check(rebased_next == 6, "mel stream can be rebased repeatedly with a fixed memory bound");
    baseline.accept_audio(signal.data() + 96, 16);
    rebased.accept_audio(signal.data() + 96, 16);
    baseline.ensure_frames(baseline_next + 8, false);
    rebased.ensure_frames(rebased_next + 8, false);
    compare_chunk(baseline_next - kHistoryFrames, rebased_next - kHistoryFrames, 11,
                  "a repeated mel rebase preserves sample phase and overlapping features");

    auto wrong_rate = trtmc::voicechat_audio::IncrementalMelSpectrogram(
        filterbank.data(), 5, 2, options, options.sample_rate * 2, exact_window.data(),
        static_cast<int32_t>(exact_window.size()));
    bool rejected_rate = false;
    try {
        (void)wrong_rate.rebase_streaming(25, kHistoryFrames);
    } catch (const std::logic_error&) {
        rejected_rate = true;
    }
    check(rejected_rate, "mel rebase rejects a stream with unresolved resampling phase");

    auto short_stream = make_mel();
    short_stream.accept_audio(signal.data(), 8);
    short_stream.ensure_frames(1, false);
    check(short_stream.rebase_streaming(1, kHistoryFrames) == 1 && short_stream.frame_count() == 1,
          "an early recovery rollover preserves an undersized mel prefix verbatim");
}

} // namespace

int main() {
    test_fixed_first_and_steady_contract();
    test_model_card_partial_tail();
    test_long_session_policy();
    test_resampled_stream_crosses_physical_tts_cache_boundary();
    test_checkpoint_window_and_reflect_boundary();
    test_equal_rate_stream_rebase_is_sample_exact();
    return failures;
}

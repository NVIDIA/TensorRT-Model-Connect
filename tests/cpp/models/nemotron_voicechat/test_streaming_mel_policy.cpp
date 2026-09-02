/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/nemotron_voicechat/audio_helpers.h"
#include "runtime/models/nemotron_voicechat/pipeline.h"

#include <array>
#include <cmath>
#include <cstdint>
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
    check(voicechat::streaming_frontend_capacity_seconds(config) == 601,
          "frontend capacity covers the 600-second TTS state plus final padding");
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
}

} // namespace

int main() {
    test_fixed_first_and_steady_contract();
    test_model_card_partial_tail();
    test_long_session_policy();
    test_checkpoint_window_and_reflect_boundary();
    return failures;
}

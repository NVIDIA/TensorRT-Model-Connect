/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/whisper/runtime/whisper_mel_spectrogram.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

struct Golden {
    int mel;
    int frame;
    float value;
};

void check_values(const trtmc::whisper::MelResult& result, std::initializer_list<Golden> expected) {
    const bool valid_shape = (result.n_mels == 80 || result.n_mels == 128) &&
                             result.n_frames == 3000 &&
                             result.data.size() == static_cast<std::size_t>(result.n_mels) * 3000;
    check(valid_shape, "30-second output has the expected mel shape");
    if (!valid_shape) {
        return;
    }
    for (const auto& point : expected) {
        const float actual = result.data[point.mel * result.n_frames + point.frame];
        if (!std::isfinite(actual) || std::abs(actual - point.value) > 3e-5F) {
            std::fprintf(stderr, "FAIL: mel %d frame %d: expected %.8f, got %.8f\n", point.mel,
                         point.frame, point.value, actual);
            ++failures;
        }
    }
}

trtmc::whisper::MelResult extract(const std::vector<float>& audio, int mels) {
    // A sparse test filterbank selects FFT bin m + 1 for mel m.
    std::vector<float> filters(201 * mels, 0.0F);
    for (int m = 0; m < mels; ++m) {
        filters[(m + 1) * mels + m] = 1.0F;
    }
    return trtmc::whisper::extract_mel_spectrogram(audio.data(), static_cast<int32_t>(audio.size()),
                                                   filters.data(), 201, mels, 400, 160, 30, 16000);
}

std::vector<float> tone(int samples) {
    std::vector<float> audio(samples);
    constexpr double pi2 = 6.28318530717958647692;
    for (int i = 0; i < samples; ++i) {
        audio[i] = static_cast<float>(0.25 * std::cos(pi2 * 3.0 * i / 400.0) +
                                      0.15 * std::sin(pi2 * 7.0 * i / 400.0));
    }
    return audio;
}

// Goldens from Transformers 5.2.0 WhisperFeatureExtractor with the sparse
// filterbank above and n_fft=400, hop_length=160, sampling_rate=16000,
// chunk_length=30. The signal has nonzero endpoints so center padding is visible.
void test_center_padding(int mels) {
    const auto short_result = extract(tone(16000), mels);
    check_values(short_result, {{0, 0, 1.22718477F},
                                {2, 0, 1.72748566F},
                                {6, 0, 0.91715926F},
                                {0, 1, 0.67869687F},
                                {0, 2999, -0.27251434F}});

    auto audio = tone(480000);
    const auto full_result = extract(audio, mels);
    check_values(full_result, {{0, 0, 1.22718477F},
                               {0, 2999, 0.70422518F},
                               {2, 2999, 1.69706297F},
                               {6, 2999, 1.58521879F}});

    audio.resize(480160, 1.0F);
    check(extract(audio, mels).data == full_result.data,
          "samples beyond the 30-second chunk do not affect its reflection");
}

void test_discarded_frame_does_not_set_log_floor(int mels) {
    std::vector<float> audio(480000, 0.0F);
    for (int i = 0; i < 160; ++i) {
        audio[480000 - 160 + i] = static_cast<float>(i + 1) / 320.0F;
    }
    const auto result = extract(audio, mels);
    check_values(result, {{0, 0, -0.40593076F},
                          {79, 1000, -0.40593076F},
                          {0, 2999, 1.59406924F},
                          {2, 2999, 1.10322535F},
                          {6, 2999, 0.74954677F}});
}

void test_empty_and_single_sample(int mels) {
    const auto empty = extract({}, mels);
    check(empty.n_mels == mels && empty.n_frames == 3000, "empty input keeps output shape");
    check(std::all_of(empty.data.begin(), empty.data.end(),
                      [](float value) { return value == -1.5F; }),
          "empty input produces the silence floor");
    check_values(extract({0.5F}, mels),
                 {{0, 0, 0.84948498F}, {0, 1, 0.33946735F}, {0, 2999, -1.15051508F}});
}

} // namespace

int main() {
    for (const int mels : {80, 128}) {
        test_center_padding(mels);
        test_discarded_frame_does_not_set_log_floor(mels);
        test_empty_and_single_sample(mels);
    }
    if (failures == 0) {
        std::printf("Whisper mel frontend tests passed\n");
    }
    return failures == 0 ? 0 : 1;
}

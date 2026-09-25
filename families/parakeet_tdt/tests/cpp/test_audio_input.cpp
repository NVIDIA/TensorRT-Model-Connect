/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/parakeet_tdt/runtime/audio_input.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int failures = 0;
void check(bool value, const char* label) {
    if (!value) {
        std::cerr << label << '\n';
        ++failures;
    }
}
template <class F>
void rejects(F run, const char* label) {
    try {
        run();
        check(false, label);
    } catch (const std::invalid_argument&) {
    }
}

int main() {
    using trtmc::internal::SpeechTranscriptionRequest;
    using trtmc::tdt::prepare_audio;
    float stereo[] = {1, 3, -2, 2, 4, 2};
    SpeechTranscriptionRequest request;
    request.audio = {{stereo, 6}, 16000, 2};
    auto mono = prepare_audio(request);
    check(mono == std::vector<float>({2, 0, 3}), "downmix interleaved frames");
    check(stereo[0] == 1 && stereo[1] == 3, "preserve borrowed input");
    request.audio.sample_rate.reset();
    check(prepare_audio(request) == mono, "omitted rate uses model 16 kHz default");
    request.audio.channels = 0;
    rejects([&] { prepare_audio(request); }, "reject missing channels");
    request.audio.channels = 4;
    rejects([&] { prepare_audio(request); }, "reject partial frame");
    request.audio.channels = 2;
    request.audio.sample_rate = 0;
    rejects([&] { prepare_audio(request); }, "reject zero rate");
    request.audio.sample_rate = std::numeric_limits<unsigned>::max();
    rejects([&] { prepare_audio(request); }, "reject out-of-range rate");
    request.audio.sample_rate = 16000;
    request.source_language = "en";
    rejects([&] { prepare_audio(request); }, "reject unsupported language control");
    request.source_language.reset();
    request.audio.samples = {nullptr, 6};
    rejects([&] { prepare_audio(request); }, "reject null borrowed buffer");
    request.audio.samples = {};
    rejects([&] { prepare_audio(request); }, "reject empty input");
    request.audio.samples = {stereo, 6};
    stereo[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { prepare_audio(request); }, "reject NaN");
    stereo[0] = std::numeric_limits<float>::infinity();
    rejects([&] { prepare_audio(request); }, "reject infinity");
    stereo[0] = stereo[1] = std::numeric_limits<float>::max();
    check(std::isfinite(prepare_audio(request)[0]), "downmix avoids intermediate float overflow");
    std::vector<float> pcm(48000, 0.25F);
    request.audio = {{pcm.data(), pcm.size()}, 48000, 1};
    auto resampled = prepare_audio(request);
    check(resampled.size() == 16000, "resample 48 kHz to 16 kHz");
    check(std::abs(resampled[8000] - 0.25F) < 1e-6F, "resampling preserves constant signal");
    pcm.resize(16000 * 30);
    request.audio = {{pcm.data(), pcm.size()}, 16000, 1};
    check(prepare_audio(request).size() == pcm.size(), "accept exact 30-second window");
    pcm.push_back(0);
    request.audio.samples = {pcm.data(), pcm.size()};
    rejects([&] { prepare_audio(request); }, "reject overlong input instead of truncating");
    request.audio.samples = {stereo, static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1};
    rejects([&] { prepare_audio(request); }, "reject sample count before narrowing or dereference");
    return failures == 0 ? 0 : 1;
}

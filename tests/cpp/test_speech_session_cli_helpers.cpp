/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/speech_session_helpers.h"

#include <iostream>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

trtmc::SpeechSessionEvent audio_event(std::initializer_list<float> samples) {
    trtmc::SpeechSessionEvent event;
    event.kind = trtmc::SpeechSessionEventKind::kAgentAudio;
    event.audio_samples = samples;
    event.sample_rate = 22050;
    return event;
}

trtmc::SpeechSessionEvent text_event(std::uint64_t epoch, std::string text, bool final) {
    trtmc::SpeechSessionEvent event;
    event.kind = trtmc::SpeechSessionEventKind::kAgentText;
    event.epoch = epoch;
    event.text = std::move(text);
    event.is_final = final;
    return event;
}

void test_audio_and_text_aggregation() {
    std::vector<trtmc::SpeechSessionEvent> events;
    events.push_back(audio_event({0.1F, 0.2F}));
    events.push_back(text_event(7, "Hel", false));
    events.push_back(text_event(7, "lo", false));
    events.push_back(text_event(7, "Hello", true));
    events.push_back(audio_event({0.3F}));
    events.push_back(text_event(9, "world", false));

    const auto result = trtmc::cli::aggregate_speech_session_events(std::move(events), 16000);
    check(result.audio.samples == std::vector<float>({0.1F, 0.2F, 0.3F}) &&
              result.audio.num_samples == 3 && result.audio.sample_rate == 22050,
          "CLI aggregates ordered native agent audio events");
    check(result.agent_text == "Hello world",
          "CLI prefers final turn text without duplicating partial events");
}

void test_tail_frame_contract() {
    check(trtmc::cli::speech_tail_frame_samples(16000) == 1280 &&
              trtmc::cli::speech_tail_frame_samples(22050) == 1764 &&
              trtmc::cli::speech_tail_frame_samples(44100) == 3528,
          "CLI tail frames preserve the VoiceChat 80 ms timeline at arbitrary input rates");
}

} // namespace

int main() {
    test_audio_and_text_aggregation();
    test_tail_frame_contract();
    return failures;
}

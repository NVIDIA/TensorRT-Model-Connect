/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/speech_session.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::cli {

struct SpeechSessionCliResult {
    AudioResult audio;
    std::string agent_text;
};

inline int32_t speech_tail_frame_samples(int32_t sample_rate) {
    if (sample_rate <= 0)
        throw std::invalid_argument("speech tail sample rate must be positive");
    return static_cast<int32_t>(std::llround(static_cast<double>(sample_rate) * 0.08));
}

inline std::unique_ptr<ISpeechSession>
create_cli_speech_session(IPipeline& pipeline, const SpeechSessionConfig& config) {
    if (auto* batch = dynamic_cast<ISpeechBatchSessionProvider*>(&pipeline))
        return batch->create_batch_speech_session(config);
    if (auto* live = dynamic_cast<ISpeechSessionProvider*>(&pipeline))
        return live->create_speech_session(config);
    return {};
}

inline SpeechSessionCliResult
aggregate_speech_session_events(std::vector<SpeechSessionEvent> events,
                                int32_t fallback_sample_rate) {
    struct TurnText {
        std::uint64_t epoch{0};
        std::string partial;
        std::string final;
        bool has_final{false};
    };

    SpeechSessionCliResult result;
    result.audio.sample_rate = fallback_sample_rate;
    std::vector<TurnText> turns;
    for (auto& event : events) {
        if (event.kind == SpeechSessionEventKind::kAgentAudio) {
            if (event.sample_rate <= 0)
                throw std::runtime_error("speech session emitted audio without a sample rate");
            if (!result.audio.samples.empty() && result.audio.sample_rate != event.sample_rate)
                throw std::runtime_error("speech session changed output sample rate mid-stream");
            result.audio.sample_rate = event.sample_rate;
            result.audio.samples.insert(result.audio.samples.end(), event.audio_samples.begin(),
                                        event.audio_samples.end());
            continue;
        }
        if (event.kind != SpeechSessionEventKind::kAgentText)
            continue;
        auto turn = std::find_if(turns.begin(), turns.end(), [&](const TurnText& candidate) {
            return candidate.epoch == event.epoch;
        });
        if (turn == turns.end()) {
            turns.push_back(TurnText{event.epoch});
            turn = std::prev(turns.end());
        }
        if (event.is_final) {
            turn->final = std::move(event.text);
            turn->has_final = true;
        } else {
            turn->partial += event.text;
        }
    }

    result.audio.num_samples = static_cast<int32_t>(result.audio.samples.size());
    for (const auto& turn : turns) {
        const auto& text = turn.has_final ? turn.final : turn.partial;
        if (text.empty())
            continue;
        if (!result.agent_text.empty())
            result.agent_text.push_back(' ');
        result.agent_text += text;
    }
    return result;
}

} // namespace trtmc::cli

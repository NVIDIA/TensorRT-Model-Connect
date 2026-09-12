/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include "playback_queue.h"
#include "trtmc/speech.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <iostream>
#include <iterator>
#include <mutex>
#include <thread>

namespace trtmc::examples::voicechat {
class RunState {
  public:
    bool stopping() const noexcept { return stopping_.load(); }

    void request_stop() noexcept { stopping_.store(true); }

    void fail(std::exception_ptr failure) noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!failure_)
                failure_ = std::move(failure);
        }
        request_stop();
    }

    void rethrow_if_failed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failure_)
            std::rethrow_exception(failure_);
    }

  private:
    std::atomic<bool> stopping_{false};
    mutable std::mutex mutex_;
    std::exception_ptr failure_;
};

class TranscriptPrinter {
  public:
    template <class Event>
    void agent_text(const Event& event) {
        if (event.epoch != agent_epoch_) {
            finish_agent_line();
            agent_epoch_ = event.epoch;
            saw_agent_delta_ = false;
        }
        if (!agent_line_open_) {
            std::cout << "agent> " << std::flush;
            agent_line_open_ = true;
        }
        if (!event.is_final) {
            std::cout << event.text << std::flush;
            saw_agent_delta_ = true;
        } else {
            if (!saw_agent_delta_)
                std::cout << event.text;
            finish_agent_line();
        }
    }

    template <class Event>
    void user_text(const Event& event) {
        if (!event.is_final || event.text.empty())
            return;
        finish_agent_line();
        std::cout << "user> " << event.text << '\n';
    }

    void status(const std::string& text) {
        finish_agent_line();
        std::cout << '[' << text << "]\n";
    }

    void finish_agent_line() {
        if (agent_line_open_)
            std::cout << '\n';
        agent_line_open_ = false;
        saw_agent_delta_ = false;
    }

  private:
    std::uint64_t agent_epoch_{0};
    bool agent_line_open_{false};
    bool saw_agent_delta_{false};
};

inline void require_audio_formats(const SpeechSessionInfo& info, int input_rate, int output_rate) {
    if (info.input.channels != 1 || info.input.sample_rate != static_cast<uint32_t>(input_rate) ||
        !info.output || info.output->channels != 1 ||
        info.output->sample_rate != static_cast<uint32_t>(output_rate))
        throw std::runtime_error("speech session format does not match the mono ALSA endpoints");
}
inline SpeechSession create_sdk_session(const Model& model, int input_rate, int output_rate,
                                        const std::optional<std::string>& system_prompt, int seed) {
    const SpeechDialogueRequest input{{static_cast<uint32_t>(input_rate), 1}, system_prompt};
    const Config config{{"output_sample_rate", int64_t{output_rate}},
                        {"seed", int64_t{seed}},
                        {"emit_agent_audio", true},
                        {"emit_agent_text", true},
                        {"emit_user_transcript", true},
                        {"enable_barge_in", true}};
    auto session = model.task<DuplexSpeechDialogue>().create(input, config);
    require_audio_formats(session.info(), input_rate, output_rate);
    return session;
}

// A rejected append accepted no samples. Keep this one captured chunk until
// accepted or shutdown; never read a replacement chunk or create another queue.
inline void append_captured_audio(SpeechSession& session, Span<const float> audio, RunState& state,
                                  const volatile std::sig_atomic_t& signal) {
    while (!state.stopping() && signal == 0) {
        if (session.append_audio(audio))
            return;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}
inline void consume_sdk_event(const SpeechDialogueEventView& event, int output_rate,
                              PlaybackQueue& queue, TranscriptPrinter& printer, RunState& state) {
    switch (event.kind) {
    case SpeechEventKind::AgentAudio: {
        if (event.audio.channels != 1 || !event.audio.sample_rate ||
            *event.audio.sample_rate != static_cast<uint32_t>(output_rate))
            throw std::runtime_error("speech session changed its mono output format");
        std::vector<int16_t> pcm;
        pcm.reserve(event.audio.samples.size());
        std::transform(event.audio.samples.begin(), event.audio.samples.end(),
                       std::back_inserter(pcm), float_to_pcm16);
        if (!queue.try_push(std::move(pcm)))
            throw std::runtime_error("playback queue exceeded its four-second bound");
        break;
    }
    case SpeechEventKind::AgentText:
        printer.agent_text(event);
        break;
    case SpeechEventKind::UserTranscript:
        printer.user_text(event);
        break;
    case SpeechEventKind::Yielded:
        (void)queue.request_flush();
        printer.status(event.text.empty() ? "yielded" : "yielded: " + std::string(event.text));
        break;
    case SpeechEventKind::Cancelled:
        (void)queue.request_flush();
        printer.status("cancelled");
        state.request_stop();
        break;
    case SpeechEventKind::Reset:
        (void)queue.request_flush();
        printer.status("reset");
        break;
    case SpeechEventKind::Error:
        (void)queue.request_flush();
        throw std::runtime_error(event.text.empty() ? "speech session failed"
                                                    : std::string(event.text));
    case SpeechEventKind::InputFinished:
        state.request_stop();
        break;
    case SpeechEventKind::TurnFinished:
        printer.finish_agent_line();
        break;
    default:
        break;
    }
}
inline bool poll_sdk_session(SpeechSession& session, int output_rate, PlaybackQueue& queue,
                             TranscriptPrinter& printer, RunState& state, int timeout_ms) {
    auto read = session.read_events(timeout_ms);
    if (read.status == SpeechPollStatus::Timeout)
        return true;
    if (read.status == SpeechPollStatus::EpochEnd) {
        state.request_stop();
        return false;
    }
    auto& events = *read.events;
    for (size_t index = 0; index < events.size(); ++index)
        consume_sdk_event(events.at(index), output_rate, queue, printer, state);
    if (events.state() == SpeechReadState::Failed) {
        (void)queue.request_flush();
        throw std::runtime_error("speech session failed");
    }
    if (events.state() == SpeechReadState::EpochEnded)
        state.request_stop();
    return !state.stopping();
}
} // namespace trtmc::examples::voicechat

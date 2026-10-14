/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2_audio/runtime/audio_dsp.h"
#include "trtmc/internal/speech.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <variant>

namespace trtmc::lfm2_audio {
using namespace internal;
#include "session_options.inc"

struct FakeTokenizer {
    std::function<void()> before_decode;
    std::vector<std::int32_t> encode(const std::string&) const { return {1}; }
    std::string decode(const std::vector<std::int32_t>&) const {
        if (before_decode)
            before_decode();
        return "reply";
    }
};

// The real Session below owns every lifecycle decision. This fake substitutes
// only tokenization and model execution, with a deterministic startup barrier.
struct Engines {
    std::shared_ptr<FakeTokenizer> tokenizer = std::make_shared<FakeTokenizer>();
    std::mutex mutex;
    int max_context{1024};
    int generation_calls{0};
    std::function<void()> before_assistant;
    std::function<void()> after_emit;

    History start(const std::string&) const {
        History history;
        history.embeddings.resize(2048);
        return history;
    }
    void append_text(History& history, const std::string& text) {
        if (text == "<|im_start|>assistant\n" && before_assistant)
            before_assistant();
        history.embeddings.resize(history.embeddings.size() + 2048);
    }
    void append_audio(History& history, const AudioView&) {
        history.embeddings.resize(history.embeddings.size() + 2048);
    }
    Generation generate(History&, const Options&, bool, const Emit& emit) {
        ++generation_calls;
        Generation generated;
        generated.text = {42};
        generated.waveform = {0.25F, -0.25F};
        generated.stopped = !emit(generated, false);
        if (after_emit)
            after_emit();
        if (!generated.stopped)
            generated.stopped = !emit(generated, true);
        return generated;
    }
};

// Visibility changes only: the delivery test probes the actual state mutex to
// arrange cancellation without a sleep or scheduler-dependent assertion. No
// production state is changed and every lifecycle statement stays unchanged.
#define private public
#include "session_under_test.inc"
#undef private

namespace {
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

std::string event_name(SpeechEventKind kind) {
    switch (kind) {
    case SpeechEventKind::AgentText:
        return "AgentText";
    case SpeechEventKind::AgentAudio:
        return "AgentAudio";
    case SpeechEventKind::TurnStarted:
        return "TurnStarted";
    case SpeechEventKind::TurnFinished:
        return "TurnFinished";
    case SpeechEventKind::Cancelled:
        return "Cancelled";
    case SpeechEventKind::Reset:
        return "Reset";
    case SpeechEventKind::InputFinished:
        return "InputFinished";
    default:
        return std::to_string(static_cast<unsigned>(kind));
    }
}

void expect_events(Session& session, std::initializer_list<SpeechEventKind> kinds,
                   SpeechReadState state, std::uint64_t epoch = 1) {
    const auto batch = session.read_events(0);
    require(batch.state == state, "unexpected dialogue terminal state");
    std::string actual_order, expected_order;
    for (const auto& event : batch.events)
        actual_order += event_name(event.kind) + " ";
    for (const auto kind : kinds)
        expected_order += event_name(kind) + " ";
    if (actual_order != expected_order)
        throw std::runtime_error("event order: expected [" + expected_order + "], got [" +
                                 actual_order + "]");
    auto expected = kinds.begin();
    std::uint64_t previous = 0;
    for (const auto& event : batch.events) {
        require(event.kind == *expected++, "unexpected dialogue event order");
        require(event.epoch == epoch, "event belongs to the wrong epoch");
        require(event.sequence > previous, "event sequence is not increasing");
        previous = event.sequence;
        if (event.kind == SpeechEventKind::TurnFinished)
            require(event.is_final, "TurnFinished is not final");
        if (event.kind == SpeechEventKind::AgentText)
            require(event.text == "reply", "generated text changed");
        if (event.kind == SpeechEventKind::AgentAudio) {
            require(event.audio.samples == std::vector<float>({0.25F, -0.25F}),
                    "generated audio changed");
            require(event.audio.sample_rate == 24000 && event.audio.channels == 1 &&
                        event.media_start_sample == 0 && event.media_end_sample == 2,
                    "generated media bounds changed");
        }
    }
}

void expect_drained(Session& session) {
    // A terminal epoch must return immediately even for a blocking reader.
    // The Python subprocess deadline catches a regression that waits forever.
    const auto batch = session.read_events(-1);
    require(batch.events.empty() && batch.state == SpeechReadState::EpochEnded,
            "drained terminal epoch did not end its blocking read");
}

void expect_closed_input(Session& session) {
    bool rejected = false;
    try {
        session.append_text("must not be accepted");
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "closed epoch accepted text input");
}

void check_finished(Session& session, const std::shared_ptr<Engines>& engines, int calls) {
    require(engines->generation_calls == calls, "finish regenerated a completed response");
    session.finish_input();
    expect_events(session, {}, SpeechReadState::EpochEnded);
    expect_drained(session);
    expect_closed_input(session);
    session.reset();
    expect_events(session, {SpeechEventKind::Reset}, SpeechReadState::Active, 2);
    require(session.append_text("next epoch"), "reset did not reopen dialogue input");
}

void finish_case(const std::string& mode) {
    const auto engines = std::make_shared<Engines>();
    Session session(engines, {{16000, 1}, std::nullopt}, {});
    if (mode != "finish-empty")
        require(session.append_text("user input"), "input was unexpectedly backpressured");
    if (mode == "finish-completed") {
        session.commit_input_turn();
        expect_events(session,
                      {SpeechEventKind::TurnStarted, SpeechEventKind::AgentText,
                       SpeechEventKind::AgentAudio, SpeechEventKind::TurnFinished},
                      SpeechReadState::EpochEnded);
    } else if (mode == "finish-deferred") {
        session.commit_input_turn(false);
        expect_events(session, {}, SpeechReadState::Active);
    }
    session.finish_input();
    if (mode == "finish-completed" || mode == "finish-empty")
        expect_events(session, {SpeechEventKind::InputFinished}, SpeechReadState::EpochEnded);
    else
        expect_events(session,
                      {SpeechEventKind::TurnStarted, SpeechEventKind::AgentText,
                       SpeechEventKind::AgentAudio, SpeechEventKind::TurnFinished,
                       SpeechEventKind::InputFinished},
                      SpeechReadState::EpochEnded);
    check_finished(session, engines, mode == "finish-empty" ? 0 : 1);
}

class StartupBarrier {
  public:
    void arrive() {
        std::lock_guard<std::mutex> lock(mutex_);
        entered_ = true;
        cv_.notify_all();
    }
    void pause() {
        std::unique_lock<std::mutex> lock(mutex_);
        entered_ = true;
        cv_.notify_all();
        cv_.wait(lock, [&] { return released_; });
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        require(cv_.wait_for(lock, std::chrono::seconds(5), [&] { return entered_; }),
                "response never reached the startup barrier");
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }

  private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool entered_{false}, released_{false};
};

void cancel_startup() {
    const auto engines = std::make_shared<Engines>();
    Session session(engines, {{16000, 1}, std::nullopt}, {});
    require(session.append_text("user input"), "input was unexpectedly backpressured");
    session.commit_input_turn(false);
    StartupBarrier barrier;
    engines->before_assistant = [&] { barrier.pause(); };
    std::exception_ptr worker_error;
    std::thread response([&] {
        try {
            session.create_response();
        } catch (...) {
            worker_error = std::current_exception();
        }
    });
    try {
        barrier.wait();
        session.cancel();
    } catch (...) {
        barrier.release();
        response.join();
        throw;
    }
    barrier.release();
    response.join();
    if (worker_error)
        std::rethrow_exception(worker_error);
    expect_events(
        session,
        {SpeechEventKind::TurnStarted, SpeechEventKind::Cancelled, SpeechEventKind::TurnFinished},
        SpeechReadState::EpochEnded);
    expect_events(session, {}, SpeechReadState::EpochEnded);
    expect_drained(session);
    expect_closed_input(session);
    engines->before_assistant = {};
    session.reset();
    expect_events(session, {SpeechEventKind::Reset}, SpeechReadState::Active, 2);
    require(session.append_text("next epoch"), "cancelled session did not reset");
    session.commit_input_turn();
    expect_events(session,
                  {SpeechEventKind::TurnStarted, SpeechEventKind::AgentText,
                   SpeechEventKind::AgentAudio, SpeechEventKind::TurnFinished},
                  SpeechReadState::EpochEnded, 2);
}

void cancel_delivery() {
    const auto engines = std::make_shared<Engines>();
    Session session(engines, {{16000, 1}, std::nullopt}, {});
    require(session.append_text("user input"), "input was unexpectedly backpressured");
    session.commit_input_turn(false);
    StartupBarrier decoding, cancelled;
    engines->tokenizer->before_decode = [&] { decoding.pause(); };
    // Do not let response completion overtake the cancellation being tested.
    engines->after_emit = [&] { cancelled.wait(); };
    std::exception_ptr response_error, cancel_error;
    std::thread response([&] {
        try {
            session.create_response();
        } catch (...) {
            response_error = std::current_exception();
        }
    });
    std::thread canceller;
    const auto cancel = [&] {
        try {
            session.cancel();
        } catch (...) {
            cancel_error = std::current_exception();
        }
        cancelled.arrive();
    };
    try {
        decoding.wait();
        if (session.state_mutex_.try_lock()) {
            session.state_mutex_.unlock();
            // The unfixed callback permits cancellation while decode is paused.
            // Complete that cancellation first to force the forbidden ordering.
            cancel();
        } else {
            // With serialized delivery, cancellation waits for this callback.
            canceller = std::thread(cancel);
        }
    } catch (...) {
        cancelled.arrive();
        decoding.release();
        response.join();
        if (canceller.joinable())
            canceller.join();
        throw;
    }
    decoding.release();
    response.join();
    if (canceller.joinable())
        canceller.join();
    if (response_error)
        std::rethrow_exception(response_error);
    if (cancel_error)
        std::rethrow_exception(cancel_error);
    expect_events(session,
                  {SpeechEventKind::TurnStarted, SpeechEventKind::AgentText,
                   SpeechEventKind::AgentAudio, SpeechEventKind::Cancelled,
                   SpeechEventKind::TurnFinished},
                  SpeechReadState::EpochEnded);
    expect_drained(session);
    expect_closed_input(session);
    engines->tokenizer->before_decode = {};
    engines->after_emit = {};
    session.reset();
    expect_events(session, {SpeechEventKind::Reset}, SpeechReadState::Active, 2);
    require(session.append_text("next epoch"), "cancelled delivery did not reset");
}
} // namespace
} // namespace trtmc::lfm2_audio

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("expected one lifecycle case name");
        if (std::string(argv[1]) == "cancel-startup")
            trtmc::lfm2_audio::cancel_startup();
        else if (std::string(argv[1]) == "cancel-delivery")
            trtmc::lfm2_audio::cancel_delivery();
        else
            trtmc::lfm2_audio::finish_case(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

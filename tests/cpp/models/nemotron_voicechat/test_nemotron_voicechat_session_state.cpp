/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/nemotron_voicechat/session_state.h"
#include "runtime/models/nemotron_voicechat/thinker_hybrid_state.h"
#include "runtime/models/nemotron_voicechat/thinker_inference_state.h"
#include "runtime/models/nemotron_voicechat/thinker_kv_cache.h"
#include "runtime/models/nemotron_voicechat/thinker_mamba_state.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace voicechat = trtmc::nemotron_voicechat;

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_frame_constants_and_chunk_accumulation() {
    static_assert(voicechat::kInputFrameSamples == 1280);
    static_assert(voicechat::kOutputFrameSamples == 1764);
    static_assert(
        std::is_base_of_v<trtmc::VoiceChatThinkerInferenceState, trtmc::VoiceChatThinkerKvCache>);
    static_assert(std::is_base_of_v<trtmc::VoiceChatThinkerInferenceState,
                                    trtmc::VoiceChatThinkerMambaState>);
    static_assert(std::is_base_of_v<trtmc::VoiceChatThinkerInferenceState,
                                    trtmc::VoiceChatThinkerHybridState>);
    static_assert(
        std::is_same_v<decltype(&trtmc::VoiceChatThinkerInferenceState::prepare_step),
                       void (trtmc::VoiceChatThinkerInferenceState::*)(trtmc::TensorMap&)>);
    static_assert(std::is_same_v<decltype(&trtmc::VoiceChatThinkerInferenceState::advance),
                                 void (trtmc::VoiceChatThinkerInferenceState::*)()>);
    static_assert(std::is_constructible_v<trtmc::VoiceChatThinkerKvCache, int32_t, int32_t, int32_t,
                                          cudaStream_t>);
    static_assert(!std::is_constructible_v<trtmc::VoiceChatThinkerKvCache, int32_t, int32_t,
                                           int32_t, cudaStream_t, trtmc::DType>);

    voicechat::FrameScheduler scheduler(9);
    std::vector<float> first(1000, 1.0F);
    std::vector<float> second(1560, 2.0F);
    scheduler.append(first.data(), static_cast<int32_t>(first.size()));
    check(!scheduler.pop().has_value(), "scheduler retains an incomplete input chunk");
    scheduler.append(second.data(), static_cast<int32_t>(second.size()));

    const auto frame0 = scheduler.pop();
    const auto frame1 = scheduler.pop();
    check(frame0.has_value() && frame1.has_value() && !scheduler.pop().has_value(),
          "scheduler emits two complete frames across chunk boundaries");
    check(frame0->epoch == 9 && frame0->frame_index == 0 && frame1->frame_index == 1,
          "scheduler tags frames with epoch and monotonic indices");
    check(frame0->input_start_sample == 0 && frame0->input_end_sample == 1280 &&
              frame1->input_start_sample == 1280 && frame1->input_end_sample == 2560,
          "scheduler exposes input media ranges");
    check(frame0->output_start_sample == 0 && frame0->output_end_sample == 1764 &&
              frame1->output_start_sample == 1764 && frame1->output_end_sample == 3528,
          "scheduler maps the shared 80 ms timeline to 22.05 kHz output");
    check(frame0->samples[999] == 1.0F && frame0->samples[1000] == 2.0F &&
              frame1->samples.front() == 2.0F && frame1->samples.back() == 2.0F,
          "scheduler preserves sample order across chunk boundaries");
}

void test_frame_finish_padding_and_reset() {
    voicechat::FrameScheduler scheduler(2);
    const std::array<float, 3> tail = {0.1F, 0.2F, 0.3F};
    scheduler.append(tail.data(), static_cast<int32_t>(tail.size()));
    scheduler.finish();
    const auto frame = scheduler.pop();
    check(frame.has_value() && frame->valid_input_samples == 3 && frame->is_final,
          "finish exposes a final partial frame");
    check(frame->samples[0] == 0.1F && frame->samples[2] == 0.3F && frame->samples[3] == 0.0F &&
              frame->samples.back() == 0.0F,
          "final partial frame is zero padded to 1280 samples");
    check(!scheduler.pop().has_value(), "finish does not fabricate an empty frame");

    bool rejected = false;
    try {
        scheduler.append(tail.data(), 1);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    check(rejected, "scheduler rejects append after finish");

    scheduler.reset(3);
    check(scheduler.epoch() == 3 && scheduler.next_frame_index() == 0 &&
              scheduler.pending_samples() == 0 && !scheduler.finished(),
          "scheduler reset starts a fresh epoch and timeline");
}

void test_conversation_epoch_and_yield_contract() {
    voicechat::ConversationState state;
    check(state.phase() == voicechat::ConversationPhase::kListening && state.can_accept_audio(),
          "conversation starts listening");

    const auto first_epoch = state.begin_agent_turn();
    check(state.phase() == voicechat::ConversationPhase::kAgentSpeaking &&
              state.accepts_output(first_epoch),
          "agent generation owns the active epoch");
    check(state.next_sequence() == 0 && state.next_sequence() == 1,
          "event sequence is monotonic within an epoch");

    check(state.barge_in(), "barge-in yields an active agent turn");
    check(state.phase() == voicechat::ConversationPhase::kListening &&
              state.yield_reason() == voicechat::YieldReason::kBargeIn &&
              !state.accepts_output(first_epoch),
          "barge-in rejects stale agent output by epoch");

    const auto second_epoch = state.begin_agent_turn();
    check(second_epoch != first_epoch && state.yield_to_user(),
          "model yield invalidates a later agent epoch");
    check(state.yield_reason() == voicechat::YieldReason::kModelYield &&
              !state.accepts_output(second_epoch),
          "model-yield state is distinct from barge-in");

    const auto third_epoch = state.begin_agent_turn();
    state.cancel();
    check(state.phase() == voicechat::ConversationPhase::kCancelled && !state.can_accept_audio() &&
              !state.accepts_output(third_epoch),
          "cancel terminates input and rejects queued output");
    state.reset();
    check(state.phase() == voicechat::ConversationPhase::kListening && state.can_accept_audio(),
          "reset reopens a clean conversation");
}

void test_conversation_finish() {
    voicechat::ConversationState state;
    const auto epoch = state.begin_agent_turn();
    state.finish_input();
    check(!state.can_accept_audio() && state.accepts_output(epoch),
          "finishing input still permits the active reply to drain");
    const auto completed_epoch = state.finish_agent_turn();
    check(completed_epoch == epoch && state.phase() == voicechat::ConversationPhase::kFinished &&
              !state.accepts_output(epoch),
          "finished conversation rejects late output from the completed turn");
}

void test_wait_events_terminal_phase_policy() {
    check(!voicechat::event_wait_is_terminal(voicechat::ConversationPhase::kFinished, false),
          "caller-side finished phase does not bypass queued worker input");
    check(voicechat::event_wait_is_terminal(voicechat::ConversationPhase::kFinished, true),
          "worker completion latch terminates a finished conversation wait");
    check(voicechat::event_wait_is_terminal(voicechat::ConversationPhase::kCancelled, false),
          "cancellation terminates waits without worker input completion");
    check(
        !voicechat::event_wait_is_terminal(voicechat::ConversationPhase::kListening, true) &&
            !voicechat::event_wait_is_terminal(voicechat::ConversationPhase::kAgentSpeaking, true),
        "input completion does not terminate listening or speaking waits");
}

void test_async_epoch_gate_cancels_without_waiting_for_worker() {
    voicechat::AsyncEpochGate gate;
    const auto queued_epoch = gate.current();
    std::atomic<bool> worker_entered{false};
    std::atomic<std::uint64_t> observed_steps{0};
    std::thread worker([&] {
        worker_entered.store(true, std::memory_order_release);
        while (gate.accepts(queued_epoch))
            observed_steps.fetch_add(1, std::memory_order_relaxed);
    });
    while (!worker_entered.load(std::memory_order_acquire) ||
           observed_steps.load(std::memory_order_relaxed) == 0)
        std::this_thread::yield();

    const auto started = std::chrono::steady_clock::now();
    const auto replacement_epoch = gate.invalidate();
    const auto invalidation_elapsed = std::chrono::steady_clock::now() - started;
    worker.join();

    check(replacement_epoch != queued_epoch && !gate.accepts(queued_epoch) &&
              gate.accepts(replacement_epoch),
          "async epoch invalidation rejects in-flight work and accepts replacement work");
    check(invalidation_elapsed < std::chrono::milliseconds(10) && observed_steps.load() > 0,
          "cancel epoch advances without waiting for the worker");
}

void test_interruption_filter_preserves_completed_epochs() {
    std::vector<trtmc::SpeechSessionEvent> events(5);
    events[0].kind = trtmc::SpeechSessionEventKind::kAgentAudio;
    events[0].epoch = 3;
    events[1].kind = trtmc::SpeechSessionEventKind::kAgentText;
    events[1].epoch = 7;
    events[2].kind = trtmc::SpeechSessionEventKind::kAgentAudio;
    events[2].epoch = 7;
    events[3].kind = trtmc::SpeechSessionEventKind::kTurnStarted;
    events[3].epoch = 7;
    events[4].kind = trtmc::SpeechSessionEventKind::kUserTranscript;
    events[4].epoch = 7;

    events.erase(std::remove_if(events.begin(), events.end(),
                                [](const auto& event) {
                                    return event.epoch == 7 &&
                                           voicechat::is_agent_output_event(event.kind);
                                }),
                 events.end());
    check(events.size() == 3 && events[0].epoch == 3 &&
              events[0].kind == trtmc::SpeechSessionEventKind::kAgentAudio &&
              events[1].kind == trtmc::SpeechSessionEventKind::kTurnStarted &&
              events[2].kind == trtmc::SpeechSessionEventKind::kUserTranscript,
          "barge-in removes only interrupted agent payloads and preserves prior epochs");
}

void test_bounded_finish_tail_policy() {
    check(voicechat::resolve_finish_tail_frames(-1, 256) == 256,
          "live finish uses the model-owned response-frame bound");
    check(voicechat::resolve_finish_tail_frames(0, 256) == 0,
          "offline finish can flush without adding hidden tail frames");
    check(voicechat::resolve_finish_tail_frames(17, 256) == 17,
          "live callers can choose a smaller explicit tail bound");
}

} // namespace

int main() {
    test_frame_constants_and_chunk_accumulation();
    test_frame_finish_padding_and_reset();
    test_conversation_epoch_and_yield_contract();
    test_conversation_finish();
    test_wait_events_terminal_phase_policy();
    test_async_epoch_gate_cancels_without_waiting_for_worker();
    test_interruption_filter_preserves_completed_epochs();
    test_bounded_finish_tail_policy();
    if (failures > 0)
        std::cerr << failures << " VoiceChat session-state test(s) FAILED\n";
    return failures;
}

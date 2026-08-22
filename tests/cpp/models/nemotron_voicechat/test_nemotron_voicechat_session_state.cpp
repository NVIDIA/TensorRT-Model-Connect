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
#include <limits>
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

void test_frame_commit_and_clear_keep_session_open() {
    voicechat::FrameScheduler scheduler(4);
    std::vector<float> partial(320, 0.5F);
    scheduler.append(partial.data(), static_cast<int32_t>(partial.size()));
    scheduler.commit();
    const auto committed = scheduler.pop();
    check(committed.has_value() && committed->valid_input_samples == 320 && !committed->is_final &&
              !scheduler.finished(),
          "input commit exposes a padded model frame without finishing the session");

    scheduler.append(partial.data(), static_cast<int32_t>(partial.size()));
    scheduler.clear_pending();
    check(scheduler.pending_samples() == 0 && !scheduler.pop().has_value() &&
              scheduler.next_frame_index() == 1,
          "input clear drops only the pending fragment and preserves timeline position");

    std::vector<float> full(static_cast<std::size_t>(voicechat::kInputFrameSamples), 1.0F);
    scheduler.append(full.data(), static_cast<int32_t>(full.size()));
    const auto next = scheduler.pop();
    check(next.has_value() && next->frame_index == 1 && !next->is_final,
          "audio remains appendable after commit and clear");
}

void test_response_playback_timeline_uses_safe_boundaries() {
    voicechat::ResponsePlaybackTimeline timeline;
    timeline.begin(17);
    timeline.append_boundary(1920);
    timeline.append_boundary(3840);
    timeline.append_boundary(5760);

    check(timeline.retained_boundary_count(17, 0) == 0 &&
              timeline.retained_output_samples(17, 1919) == 0,
          "truncate before the first complete model frame retains no response audio");
    check(timeline.retained_boundary_count(17, 1920) == 1 &&
              timeline.retained_output_samples(17, 5759) == 3840,
          "millisecond playback cutoffs round down to the latest complete model frame");
    check(timeline.generated_output_samples() == 5760 &&
              timeline.retained_output_samples(17, 5760) == 5760,
          "the generated response boundary can be retained exactly");

    bool rejected = false;
    try {
        (void)timeline.retained_output_samples(16, 1920);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "response playback timeline rejects stale epochs");

    rejected = false;
    try {
        (void)timeline.retained_output_samples(17, 5761);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "response playback timeline rejects ungenerated playback positions");
    timeline.reset();
    check(!timeline.active(), "response playback reset removes the stale response epoch");
}

void test_realtime_turn_control_separates_commit_and_response_creation() {
    voicechat::RealtimeTurnControlState state;
    bool rejected = false;
    try {
        state.commit(false);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    check(rejected, "realtime turn control rejects an empty input commit");

    state.note_input();
    state.commit(false);
    check(!state.input_pending() && state.response_available(),
          "input commit preserves a response latch without starting generation");
    state.consume_response();
    check(!state.response_available(),
          "response creation consumes exactly one committed-turn latch");

    rejected = false;
    try {
        state.consume_response();
    } catch (const std::logic_error&) {
        rejected = true;
    }
    check(rejected, "response creation cannot reuse a consumed commit");

    state.restore_response();
    check(state.response_available(),
          "cancelled or truncated model state can create a replacement response");
    state.reset();
    check(!state.input_pending() && !state.response_available(),
          "realtime turn reset removes input and response latches");
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

void test_rnnt_turn_detector_rejects_noise_and_invalid_policy() {
    voicechat::RnntTurnPolicy invalid;
    invalid.end_of_utterance_blank_frames = 0;
    bool rejected = false;
    try {
        voicechat::RnntTurnDetector detector(invalid);
        (void)detector;
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "RNNT turn detector rejects non-positive thresholds");

    voicechat::RnntTurnPolicy policy;
    policy.first_utterance_min_speech_frames = 3;
    policy.subsequent_utterance_min_speech_frames = 4;
    policy.end_of_utterance_blank_frames = 2;
    policy.beginning_of_utterance_speech_frames = 3;
    voicechat::RnntTurnDetector detector(policy);

    check(!detector.observe(false, false, 0).speech_started &&
              !detector.observe(true, false, 1).speech_started &&
              !detector.observe(false, false, 2).speech_stopped &&
              !detector.observe(false, false, 3).speech_stopped,
          "blank, unknown, and short noise activity do not form an utterance");
    check(!detector.utterance_active() && detector.completed_utterances() == 0 &&
              detector.speech_frames() == 0,
          "EOU silence clears unconfirmed RNNT noise without consuming the first turn");

    rejected = false;
    try {
        (void)detector.observe(false, false, 3);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "RNNT turn observations require increasing frame indices");
}

void test_rnnt_first_and_subsequent_utterances() {
    voicechat::RnntTurnPolicy policy;
    policy.first_utterance_min_speech_frames = 2;
    policy.subsequent_utterance_min_speech_frames = 3;
    policy.end_of_utterance_blank_frames = 2;
    policy.beginning_of_utterance_speech_frames = 3;
    voicechat::RnntTurnDetector detector(policy);

    check(!detector.observe(true, false, 0).speech_started,
          "first RNNT speech frame waits for the first-turn minimum");
    const auto first_start = detector.observe(true, false, 1);
    check(first_start.speech_started && first_start.speech_start_frame == 0 &&
              !first_start.start_agent,
          "first-turn minimum emits one speech-start decision at the original frame");
    check(!detector.observe(false, false, 2).speech_stopped &&
              !detector.observe(true, false, 3).speech_started,
          "a mid-word pause shorter than EOU preserves the active utterance");
    check(!detector.observe(false, false, 4).speech_stopped,
          "EOU waits for the configured number of blank frames");
    const auto first_stop = detector.observe(false, false, 5);
    check(first_stop.speech_stopped && first_stop.start_agent &&
              first_stop.speech_start_frame == 0 && first_stop.speech_end_frame == 3 &&
              detector.completed_utterances() == 1,
          "first utterance finalizes once and requests an agent response");

    check(!detector.observe(true, false, 6).speech_started &&
              !detector.observe(false, false, 7).speech_stopped &&
              !detector.observe(true, false, 8).speech_started,
          "subsequent speech accumulates across a short pause without premature start");
    const auto second_start = detector.observe(true, false, 9);
    check(second_start.speech_started && second_start.speech_start_frame == 6,
          "subsequent utterances use their independent minimum speech threshold");
    const auto second_stop = detector.finalize_utterance(false, 9);
    check(second_stop.speech_stopped && second_stop.start_agent &&
              second_stop.speech_start_frame == 6 && second_stop.speech_end_frame == 9 &&
              detector.completed_utterances() == 2,
          "explicit utterance finalization flushes an active RNNT turn");
}

void test_rnnt_barge_in_and_reset() {
    voicechat::RnntTurnPolicy policy;
    policy.first_utterance_min_speech_frames = 2;
    policy.subsequent_utterance_min_speech_frames = 3;
    policy.end_of_utterance_blank_frames = 2;
    policy.beginning_of_utterance_speech_frames = 3;
    voicechat::RnntTurnDetector detector(policy);

    check(!detector.observe(true, true, 0).interrupt_agent &&
              !detector.observe(false, true, 1).interrupt_agent,
          "one speech token followed by silence is not enough to interrupt");
    const auto accumulated = detector.observe(true, true, 2);
    check(accumulated.speech_started && !accumulated.interrupt_agent &&
              accumulated.speech_start_frame == 0,
          "accumulated speech can confirm an utterance without bypassing consecutive BOU");
    check(!detector.observe(false, true, 3).interrupt_agent &&
              !detector.observe(true, true, 4).interrupt_agent &&
              !detector.observe(true, true, 5).interrupt_agent,
          "blank or unknown activity resets the consecutive BOU counter");
    const auto barge = detector.observe(true, true, 6);
    check(barge.interrupt_agent && barge.speech_start_frame == 0,
          "barge-in requires the configured consecutive non-unknown RNNT frames");
    check(!detector.observe(true, true, 7).interrupt_agent &&
              !detector.observe(true, true, 8).interrupt_agent,
          "one utterance cannot repeatedly interrupt the same agent turn");
    check(!detector.observe(false, false, 9).speech_stopped,
          "barge-in utterance remains live across a short blank");
    const auto stopped = detector.observe(false, false, 10);
    check(stopped.speech_stopped && stopped.start_agent && stopped.speech_end_frame == 8,
          "barge-in utterance starts a replacement response after EOU");

    detector.reset();
    check(!detector.utterance_active() && detector.completed_utterances() == 0 &&
              !detector.observe(true, false, 0).speech_started,
          "RNNT reset restores first-turn thresholds and frame numbering");
    const auto restarted = detector.observe(true, false, 1);
    check(restarted.speech_started && restarted.speech_start_frame == 0,
          "RNNT detector starts cleanly after reset");
    const auto finalized = detector.finalize_utterance(true, 1);
    check(finalized.speech_stopped && !finalized.start_agent,
          "finalization does not start a second agent while one is speaking");
}

void test_rnnt_single_frame_bou_policy() {
    voicechat::RnntTurnPolicy policy;
    policy.first_utterance_min_speech_frames = 4;
    policy.subsequent_utterance_min_speech_frames = 4;
    policy.end_of_utterance_blank_frames = 2;
    policy.beginning_of_utterance_speech_frames = 1;
    voicechat::RnntTurnDetector detector(policy);

    const auto barge = detector.observe(true, true, 0);
    check(barge.speech_started && barge.interrupt_agent && barge.speech_start_frame == 0,
          "one-frame BOU remains a valid low-latency model-aware barge policy");
}

void test_rnnt_bou_counts_only_agent_overlap() {
    voicechat::RnntTurnPolicy policy;
    policy.first_utterance_min_speech_frames = 2;
    policy.subsequent_utterance_min_speech_frames = 2;
    policy.end_of_utterance_blank_frames = 2;
    policy.beginning_of_utterance_speech_frames = 3;
    voicechat::RnntTurnDetector detector(policy);

    (void)detector.observe(true, false, 0);
    (void)detector.observe(true, false, 1);
    check(!detector.observe(true, true, 2).interrupt_agent &&
              !detector.observe(true, true, 3).interrupt_agent,
          "speech before the agent turn does not prefill the BOU overlap counter");
    check(detector.observe(true, true, 4).interrupt_agent,
          "BOU fires after the required consecutive speech frames overlap agent output");
}

void test_bounded_input_queue_budget() {
    bool rejected = false;
    try {
        voicechat::BoundedInputQueueBudget invalid(0);
        (void)invalid;
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "input queue budget rejects zero capacity");

    voicechat::BoundedInputQueueBudget budget(10);
    check(budget.reserve(6) && budget.queued_samples() == 6 && budget.available_samples() == 4,
          "input queue budget reserves samples exactly");
    check(budget.would_overflow(5) && !budget.reserve(5) && budget.queued_samples() == 6,
          "overflow leaves queued input accounting unchanged");
    check(budget.reserve(4) && budget.reserve(0) && budget.queued_samples() == 10 &&
              !budget.reserve(std::numeric_limits<std::size_t>::max()),
          "input budget accepts its exact boundary without integer overflow");
    budget.release(3);
    check(budget.queued_samples() == 7 && budget.reserve(3), "released input capacity is reusable");

    rejected = false;
    try {
        budget.release(11);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    check(rejected && budget.queued_samples() == 10,
          "input queue budget rejects release underflow without mutation");
    budget.reset();
    check(budget.queued_samples() == 0 && budget.available_samples() == 10,
          "input queue reset restores the full capacity");
}

void test_output_event_capacity_policy() {
    bool rejected = false;
    try {
        voicechat::OutputEventCapacityPolicy invalid(0, 1);
        (void)invalid;
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "output event policy rejects zero event capacity");

    voicechat::OutputEventCapacityPolicy policy(2, 4);
    trtmc::SpeechSessionEvent text;
    text.kind = trtmc::SpeechSessionEventKind::kAgentText;
    trtmc::SpeechSessionEvent audio;
    audio.kind = trtmc::SpeechSessionEventKind::kAgentAudio;
    audio.audio_samples = {0.1F, 0.2F, 0.3F, 0.4F};
    check(policy.accepts(0, 0, text) && policy.accepts(1, 0, audio),
          "output event policy accepts exact event and audio boundaries");
    check(!policy.accepts(2, 0, text) && !policy.accepts(0, 1, audio) &&
              !policy.accepts(0, 5, text),
          "output event policy detects count, addition, and invalid-current overflow");
    policy.validate(1, 0, audio);
    rejected = false;
    try {
        policy.validate(1, 1, audio);
    } catch (const std::length_error&) {
        rejected = true;
    }
    check(rejected, "output event validation reports capacity overflow");

    voicechat::OutputEventCapacityPolicy text_only(1, 0);
    check(text_only.accepts(0, 0, text) && !text_only.accepts(0, 0, audio),
          "zero audio capacity still permits control and text events");
}

} // namespace

int main() {
    test_frame_constants_and_chunk_accumulation();
    test_frame_finish_padding_and_reset();
    test_frame_commit_and_clear_keep_session_open();
    test_response_playback_timeline_uses_safe_boundaries();
    test_realtime_turn_control_separates_commit_and_response_creation();
    test_conversation_epoch_and_yield_contract();
    test_conversation_finish();
    test_wait_events_terminal_phase_policy();
    test_async_epoch_gate_cancels_without_waiting_for_worker();
    test_interruption_filter_preserves_completed_epochs();
    test_bounded_finish_tail_policy();
    test_rnnt_turn_detector_rejects_noise_and_invalid_policy();
    test_rnnt_first_and_subsequent_utterances();
    test_rnnt_barge_in_and_reset();
    test_rnnt_single_frame_bou_policy();
    test_rnnt_bou_counts_only_agent_overlap();
    test_bounded_input_queue_budget();
    test_output_event_capacity_policy();
    if (failures > 0)
        std::cerr << failures << " VoiceChat session-state test(s) FAILED\n";
    return failures;
}

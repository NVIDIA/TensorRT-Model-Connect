/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron_voicechat/runtime/session_state.h"
#include "families/nemotron_voicechat/runtime/thinker_hybrid_state.h"
#include "families/nemotron_voicechat/runtime/thinker_inference_state.h"
#include "families/nemotron_voicechat/runtime/thinker_kv_cache.h"
#include "families/nemotron_voicechat/runtime/thinker_mamba_state.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
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

    voicechat::FrameScheduler scheduler;
    std::vector<float> first(1000, 1.0F);
    std::vector<float> second(1560, 2.0F);
    scheduler.append(first.data(), static_cast<int32_t>(first.size()));
    check(!scheduler.pop().has_value(), "scheduler retains an incomplete input chunk");
    scheduler.append(second.data(), static_cast<int32_t>(second.size()));

    const auto frame0 = scheduler.pop();
    const auto frame1 = scheduler.pop();
    check(frame0.has_value() && frame1.has_value() && !scheduler.pop().has_value(),
          "scheduler emits two complete frames across chunk boundaries");
    check(frame0->samples[999] == 1.0F && frame0->samples[1000] == 2.0F &&
              frame1->samples.front() == 2.0F && frame1->samples.back() == 2.0F,
          "scheduler preserves sample order across chunk boundaries");
}

void test_frame_finish_padding_and_reset() {
    voicechat::FrameScheduler scheduler;
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

    scheduler.reset();
    scheduler.append(tail.data(), static_cast<int32_t>(tail.size()));
    check(scheduler.pending_samples() == tail.size() && !scheduler.pop().has_value(),
          "scheduler reset accepts a fresh incomplete chunk");
}

void test_frame_commit_and_clear_keep_session_open() {
    voicechat::FrameScheduler scheduler;
    std::vector<float> partial(320, 0.5F);
    scheduler.append(partial.data(), static_cast<int32_t>(partial.size()));
    scheduler.commit();
    const auto committed = scheduler.pop();
    check(committed.has_value() && committed->valid_input_samples == 320 && !committed->is_final,
          "input commit exposes a padded model frame without finishing the session");

    scheduler.append(partial.data(), static_cast<int32_t>(partial.size()));
    scheduler.clear_pending();
    check(scheduler.pending_samples() == 0 && !scheduler.pop().has_value(),
          "input clear drops only the pending fragment");

    std::vector<float> full(static_cast<std::size_t>(voicechat::kInputFrameSamples), 1.0F);
    scheduler.append(full.data(), static_cast<int32_t>(full.size()));
    const auto next = scheduler.pop();
    check(next.has_value() && next->samples.front() == 1.0F && !next->is_final,
          "audio remains appendable after commit and clear");
}

void test_response_checkpoints_use_safe_boundaries() {
    const std::vector<std::int64_t> checkpoints = {0, 1920, 3840, 5760};
    const auto retained = [&](std::int64_t played) {
        return voicechat::retained_response_checkpoint(checkpoints, played,
                                                       [](std::int64_t value) { return value; });
    };

    check(retained(0) == 0 && retained(1919) == 0,
          "truncate before the first complete model frame retains no response audio");
    check(retained(1920) == 1 && retained(5759) == 2,
          "millisecond playback cutoffs round down to the latest complete model frame");
    check(retained(5760) == 3, "the generated response boundary can be retained exactly");

    bool rejected = false;
    try {
        voicechat::validate_response_cursor(17, 16, 1920, checkpoints.back());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "response playback timeline rejects stale epochs");

    rejected = false;
    try {
        voicechat::validate_response_cursor(17, 17, 5761, checkpoints.back());
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "response playback timeline rejects ungenerated playback positions");
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
              !state.accepts_output(first_epoch),
          "barge-in rejects stale agent output by epoch");

    const auto second_epoch = state.begin_agent_turn();
    check(second_epoch != first_epoch && state.yield_to_user(),
          "model yield invalidates a later agent epoch");
    check(state.phase() == voicechat::ConversationPhase::kListening &&
              !state.accepts_output(second_epoch),
          "model yield returns to listening and rejects stale output");

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

enum class TestWork { kAudio, kCancel, kTruncate, kClear, kCreate };

bool is_test_priority(TestWork work) {
    return work != TestWork::kAudio;
}

void test_priority_controls_are_fifo_ahead_of_audio() {
    std::deque<TestWork> queue = {TestWork::kAudio,    TestWork::kCancel, TestWork::kAudio,
                                  TestWork::kTruncate, TestWork::kClear,  TestWork::kCreate};
    const auto first = voicechat::take_priority_fifo(queue, is_test_priority);
    const auto second = voicechat::take_priority_fifo(queue, is_test_priority);
    const auto third = voicechat::take_priority_fifo(queue, is_test_priority);
    const auto fourth = voicechat::take_priority_fifo(queue, is_test_priority);
    check(first == TestWork::kCancel && second == TestWork::kTruncate &&
              third == TestWork::kClear && fourth == TestWork::kCreate,
          "priority controls preserve FIFO order while bypassing queued audio");
    check(queue.size() == 2 && queue.front() == TestWork::kAudio &&
              queue.back() == TestWork::kAudio,
          "priority selection leaves audio order unchanged");
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

void test_bounded_pending_transcript_prefers_newest_distinct_text() {
    std::string pending;
    const bool accepted_first = voicechat::append_bounded_transcript(pending, "first request");
    const bool accepted_duplicate = voicechat::append_bounded_transcript(pending, "first request");
    check(accepted_first && !accepted_duplicate && pending == "first request",
          "bounded transcript reports and ignores an exact duplicate final transcript");

    voicechat::append_bounded_transcript(pending, "second request");
    check(pending == "first request / second request",
          "bounded transcript separates distinct finalized fragments");
    voicechat::append_bounded_transcript(pending, "second request");
    check(pending == "first request / second request",
          "bounded transcript ignores a duplicate newest fragment");

    voicechat::append_bounded_transcript(pending, "second request", 14);
    check(pending == "second request",
          "a smaller bound keeps the exact duplicate newest fragment without a broken separator");

    voicechat::append_bounded_transcript(pending, "third", 22);
    check(pending == "second request / third" && pending.size() == 22,
          "bounded transcript trims its oldest bytes and retains the newest text");

    voicechat::append_bounded_transcript(pending, "", 22);
    check(pending == "second request / third",
          "bounded transcript ignores an empty final transcript");

    voicechat::append_bounded_transcript(pending, "discarded", 0);
    check(pending.empty(), "zero transcript capacity retains no pending text");

    voicechat::append_bounded_transcript(pending, std::string(5000, 'n'));
    check(pending.size() == voicechat::kDefaultPendingTranscriptMaxBytes &&
              pending == std::string(voicechat::kDefaultPendingTranscriptMaxBytes, 'n'),
          "default transcript capacity bounds an oversized newest fragment");
}

void test_bounded_pending_transcript_trims_at_utf8_boundaries() {
    std::string pending;
    voicechat::append_bounded_transcript(pending, u8"甲乙丙丁", 7);
    check(pending == u8"丙丁" && pending.size() == 6,
          "oversized newest transcript keeps a valid UTF-8 suffix");

    voicechat::append_bounded_transcript(pending, u8"新", 9);
    check(pending == u8"丁 / 新" && pending.size() == 9,
          "combined transcript trimming preserves UTF-8 and the newest fragment");
}

std::vector<float> reference_linear_resample(const std::vector<float>& source, int source_rate,
                                             int target_rate) {
    const auto output_size = static_cast<std::size_t>(
        std::llround(static_cast<double>(source.size()) * target_rate / source_rate));
    std::vector<float> output;
    output.reserve(output_size);
    for (std::size_t index = 0; index < output_size; ++index) {
        const double position = static_cast<double>(index) * source_rate / target_rate;
        const auto left = std::min(static_cast<std::size_t>(position), source.size() - 1);
        const auto right = std::min(left + 1, source.size() - 1);
        const float fraction = static_cast<float>(position - static_cast<double>(left));
        output.push_back(source[left] + fraction * (source[right] - source[left]));
    }
    return output;
}

void test_streaming_resampler_preserves_phase_with_bounded_tail() {
    std::vector<float> source(640U * 5U);
    for (std::size_t index = 0; index < source.size(); ++index)
        source[index] = static_cast<float>((index * 17U) % 101U) / 101.0F;

    voicechat::StreamingLinearResampler resampler(8000, 16000);
    std::vector<float> streamed;
    for (std::size_t offset = 0; offset < source.size(); offset += 640U) {
        resampler.append(source.data() + offset, 640);
        auto next = resampler.drain(false);
        streamed.insert(streamed.end(), next.begin(), next.end());
        check(resampler.buffered_source_samples() <= 1,
              "upsampling retains only its next interpolation source sample");
    }
    auto tail = resampler.drain(true);
    streamed.insert(streamed.end(), tail.begin(), tail.end());
    const auto expected = reference_linear_resample(source, 8000, 16000);
    bool equal = streamed.size() == expected.size();
    for (std::size_t index = 0; equal && index < streamed.size(); ++index)
        equal = std::abs(streamed[index] - expected[index]) < 1.0e-6F;
    check(equal, "bounded 8-kHz streaming resampling matches one-shot phase and values");
    check(resampler.buffered_source_samples() == 0,
          "final resampler drain releases its interpolation tail");

    voicechat::StreamingLinearResampler identity(16000, 16000);
    identity.append(source.data(), static_cast<int32_t>(source.size()));
    check(identity.drain(false) == source && identity.buffered_source_samples() == 0,
          "identity streaming resampling releases source storage immediately");

    for (const int source_rate : {44100, 48000}) {
        for (const std::size_t chunk_size : {1U, 137U}) {
            voicechat::StreamingLinearResampler downsampler(source_rate, 16000);
            std::vector<float> downsampled;
            std::size_t max_buffered = 0;
            for (std::size_t offset = 0; offset < source.size(); offset += chunk_size) {
                const auto count = std::min(chunk_size, source.size() - offset);
                downsampler.append(source.data() + offset, static_cast<int32_t>(count));
                auto next = downsampler.drain(false);
                downsampled.insert(downsampled.end(), next.begin(), next.end());
                max_buffered = std::max(max_buffered, downsampler.buffered_source_samples());
            }
            auto final = downsampler.drain(true);
            downsampled.insert(downsampled.end(), final.begin(), final.end());
            const auto reference = reference_linear_resample(source, source_rate, 16000);
            bool downsample_equal = downsampled.size() == reference.size();
            for (std::size_t index = 0; downsample_equal && index < downsampled.size(); ++index)
                downsample_equal = std::abs(downsampled[index] - reference[index]) < 1.0e-6F;
            check(downsample_equal, "streaming downsampling matches final rounded one-shot output");
            check(max_buffered <= 4 && downsampler.buffered_source_samples() == 0,
                  "streaming downsampling retains only a bounded interpolation tail");
        }
    }
}

void test_rolling_cache_position_wraps_without_exhaustion() {
    const auto empty = voicechat::rolling_cache_position(0, 4);
    const auto partial = voicechat::rolling_cache_position(3, 4);
    const auto full = voicechat::rolling_cache_position(4, 4);
    const auto wrapped = voicechat::rolling_cache_position(9, 4);

    check(empty.valid_rows == 0 && empty.write_row == 0,
          "rolling cache starts empty at its first physical row");
    check(partial.valid_rows == 3 && partial.write_row == 3,
          "rolling cache appends sequentially before reaching capacity");
    check(full.valid_rows == 4 && full.write_row == 0,
          "rolling cache exposes every row and wraps at capacity");
    check(wrapped.valid_rows == 4 && wrapped.write_row == 1,
          "rolling cache keeps a full mask while logical positions continue");

    const auto pinned_partial = voicechat::rolling_cache_position(3, 8, 3);
    const auto pinned_last = voicechat::rolling_cache_position(7, 8, 3);
    const auto pinned_wrap = voicechat::rolling_cache_position(8, 8, 3);
    const auto pinned_next = voicechat::rolling_cache_position(9, 8, 3);
    const auto pinned_end = voicechat::rolling_cache_position(12, 8, 3);
    const auto pinned_again = voicechat::rolling_cache_position(13, 8, 3);
    check(pinned_partial.valid_rows == 3 && pinned_partial.write_row == 3 &&
              pinned_last.valid_rows == 7 && pinned_last.write_row == 7,
          "pinned rolling cache still appends sequentially before capacity");
    check(pinned_wrap.valid_rows == 8 && pinned_wrap.write_row == 3 && pinned_next.write_row == 4 &&
              pinned_end.write_row == 7 && pinned_again.write_row == 3,
          "pinned rolling cache wraps only within its unpinned suffix");
    bool prefix_preserved = true;
    for (std::int64_t position = 8; position < 80; ++position)
        prefix_preserved =
            prefix_preserved && voicechat::rolling_cache_position(position, 8, 3).write_row >= 3;
    check(prefix_preserved, "rolling cache never overwrites pinned conditioning rows");

    bool rejected = false;
    try {
        (void)voicechat::rolling_cache_position(-1, 4);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "rolling cache rejects negative logical positions");

    rejected = false;
    try {
        (void)voicechat::rolling_cache_position(0, 0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "rolling cache rejects non-positive capacity");

    rejected = false;
    try {
        (void)voicechat::rolling_cache_position(0, 4, -1);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "rolling cache rejects a negative pinned prefix");

    rejected = false;
    try {
        (void)voicechat::rolling_cache_position(0, 4, 4);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "rolling cache requires at least one rolling suffix row");
}

void test_tts_prompt_remains_pinned_across_compact_cache_wraps() {
    constexpr int32_t kCacheRows = 512;
    constexpr int32_t kPromptRows = 37;
    constexpr int32_t kHardSegmentFrames = 1375;
    constexpr int32_t kMaximumResponseFrames = 256;
    constexpr int32_t kMaximumLivePosition =
        kPromptRows + kHardSegmentFrames + kMaximumResponseFrames;
    static_assert(kMaximumLivePosition < 7500,
                  "live TTS rollover must precede the checkpoint's local-attention window");
    bool prefix_preserved = true;
    bool entire_suffix_used = true;
    std::vector<bool> suffix_rows(static_cast<std::size_t>(kCacheRows - kPromptRows), false);
    for (int32_t position = kCacheRows; position < kMaximumLivePosition; ++position) {
        const auto cache = voicechat::rolling_cache_position(position, kCacheRows, kPromptRows);
        prefix_preserved = prefix_preserved && cache.write_row >= kPromptRows;
        suffix_rows[static_cast<std::size_t>(cache.write_row - kPromptRows)] = true;
    }
    for (const bool used : suffix_rows)
        entire_suffix_used = entire_suffix_used && used;
    check(prefix_preserved,
          "compact EAR-TTS cache never overwrites the speaker-conditioning prompt");
    check(entire_suffix_used,
          "compact EAR-TTS cache rolls through every non-prompt row during a segment");

    // Model the logical position stored in every physical row and verify the
    // complete visible set at both sides of several ring boundaries. This
    // catches mappings that preserve the prefix but silently retain a stale
    // or non-contiguous generated suffix.
    constexpr std::array<int32_t, 8> kQueryPositions = {
        511, 512, 513, 986, 987, 1162, 1418, kMaximumLivePosition,
    };
    for (const int32_t query_position : kQueryPositions) {
        std::vector<int32_t> physical_rows(static_cast<std::size_t>(kCacheRows), -1);
        for (int32_t logical_position = 0; logical_position < query_position; ++logical_position) {
            const auto cache =
                voicechat::rolling_cache_position(logical_position, kCacheRows, kPromptRows);
            physical_rows[static_cast<std::size_t>(cache.write_row)] = logical_position;
        }

        std::vector<int32_t> visible;
        for (const int32_t logical_position : physical_rows) {
            if (logical_position >= 0)
                visible.push_back(logical_position);
        }
        std::sort(visible.begin(), visible.end());

        std::vector<int32_t> expected;
        if (query_position <= kCacheRows) {
            for (int32_t logical_position = 0; logical_position < query_position;
                 ++logical_position)
                expected.push_back(logical_position);
        } else {
            for (int32_t logical_position = 0; logical_position < kPromptRows; ++logical_position)
                expected.push_back(logical_position);
            const int32_t suffix_begin = query_position - (kCacheRows - kPromptRows);
            for (int32_t logical_position = suffix_begin; logical_position < query_position;
                 ++logical_position)
                expected.push_back(logical_position);
        }
        check(visible == expected,
              "compact EAR-TTS cache exposes the prompt and exact newest generated suffix");
    }
}

bool observe_tokens(voicechat::RepetitionWatchdog& watchdog, const std::vector<int32_t>& tokens) {
    bool tripped = false;
    for (const int32_t token : tokens)
        tripped = watchdog.observe(token);
    return tripped;
}

std::vector<int32_t> repeat_block(const std::vector<int32_t>& block, int repetitions) {
    std::vector<int32_t> tokens;
    tokens.reserve(block.size() * static_cast<std::size_t>(repetitions));
    for (int repetition = 0; repetition < repetitions; ++repetition)
        tokens.insert(tokens.end(), block.begin(), block.end());
    return tokens;
}

void test_repetition_watchdog_thresholds_and_reset() {
    voicechat::RepetitionWatchdog watchdog;

    check(!observe_tokens(watchdog, std::vector<int32_t>(7, 41)) && watchdog.observe(41) &&
              watchdog.tripped(),
          "repetition watchdog detects one token repeated eight times");
    check(watchdog.observe(99), "repetition watchdog remains tripped until reset");

    watchdog.reset();
    check(!watchdog.tripped() && !observe_tokens(watchdog, std::vector<int32_t>(7, 41)),
          "repetition watchdog reset clears its latch and token history");

    watchdog.reset();
    const std::vector<int32_t> three_token_block = {1, 2, 3};
    check(!observe_tokens(watchdog, repeat_block(three_token_block, 2)) &&
              observe_tokens(watchdog, three_token_block),
          "repetition watchdog detects a three-token block repeated three times");

    watchdog.reset();
    const std::vector<int32_t> seven_token_block = {11, 12, 13, 14, 15, 16, 17};
    check(!observe_tokens(watchdog, repeat_block(seven_token_block, 2)) &&
              observe_tokens(watchdog, seven_token_block),
          "repetition watchdog detects a seven-token block repeated three times");

    watchdog.reset();
    const std::vector<int32_t> eight_token_block = {21, 22, 23, 24, 25, 26, 27, 28};
    check(!observe_tokens(watchdog, eight_token_block) &&
              observe_tokens(watchdog, eight_token_block),
          "repetition watchdog detects an eight-token block repeated twice");

    watchdog.reset();
    std::vector<int32_t> forty_eight_token_block(48);
    for (std::size_t index = 0; index < forty_eight_token_block.size(); ++index)
        forty_eight_token_block[index] = 1000 + static_cast<int32_t>(index);
    check(!observe_tokens(watchdog, forty_eight_token_block) &&
              observe_tokens(watchdog, forty_eight_token_block),
          "repetition watchdog detects a forty-eight-token block repeated twice");
}

void test_repetition_watchdog_ignores_near_misses() {
    voicechat::RepetitionWatchdog watchdog;
    check(!observe_tokens(watchdog, {1, 2, 1, 2, 1, 2}),
          "repetition watchdog ignores short two-token cycles below its long-block threshold");

    watchdog.reset();
    check(!observe_tokens(watchdog, {3, 4, 5, 3, 4, 5, 3, 4, 6}),
          "repetition watchdog requires exact equality in a repeated block");

    watchdog.reset();
    std::vector<int32_t> unique_tokens(200);
    for (std::size_t index = 0; index < unique_tokens.size(); ++index)
        unique_tokens[index] = static_cast<int32_t>(index);
    check(!observe_tokens(watchdog, unique_tokens),
          "repetition watchdog permits long non-repeating output with bounded history");
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

void test_rnnt_turn_detector_reports_expired_subthreshold_candidate_once() {
    voicechat::RnntTurnPolicy policy;
    policy.first_utterance_min_speech_frames = 3;
    policy.subsequent_utterance_min_speech_frames = 4;
    policy.end_of_utterance_blank_frames = 2;
    policy.beginning_of_utterance_speech_frames = 3;
    voicechat::RnntTurnDetector detector(policy);

    const auto initial_blank = detector.observe(false, false, 0);
    check(!initial_blank.discarded_candidate,
          "ordinary RNNT silence does not report a discarded candidate");

    const auto subthreshold = detector.observe(true, false, 1);
    const auto unknown_or_blank = detector.observe(false, false, 2);
    const auto expired = detector.observe(false, false, 3);
    check(!subthreshold.speech_started && !unknown_or_blank.discarded_candidate &&
              expired.discarded_candidate && !expired.speech_started && !expired.speech_stopped &&
              !expired.start_agent && !expired.interrupt_agent,
          "EOU reports a subthreshold or unknown-interrupted RNNT candidate for discard");
    check(!detector.utterance_active() && detector.speech_frames() == 0 &&
              detector.completed_utterances() == 0,
          "discarding a candidate clears noise without consuming an utterance");

    const auto following_blank = detector.observe(false, false, 4);
    check(!following_blank.discarded_candidate,
          "an expired RNNT candidate emits its discard signal exactly once");

    (void)detector.observe(true, false, 5);
    (void)detector.observe(true, false, 6);
    const auto confirmed_start = detector.observe(true, false, 7);
    (void)detector.observe(false, false, 8);
    const auto confirmed_stop = detector.observe(false, false, 9);
    check(confirmed_start.speech_started && confirmed_stop.speech_stopped &&
              confirmed_stop.start_agent && !confirmed_stop.discarded_candidate,
          "a confirmed RNNT utterance preserves normal start and stop behavior");
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

void test_rnnt_stream_frontier_reset_preserves_conversation_threshold() {
    voicechat::RnntTurnPolicy policy;
    policy.first_utterance_min_speech_frames = 1;
    policy.subsequent_utterance_min_speech_frames = 3;
    policy.end_of_utterance_blank_frames = 1;
    voicechat::RnntTurnDetector detector(policy);

    check(detector.observe(true, false, 0).speech_started &&
              detector.observe(false, false, 1).speech_stopped &&
              detector.completed_utterances() == 1,
          "RNNT test establishes a completed first conversation utterance");
    detector.reset_stream_frontier();
    check(detector.completed_utterances() == 1 &&
              !detector.observe(true, false, 50).speech_started &&
              !detector.observe(true, false, 51).speech_started &&
              detector.observe(true, false, 52).speech_started,
          "transparent frontier reset retains the subsequent-turn threshold");
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

} // namespace

int main() {
    test_frame_constants_and_chunk_accumulation();
    test_frame_finish_padding_and_reset();
    test_frame_commit_and_clear_keep_session_open();
    test_response_checkpoints_use_safe_boundaries();
    test_realtime_turn_control_separates_commit_and_response_creation();
    test_conversation_epoch_and_yield_contract();
    test_conversation_finish();
    test_wait_events_terminal_phase_policy();
    test_async_epoch_gate_cancels_without_waiting_for_worker();
    test_priority_controls_are_fifo_ahead_of_audio();
    test_interruption_filter_preserves_completed_epochs();
    test_bounded_finish_tail_policy();
    test_bounded_pending_transcript_prefers_newest_distinct_text();
    test_bounded_pending_transcript_trims_at_utf8_boundaries();
    test_streaming_resampler_preserves_phase_with_bounded_tail();
    test_rolling_cache_position_wraps_without_exhaustion();
    test_tts_prompt_remains_pinned_across_compact_cache_wraps();
    test_repetition_watchdog_thresholds_and_reset();
    test_repetition_watchdog_ignores_near_misses();
    test_rnnt_turn_detector_rejects_noise_and_invalid_policy();
    test_rnnt_turn_detector_reports_expired_subthreshold_candidate_once();
    test_rnnt_first_and_subsequent_utterances();
    test_rnnt_stream_frontier_reset_preserves_conversation_threshold();
    test_rnnt_barge_in_and_reset();
    test_rnnt_single_frame_bou_policy();
    test_rnnt_bou_counts_only_agent_overlap();
    if (failures > 0)
        std::cerr << failures << " VoiceChat session-state test(s) FAILED\n";
    return failures;
}

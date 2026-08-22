/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc::nemotron_voicechat {

inline constexpr double kFrameDurationSeconds = 0.08;
inline constexpr int32_t kInputSampleRate = 16000;
inline constexpr int32_t kInputFrameSamples = 1280;

static_assert(kInputFrameSamples == static_cast<int32_t>(kInputSampleRate * kFrameDurationSeconds));

// Lock-free invalidation token shared by the public control plane and the
// inference worker. A worker tags every queued item and checks the tag between
// TensorRT stages, so cancel/barge-in never waits for the GPU and stale work
// cannot reach TTS, codec, or the public event queue.
class AsyncEpochGate {
  public:
    std::uint64_t current() const { return epoch_.load(std::memory_order_acquire); }
    bool accepts(std::uint64_t epoch) const { return current() == epoch; }
    std::uint64_t invalidate();

  private:
    std::atomic<std::uint64_t> epoch_{1};
};

// Priority controls are appended under the session mutex. Selecting the first
// matching entry gives them stable FIFO order without a second queue.
template <typename Work, typename Predicate>
std::optional<Work> take_priority_fifo(std::deque<Work>& queue, Predicate is_priority) {
    const auto selected = std::find_if(queue.begin(), queue.end(), is_priority);
    if (selected == queue.end())
        return std::nullopt;
    Work work = std::move(*selected);
    queue.erase(selected);
    return work;
}

bool is_agent_output_event(SpeechSessionEventKind kind);
int32_t resolve_finish_tail_frames(int32_t requested_frames, int32_t model_max_frames);

// Host-only RNNT turn-taking policy. One observation represents one 80 ms
// VoiceChat frame after blank and unknown tokens have been filtered out.
struct RnntTurnPolicy {
    int32_t first_utterance_min_speech_frames{2};
    int32_t subsequent_utterance_min_speech_frames{3};
    int32_t end_of_utterance_blank_frames{15};
    int32_t beginning_of_utterance_speech_frames{3};
};

struct RnntTurnDecision {
    bool speech_started{false};
    bool speech_stopped{false};
    bool start_agent{false};
    bool interrupt_agent{false};
    std::int64_t speech_start_frame{-1};
    std::int64_t speech_end_frame{-1};
};

// Debounces RNNT activity into user utterances. The detector owns no locks and
// performs no I/O; the session worker supplies observations in frame order and
// applies the returned actions to its conversation state.
class RnntTurnDetector {
  public:
    explicit RnntTurnDetector(RnntTurnPolicy policy = {});

    RnntTurnDecision observe(bool has_speech_token, bool agent_speaking, std::int64_t frame_index);
    RnntTurnDecision finalize_utterance(bool agent_speaking, std::int64_t frame_index);
    void reset();

    bool utterance_active() const { return utterance_active_; }
    std::uint64_t completed_utterances() const { return completed_utterances_; }
    int32_t speech_frames() const { return speech_frames_; }

  private:
    int32_t minimum_speech_frames() const;
    void validate_observation_frame(std::int64_t frame_index) const;
    RnntTurnDecision stop_utterance(bool agent_speaking);
    void clear_utterance();

    RnntTurnPolicy policy_;
    std::int64_t last_frame_index_{-1};
    std::int64_t candidate_start_frame_{-1};
    std::int64_t utterance_start_frame_{-1};
    std::int64_t last_speech_frame_{-1};
    std::uint64_t completed_utterances_{0};
    int32_t speech_frames_{0};
    int32_t blank_frames_{0};
    int32_t consecutive_bou_speech_frames_{0};
    bool utterance_active_{false};
    bool interrupt_requested_{false};
};

// One frame on VoiceChat's shared 12.5 Hz perception/text/TTS timeline.
struct ScheduledInputFrame {
    std::array<float, static_cast<std::size_t>(kInputFrameSamples)> samples{};
    int32_t valid_input_samples{0};
    bool is_final{false};
};

// Collects resampled 16 kHz input across arbitrary public API chunk boundaries
// and releases complete 80 ms model frames. finish() exposes one zero-padded
// final frame when the stream ends between frame boundaries.
class FrameScheduler {
  public:
    void append(const float* samples, int32_t num_samples);
    void commit();
    void clear_pending();
    void finish();
    std::optional<ScheduledInputFrame> pop();
    void reset();

    std::size_t pending_samples() const;

  private:
    void compact();

    std::vector<float> samples_;
    std::size_t read_offset_{0};
    bool finished_{false};
    bool commit_pending_{false};
};

void validate_response_cursor(std::uint64_t active_epoch, std::uint64_t requested_epoch,
                              std::int64_t played_output_samples,
                              std::int64_t generated_output_samples);

// VoiceChat checkpoints include the zero-sample response start. Conservatively
// retain only the latest complete model frame at or before the playback cursor.
template <typename Checkpoints, typename EndSample>
std::size_t retained_response_checkpoint(const Checkpoints& checkpoints,
                                         std::int64_t played_output_samples, EndSample end_sample) {
    if (checkpoints.empty() || end_sample(checkpoints.front()) != 0)
        throw std::logic_error("VoiceChat response checkpoints must start at zero");
    if (played_output_samples < 0 || played_output_samples > end_sample(checkpoints.back()))
        throw std::invalid_argument("VoiceChat playback cursor is outside the generated response");
    std::size_t retained = 0;
    while (retained + 1 < checkpoints.size() &&
           end_sample(checkpoints[retained + 1]) <= played_output_samples)
        ++retained;
    return retained;
}

// Transport-level input/response latch. Committing input and creating a
// response are distinct operations; a cancelled or truncated response may be
// created again without pretending that a second input commit occurred.
class RealtimeTurnControlState {
  public:
    void note_input() noexcept { input_pending_ = true; }
    void commit(bool model_observed_turn);
    void consume_response();
    void restore_response() noexcept { response_available_ = true; }
    void reset() noexcept;

    bool input_pending() const { return input_pending_; }
    bool response_available() const { return response_available_; }

  private:
    bool input_pending_{false};
    bool response_available_{false};
};

enum class ConversationPhase {
    kListening,
    kAgentSpeaking,
    kFinished,
    kCancelled,
};

bool event_wait_is_terminal(ConversationPhase phase, bool input_work_completed) noexcept;

// Host-owned conversation lifecycle. Long-running inference work captures the
// epoch returned by begin_agent_turn(); results are publishable only while
// accepts_output(epoch) remains true. Barge-in, yield, cancel, reset, and turn
// completion synchronously invalidate stale queued GPU/decoder output.
class ConversationState {
  public:
    std::uint64_t begin_agent_turn();
    std::uint64_t finish_agent_turn();
    bool barge_in();
    bool yield_to_user();
    void finish_input();
    void cancel();
    void reset();

    bool accepts_output(std::uint64_t output_epoch) const;
    bool can_accept_audio() const {
        return !input_finished_ && phase_ != ConversationPhase::kCancelled;
    }
    std::uint64_t next_sequence() { return next_sequence_++; }

    std::uint64_t epoch() const { return epoch_; }
    ConversationPhase phase() const { return phase_; }

  private:
    void advance_epoch();
    bool invalidate_for_yield();

    std::uint64_t epoch_{1};
    std::uint64_t next_sequence_{0};
    ConversationPhase phase_{ConversationPhase::kListening};
    bool input_finished_{false};
};

} // namespace trtmc::nemotron_voicechat

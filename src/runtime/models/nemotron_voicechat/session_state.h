/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace trtmc::nemotron_voicechat {

inline constexpr double kFrameDurationSeconds = 0.08;
inline constexpr int32_t kInputSampleRate = 16000;
inline constexpr int32_t kOutputSampleRate = 22050;
inline constexpr int32_t kInputFrameSamples = 1280;
inline constexpr int32_t kOutputFrameSamples = 1764;

static_assert(kInputFrameSamples == static_cast<int32_t>(kInputSampleRate * kFrameDurationSeconds));
static_assert(kOutputFrameSamples ==
              static_cast<int32_t>(kOutputSampleRate * kFrameDurationSeconds));

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

bool is_agent_output_event(SpeechSessionEventKind kind);
int32_t resolve_finish_tail_frames(int32_t requested_frames, int32_t model_max_frames);

// One frame on VoiceChat's shared 12.5 Hz perception/text/TTS timeline.
struct ScheduledInputFrame {
    std::array<float, static_cast<std::size_t>(kInputFrameSamples)> samples{};
    std::uint64_t epoch{0};
    std::int64_t frame_index{0};
    std::int64_t input_start_sample{0};
    std::int64_t input_end_sample{0};
    std::int64_t output_start_sample{0};
    std::int64_t output_end_sample{0};
    int32_t valid_input_samples{0};
    bool is_final{false};
};

// Collects resampled 16 kHz input across arbitrary public API chunk boundaries
// and releases complete 80 ms model frames. finish() exposes one zero-padded
// final frame when the stream ends between frame boundaries.
class FrameScheduler {
  public:
    explicit FrameScheduler(std::uint64_t epoch = 1) : epoch_(epoch) {}

    void append(const float* samples, int32_t num_samples);
    void finish();
    std::optional<ScheduledInputFrame> pop();
    void reset(std::uint64_t epoch);

    std::size_t pending_samples() const;
    std::int64_t next_frame_index() const { return next_frame_index_; }
    std::uint64_t epoch() const { return epoch_; }
    bool finished() const { return finished_; }

  private:
    void compact();

    std::vector<float> samples_;
    std::size_t read_offset_{0};
    std::uint64_t epoch_{1};
    std::int64_t next_frame_index_{0};
    bool finished_{false};
};

enum class ConversationPhase {
    kListening,
    kAgentSpeaking,
    kFinished,
    kCancelled,
};

bool event_wait_is_terminal(ConversationPhase phase, bool input_work_completed) noexcept;

enum class YieldReason {
    kNone,
    kBargeIn,
    kModelYield,
};

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
    YieldReason yield_reason() const { return yield_reason_; }
    bool input_finished() const { return input_finished_; }

  private:
    void advance_epoch();
    bool invalidate_for_yield(YieldReason reason);

    std::uint64_t epoch_{1};
    std::uint64_t next_sequence_{0};
    ConversationPhase phase_{ConversationPhase::kListening};
    YieldReason yield_reason_{YieldReason::kNone};
    bool input_finished_{false};
};

} // namespace trtmc::nemotron_voicechat

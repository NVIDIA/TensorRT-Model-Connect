/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/nemotron_voicechat/session_state.h"

#include <algorithm>
#include <stdexcept>

namespace trtmc::nemotron_voicechat {

std::uint64_t AsyncEpochGate::invalidate() {
    // uint64 wrap would require centuries even at GHz invalidation rates. Keep
    // zero reserved so default-initialized work can never become valid.
    auto next = epoch_.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (next == 0) {
        std::uint64_t expected = 0;
        (void)epoch_.compare_exchange_strong(expected, 1, std::memory_order_acq_rel);
        next = current();
    }
    return next;
}

bool event_wait_is_terminal(ConversationPhase phase, bool input_work_completed) noexcept {
    return phase == ConversationPhase::kCancelled ||
           (input_work_completed && phase == ConversationPhase::kFinished);
}

bool is_agent_output_event(SpeechSessionEventKind kind) {
    return kind == SpeechSessionEventKind::kAgentAudio ||
           kind == SpeechSessionEventKind::kAgentText;
}

int32_t resolve_finish_tail_frames(int32_t requested_frames, int32_t model_max_frames) {
    if (requested_frames < -1)
        throw std::invalid_argument("VoiceChat finish tail must be -1 or non-negative");
    if (model_max_frames < 0)
        throw std::invalid_argument("VoiceChat model response bound must be non-negative");
    return requested_frames < 0 ? model_max_frames : requested_frames;
}

void FrameScheduler::append(const float* samples, int32_t num_samples) {
    if (finished_)
        throw std::logic_error("VoiceChat input is already finished");
    if (num_samples < 0 || (num_samples > 0 && samples == nullptr))
        throw std::invalid_argument("VoiceChat audio chunk must have valid mono samples");
    if (num_samples == 0)
        return;
    samples_.insert(samples_.end(), samples, samples + num_samples);
}

void FrameScheduler::finish() {
    finished_ = true;
}

std::optional<ScheduledInputFrame> FrameScheduler::pop() {
    const std::size_t available = pending_samples();
    if (available < static_cast<std::size_t>(kInputFrameSamples) && (!finished_ || available == 0))
        return std::nullopt;

    ScheduledInputFrame frame;
    frame.epoch = epoch_;
    frame.frame_index = next_frame_index_;
    frame.input_start_sample = next_frame_index_ * kInputFrameSamples;
    frame.output_start_sample = next_frame_index_ * kOutputFrameSamples;
    frame.output_end_sample = frame.output_start_sample + kOutputFrameSamples;

    const std::size_t consumed = std::min(available, static_cast<std::size_t>(kInputFrameSamples));
    frame.valid_input_samples = static_cast<int32_t>(consumed);
    frame.input_end_sample = frame.input_start_sample + frame.valid_input_samples;
    frame.is_final = finished_ && available <= static_cast<std::size_t>(kInputFrameSamples);
    std::copy_n(samples_.data() + read_offset_, consumed, frame.samples.data());

    read_offset_ += consumed;
    ++next_frame_index_;
    compact();
    return frame;
}

void FrameScheduler::reset(std::uint64_t epoch) {
    samples_.clear();
    read_offset_ = 0;
    epoch_ = epoch;
    next_frame_index_ = 0;
    finished_ = false;
}

std::size_t FrameScheduler::pending_samples() const {
    return samples_.size() - read_offset_;
}

void FrameScheduler::compact() {
    if (read_offset_ == samples_.size()) {
        samples_.clear();
        read_offset_ = 0;
        return;
    }
    if (read_offset_ >= static_cast<std::size_t>(kInputFrameSamples) * 4U) {
        samples_.erase(samples_.begin(),
                       samples_.begin() + static_cast<std::ptrdiff_t>(read_offset_));
        read_offset_ = 0;
    }
}

void ConversationState::advance_epoch() {
    ++epoch_;
    if (epoch_ == 0)
        epoch_ = 1;
    next_sequence_ = 0;
}

std::uint64_t ConversationState::begin_agent_turn() {
    if (phase_ == ConversationPhase::kCancelled)
        throw std::logic_error("VoiceChat conversation is cancelled; reset it before reuse");
    if (phase_ == ConversationPhase::kAgentSpeaking)
        throw std::logic_error("VoiceChat agent turn is already active");
    advance_epoch();
    phase_ = ConversationPhase::kAgentSpeaking;
    yield_reason_ = YieldReason::kNone;
    return epoch_;
}

std::uint64_t ConversationState::finish_agent_turn() {
    if (phase_ != ConversationPhase::kAgentSpeaking)
        throw std::logic_error("VoiceChat has no active agent turn to finish");
    const std::uint64_t completed_epoch = epoch_;
    advance_epoch();
    phase_ = input_finished_ ? ConversationPhase::kFinished : ConversationPhase::kListening;
    yield_reason_ = YieldReason::kNone;
    return completed_epoch;
}

bool ConversationState::invalidate_for_yield(YieldReason reason) {
    if (phase_ != ConversationPhase::kAgentSpeaking)
        return false;
    advance_epoch();
    phase_ = input_finished_ ? ConversationPhase::kFinished : ConversationPhase::kListening;
    yield_reason_ = reason;
    return true;
}

bool ConversationState::barge_in() {
    return invalidate_for_yield(YieldReason::kBargeIn);
}

bool ConversationState::yield_to_user() {
    return invalidate_for_yield(YieldReason::kModelYield);
}

void ConversationState::finish_input() {
    if (phase_ == ConversationPhase::kCancelled)
        return;
    input_finished_ = true;
    if (phase_ == ConversationPhase::kListening)
        phase_ = ConversationPhase::kFinished;
}

void ConversationState::cancel() {
    advance_epoch();
    phase_ = ConversationPhase::kCancelled;
    yield_reason_ = YieldReason::kNone;
    input_finished_ = true;
}

void ConversationState::reset() {
    advance_epoch();
    phase_ = ConversationPhase::kListening;
    yield_reason_ = YieldReason::kNone;
    input_finished_ = false;
}

bool ConversationState::accepts_output(std::uint64_t output_epoch) const {
    return phase_ == ConversationPhase::kAgentSpeaking && output_epoch == epoch_;
}

} // namespace trtmc::nemotron_voicechat

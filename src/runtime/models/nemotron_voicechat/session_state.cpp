/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/nemotron_voicechat/session_state.h"

#include <algorithm>
#include <stdexcept>

namespace trtmc::nemotron_voicechat {

namespace {

bool barge_in_is_confirmed(bool agent_speaking, int32_t consecutive_speech_frames,
                           int32_t required_speech_frames) {
    return agent_speaking && consecutive_speech_frames >= required_speech_frames;
}

} // namespace

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
           kind == SpeechSessionEventKind::kAgentText ||
           kind == SpeechSessionEventKind::kFunctionCall ||
           kind == SpeechSessionEventKind::kFunctionCallStarted ||
           kind == SpeechSessionEventKind::kFunctionResponseFinished;
}

int32_t resolve_finish_tail_frames(int32_t requested_frames, int32_t model_max_frames) {
    if (requested_frames < -1)
        throw std::invalid_argument("VoiceChat finish tail must be -1 or non-negative");
    if (model_max_frames < 0)
        throw std::invalid_argument("VoiceChat model response bound must be non-negative");
    return requested_frames < 0 ? model_max_frames : requested_frames;
}

RnntTurnDetector::RnntTurnDetector(RnntTurnPolicy policy) : policy_(policy) {
    if (policy_.first_utterance_min_speech_frames <= 0 ||
        policy_.subsequent_utterance_min_speech_frames <= 0 ||
        policy_.end_of_utterance_blank_frames <= 0 ||
        policy_.beginning_of_utterance_speech_frames <= 0) {
        throw std::invalid_argument("VoiceChat RNNT turn thresholds must be positive");
    }
}

int32_t RnntTurnDetector::minimum_speech_frames() const {
    return completed_utterances_ == 0 ? policy_.first_utterance_min_speech_frames
                                      : policy_.subsequent_utterance_min_speech_frames;
}

void RnntTurnDetector::validate_observation_frame(std::int64_t frame_index) const {
    if (frame_index < 0 || frame_index <= last_frame_index_)
        throw std::invalid_argument("VoiceChat RNNT frame indices must increase");
}

void RnntTurnDetector::clear_utterance() {
    candidate_start_frame_ = -1;
    utterance_start_frame_ = -1;
    last_speech_frame_ = -1;
    speech_frames_ = 0;
    blank_frames_ = 0;
    consecutive_bou_speech_frames_ = 0;
    utterance_active_ = false;
    interrupt_requested_ = false;
}

RnntTurnDecision RnntTurnDetector::stop_utterance(bool agent_speaking) {
    RnntTurnDecision decision;
    if (!utterance_active_) {
        clear_utterance();
        return decision;
    }

    const bool response_ready = speech_frames_ >= minimum_speech_frames();
    decision.speech_stopped = true;
    decision.start_agent = response_ready && !agent_speaking;
    decision.speech_start_frame = utterance_start_frame_;
    decision.speech_end_frame = last_speech_frame_;
    if (response_ready)
        ++completed_utterances_;
    clear_utterance();
    return decision;
}

RnntTurnDecision RnntTurnDetector::observe(bool has_speech_token, bool agent_speaking,
                                           std::int64_t frame_index) {
    validate_observation_frame(frame_index);
    last_frame_index_ = frame_index;

    if (!has_speech_token) {
        consecutive_bou_speech_frames_ = 0;
        ++blank_frames_;
        if (blank_frames_ >= policy_.end_of_utterance_blank_frames)
            return stop_utterance(agent_speaking);
        return {};
    }

    if (candidate_start_frame_ < 0)
        candidate_start_frame_ = frame_index;
    last_speech_frame_ = frame_index;
    blank_frames_ = 0;
    ++speech_frames_;
    if (agent_speaking)
        ++consecutive_bou_speech_frames_;
    else
        consecutive_bou_speech_frames_ = 0;

    RnntTurnDecision decision;
    const bool speech_confirmed = speech_frames_ >= minimum_speech_frames();
    const bool barge_in_confirmed =
        barge_in_is_confirmed(agent_speaking, consecutive_bou_speech_frames_,
                              policy_.beginning_of_utterance_speech_frames);
    if (!utterance_active_ && (speech_confirmed || barge_in_confirmed)) {
        utterance_active_ = true;
        utterance_start_frame_ = candidate_start_frame_;
        decision.speech_started = true;
        decision.speech_start_frame = utterance_start_frame_;
    }
    if (barge_in_confirmed && !interrupt_requested_) {
        interrupt_requested_ = true;
        decision.interrupt_agent = true;
        decision.speech_start_frame = utterance_start_frame_;
    }
    return decision;
}

RnntTurnDecision RnntTurnDetector::finalize_utterance(bool agent_speaking,
                                                      std::int64_t frame_index) {
    if (frame_index < 0 || frame_index < last_frame_index_)
        throw std::invalid_argument("VoiceChat RNNT final frame cannot move backwards");
    last_frame_index_ = frame_index;
    return stop_utterance(agent_speaking);
}

void RnntTurnDetector::reset() {
    completed_utterances_ = 0;
    last_frame_index_ = -1;
    clear_utterance();
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

void FrameScheduler::commit() {
    if (finished_)
        throw std::logic_error("VoiceChat input is already finished");
    commit_pending_ = pending_samples() != 0;
}

void FrameScheduler::clear_pending() {
    if (finished_)
        throw std::logic_error("VoiceChat input is already finished");
    samples_.resize(read_offset_);
    compact();
    commit_pending_ = false;
}

void FrameScheduler::finish() {
    finished_ = true;
    commit_pending_ = false;
}

std::optional<ScheduledInputFrame> FrameScheduler::pop() {
    const std::size_t available = pending_samples();
    const bool flush_partial = (finished_ || commit_pending_) && available != 0;
    if (available < static_cast<std::size_t>(kInputFrameSamples) && !flush_partial)
        return std::nullopt;

    ScheduledInputFrame frame;
    const std::size_t consumed = std::min(available, static_cast<std::size_t>(kInputFrameSamples));
    frame.valid_input_samples = static_cast<int32_t>(consumed);
    frame.is_final = finished_ && available <= static_cast<std::size_t>(kInputFrameSamples);
    std::copy_n(samples_.data() + read_offset_, consumed, frame.samples.data());

    read_offset_ += consumed;
    if (commit_pending_ && available <= static_cast<std::size_t>(kInputFrameSamples))
        commit_pending_ = false;
    compact();
    return frame;
}

void FrameScheduler::reset() {
    samples_.clear();
    read_offset_ = 0;
    finished_ = false;
    commit_pending_ = false;
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

void validate_response_cursor(std::uint64_t active_epoch, std::uint64_t requested_epoch,
                              std::int64_t played_output_samples,
                              std::int64_t generated_output_samples) {
    if (active_epoch == 0 || requested_epoch != active_epoch)
        throw std::invalid_argument("VoiceChat response epoch is stale");
    if (played_output_samples < 0)
        throw std::invalid_argument("VoiceChat played output samples must be non-negative");
    if (played_output_samples > generated_output_samples)
        throw std::invalid_argument("VoiceChat cannot truncate beyond generated response audio");
}

void RealtimeTurnControlState::commit(bool model_observed_turn) {
    if (!input_pending_ && !model_observed_turn)
        throw std::logic_error("VoiceChat cannot commit an empty input turn");
    input_pending_ = false;
    response_available_ = true;
}

void RealtimeTurnControlState::consume_response() {
    if (!response_available_)
        throw std::logic_error("VoiceChat has no committed input turn awaiting a response");
    response_available_ = false;
}

void RealtimeTurnControlState::reset() noexcept {
    input_pending_ = false;
    response_available_ = false;
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
    return epoch_;
}

std::uint64_t ConversationState::finish_agent_turn() {
    if (phase_ != ConversationPhase::kAgentSpeaking)
        throw std::logic_error("VoiceChat has no active agent turn to finish");
    const std::uint64_t completed_epoch = epoch_;
    advance_epoch();
    phase_ = input_finished_ ? ConversationPhase::kFinished : ConversationPhase::kListening;
    return completed_epoch;
}

bool ConversationState::invalidate_for_yield() {
    if (phase_ != ConversationPhase::kAgentSpeaking)
        return false;
    advance_epoch();
    phase_ = input_finished_ ? ConversationPhase::kFinished : ConversationPhase::kListening;
    return true;
}

bool ConversationState::barge_in() {
    return invalidate_for_yield();
}

bool ConversationState::yield_to_user() {
    return invalidate_for_yield();
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
    input_finished_ = true;
}

void ConversationState::reset() {
    advance_epoch();
    phase_ = ConversationPhase::kListening;
    input_finished_ = false;
}

bool ConversationState::accepts_output(std::uint64_t output_epoch) const {
    return phase_ == ConversationPhase::kAgentSpeaking && output_epoch == epoch_;
}

} // namespace trtmc::nemotron_voicechat

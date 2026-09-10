/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron_voicechat/runtime/session_state.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace trtmc::nemotron_voicechat {

namespace {

bool barge_in_is_confirmed(bool agent_speaking, int32_t consecutive_speech_frames,
                           int32_t required_speech_frames) {
    return agent_speaking && consecutive_speech_frames >= required_speech_frames;
}

constexpr std::string_view kTranscriptFragmentSeparator = " / ";

bool is_utf8_continuation_byte(char value) {
    return (static_cast<unsigned char>(value) & 0xc0U) == 0x80U;
}

void retain_utf8_suffix(std::string& text, std::size_t max_bytes) {
    if (text.size() <= max_bytes)
        return;
    std::size_t start = text.size() - max_bytes;
    while (start < text.size() && is_utf8_continuation_byte(text[start]))
        ++start;
    text.erase(0, start);
}

bool has_trailing_transcript_fragment(std::string_view pending, std::string_view fragment) {
    if (pending == fragment)
        return true;
    if (pending.size() < fragment.size() + kTranscriptFragmentSeparator.size())
        return false;
    const auto fragment_start = pending.size() - fragment.size();
    const auto separator_start = fragment_start - kTranscriptFragmentSeparator.size();
    return pending.substr(fragment_start) == fragment &&
           pending.substr(separator_start, kTranscriptFragmentSeparator.size()) ==
               kTranscriptFragmentSeparator;
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

bool append_bounded_transcript(std::string& pending, std::string_view final_text,
                               std::size_t max_bytes) {
    if (max_bytes == 0) {
        pending.clear();
        return false;
    }
    if (final_text.empty()) {
        retain_utf8_suffix(pending, max_bytes);
        return false;
    }

    const bool duplicate = has_trailing_transcript_fragment(pending, final_text);
    if (duplicate && pending.size() <= max_bytes)
        return false;

    std::string newest(final_text);
    retain_utf8_suffix(newest, max_bytes);
    if (duplicate || newest.size() != final_text.size()) {
        pending = std::move(newest);
        return !duplicate;
    }

    const auto newest_bytes = kTranscriptFragmentSeparator.size() + newest.size();
    if (pending.empty() || newest_bytes > max_bytes) {
        pending = std::move(newest);
        return true;
    }

    retain_utf8_suffix(pending, max_bytes - newest_bytes);
    if (pending.empty()) {
        pending = std::move(newest);
        return true;
    }
    pending.append(kTranscriptFragmentSeparator);
    pending.append(newest);
    return true;
}

StreamingLinearResampler::StreamingLinearResampler(int32_t source_rate, int32_t target_rate)
    : source_rate_(source_rate), target_rate_(target_rate) {
    if (source_rate_ <= 0 || target_rate_ <= 0)
        throw std::invalid_argument("VoiceChat resampler rates must be positive");
}

void StreamingLinearResampler::append(const float* samples, int32_t count) {
    if (count < 0 || (count > 0 && samples == nullptr))
        throw std::invalid_argument("VoiceChat resampler received invalid samples");
    if (count > 0)
        source_.insert(source_.end(), samples, samples + count);
}

std::vector<float> StreamingLinearResampler::drain(bool final) {
    if (source_rate_ == target_rate_) {
        std::vector<float> result(source_.begin(), source_.end());
        source_origin_ += source_.size();
        produced_ = source_origin_;
        source_.clear();
        return result;
    }

    const auto rounded_output_count = static_cast<std::size_t>(
        std::llround(static_cast<double>(source_end()) * target_rate_ / source_rate_));
    // A non-final prefix must never publish more samples than that same
    // prefix would contain if the stream ended now; final drain cannot retract
    // an early sample. The interpolation-stability bound alone is one sample
    // too permissive for some downsampling ratios (for example 44.1k -> 16k).
    const std::size_t available =
        final ? rounded_output_count : std::min(stable_output_count(), rounded_output_count);
    std::vector<float> result;
    if (available <= produced_) {
        compact_source(final);
        return result;
    }
    result.reserve(available - produced_);
    for (std::size_t output_index = produced_; output_index < available; ++output_index) {
        const double source_position =
            static_cast<double>(output_index) * source_rate_ / target_rate_;
        const auto left_absolute = std::min(static_cast<std::size_t>(source_position),
                                            source_end() == 0 ? 0U : source_end() - 1U);
        const auto right_absolute =
            std::min(left_absolute + 1U, source_end() == 0 ? 0U : source_end() - 1U);
        if (left_absolute < source_origin_ || right_absolute < source_origin_)
            throw std::logic_error("VoiceChat resampler discarded required source history");
        const auto left = left_absolute - source_origin_;
        const auto right = right_absolute - source_origin_;
        const float fraction =
            static_cast<float>(source_position - static_cast<double>(left_absolute));
        const float left_value = source_.empty() ? 0.0F : source_[left];
        const float right_value = source_.empty() ? left_value : source_[right];
        result.push_back(left_value + fraction * (right_value - left_value));
    }
    produced_ = available;
    compact_source(final);
    return result;
}

void StreamingLinearResampler::reset() {
    source_.clear();
    source_origin_ = 0;
    produced_ = 0;
}

std::size_t StreamingLinearResampler::source_end() const {
    return source_origin_ + source_.size();
}

std::size_t StreamingLinearResampler::stable_output_count() const {
    if (source_end() < 2)
        return 0;
    // j * source_rate / target_rate must have both floor and ceil samples.
    const double exclusive =
        static_cast<double>(source_end() - 1) * target_rate_ / static_cast<double>(source_rate_);
    return static_cast<std::size_t>(std::ceil(exclusive));
}

void StreamingLinearResampler::compact_source(bool final) {
    if (source_.empty())
        return;
    const auto keep_from =
        final ? source_end()
              : std::min(source_end(), static_cast<std::size_t>(static_cast<double>(produced_) *
                                                                source_rate_ / target_rate_));
    if (keep_from < source_origin_)
        throw std::logic_error("VoiceChat resampler compaction moved backwards");
    const auto discard = keep_from - source_origin_;
    source_.erase(source_.begin(), source_.begin() + static_cast<std::ptrdiff_t>(discard));
    source_origin_ = keep_from;
}

RollingCachePosition rolling_cache_position(std::int64_t logical_position, int32_t capacity,
                                            int32_t pinned_prefix_rows) {
    if (logical_position < 0)
        throw std::invalid_argument("VoiceChat rolling cache position must be non-negative");
    if (capacity <= 0)
        throw std::invalid_argument("VoiceChat rolling cache capacity must be positive");
    if (pinned_prefix_rows < 0 || pinned_prefix_rows >= capacity)
        throw std::invalid_argument(
            "VoiceChat rolling cache pinned prefix must be within its capacity");

    const int32_t valid_rows =
        static_cast<int32_t>(std::min<std::int64_t>(logical_position, capacity));
    if (logical_position < capacity)
        return {valid_rows, static_cast<int32_t>(logical_position)};

    const int32_t rolling_rows = capacity - pinned_prefix_rows;
    return {
        valid_rows,
        pinned_prefix_rows + static_cast<int32_t>((logical_position - capacity) % rolling_rows),
    };
}

bool PendingUserRequest::begin_utterance() noexcept {
    const bool needs_clean_context = cancelled_response_;
    clear();
    return needs_clean_context;
}

bool PendingUserRequest::append_final(std::string_view text) {
    const bool changed = append_bounded_transcript(text_, text);
    if (changed)
        retry_used_ = false;
    return changed;
}

bool PendingUserRequest::take_automatic_retry() noexcept {
    if (text_.empty() || retry_used_)
        return false;
    retry_used_ = true;
    return true;
}

void PendingUserRequest::clear() noexcept {
    text_.clear();
    retry_used_ = false;
    cancelled_response_ = false;
}

void ResponseUserRequest::begin(std::uint64_t utterance_id, std::string_view known_text) {
    utterance_id_ = utterance_id;
    text_ = known_text;
}

bool ResponseUserRequest::observe_final(std::uint64_t utterance_id, std::string_view text) {
    if (!text_.empty() || utterance_id == 0 || utterance_id != utterance_id_ || text.empty())
        return false;
    text_ = text;
    return true;
}

void ResponseUserRequest::clear() noexcept {
    utterance_id_ = 0;
    text_.clear();
}

void ResponseBoundaryRecovery::response_finished(std::int64_t observation_frame) noexcept {
    if (observation_frame >= 0)
        last_finished_frame_ = observation_frame;
}

bool ResponseBoundaryRecovery::needs_clean_context(std::int64_t speech_start_frame) const noexcept {
    if (!last_finished_frame_.has_value() || speech_start_frame < 0)
        return false;
    constexpr std::int64_t kRecognitionLagFrames = 4; // 320 ms at the native 12.5 Hz cadence.
    return speech_start_frame <= *last_finished_frame_ ||
           speech_start_frame - *last_finished_frame_ <= kRecognitionLagFrames;
}

bool RepetitionWatchdog::has_repeated_suffix(std::size_t block_tokens,
                                             std::size_t repetitions) const {
    const std::size_t required_tokens = block_tokens * repetitions;
    if (tokens_.size() < required_tokens)
        return false;

    const std::size_t start = tokens_.size() - required_tokens;
    for (std::size_t repetition = 1; repetition < repetitions; ++repetition) {
        for (std::size_t offset = 0; offset < block_tokens; ++offset) {
            if (tokens_[start + offset] != tokens_[start + repetition * block_tokens + offset])
                return false;
        }
    }
    return true;
}

bool RepetitionWatchdog::has_near_repeated_suffix(std::size_t block_tokens) const {
    if (tokens_.size() < 3 * block_tokens)
        return false;
    const auto start = tokens_.size() - 3 * block_tokens;
    // Three copies with at most ten percent substitutions in each copy are
    // strong collapse evidence. Two similar sentences or short refrains are
    // insufficient to trip this rule.
    for (std::size_t copy = 1; copy < 3; ++copy) {
        std::size_t different = 0;
        for (std::size_t offset = 0; offset < block_tokens; ++offset)
            different += tokens_[start + offset] != tokens_[start + copy * block_tokens + offset];
        if (different * 10 > block_tokens)
            return false;
    }
    return true;
}

bool RepetitionWatchdog::observe(int32_t token) {
    if (tripped_)
        return true;

    tokens_.push_back(token);
    if (tokens_.size() > kHistoryTokens)
        tokens_.pop_front();

    if (has_repeated_suffix(1, 8)) {
        tripped_ = true;
        return true;
    }
    for (std::size_t block_tokens = 3; block_tokens <= 7; ++block_tokens) {
        if (has_repeated_suffix(block_tokens, 3)) {
            tripped_ = true;
            return true;
        }
    }
    for (std::size_t block_tokens = 8; block_tokens <= 48; ++block_tokens) {
        if (has_repeated_suffix(block_tokens, 2)) {
            tripped_ = true;
            return true;
        }
    }
    for (std::size_t block_tokens = 12; block_tokens <= 48; ++block_tokens) {
        if (has_near_repeated_suffix(block_tokens)) {
            tripped_ = true;
            return true;
        }
    }
    return false;
}

void RepetitionWatchdog::reset() noexcept {
    tokens_.clear();
    tripped_ = false;
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
        decision.discarded_candidate = speech_frames_ != 0;
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
    reset_stream_frontier();
}

void RnntTurnDetector::reset_stream_frontier() {
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

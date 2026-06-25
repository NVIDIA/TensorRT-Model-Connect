#pragma once

#include <cstdint>

namespace trtmc {

struct SpeechDecodeStopState {
    int32_t text_eos_streak{0};
    int32_t text_pad_streak{0};
    bool stop_requested{false};
    int32_t stop_collect_until_offset{-1};
};

enum class SpeechDecodeStopReason { kNone, kTextEos, kTextPadFallback, kContinuationCap };

struct SpeechDecodeStopInput {
    int32_t text_eos_token_id{-1};
    int32_t text_padding_id{0};
    int32_t effective_frames{0};
    int32_t extra_tail{0};
    int32_t target_pos{0};
    int32_t sampled_text_token{0};
    int32_t offset{0};
    int32_t max_delay{0};
    bool text_provided{false};
};

struct SpeechDecodeStopDecision {
    SpeechDecodeStopState state;
    SpeechDecodeStopReason reason{SpeechDecodeStopReason::kNone};
    bool should_break{false};
};

inline constexpr int32_t kSpeechMinConsecutiveTextEos = 2;
inline constexpr int32_t kSpeechMinConsecutiveTextPadAfterInput = 16;
inline constexpr int32_t kSpeechMaxContinuationFramesAfterInput = 16;

inline bool SpeechDecodeReachedEffectiveFrames(const SpeechDecodeStopInput& input) {
    return input.target_pos >= input.effective_frames;
}

inline bool SpeechDecodeDrainComplete(const SpeechDecodeStopState& state,
                                      const SpeechDecodeStopInput& input) {
    return state.stop_requested && input.offset >= state.stop_collect_until_offset;
}

inline void RequestSpeechDecodeStop(SpeechDecodeStopDecision& decision,
                                    const SpeechDecodeStopInput& input,
                                    SpeechDecodeStopReason reason) {
    decision.state.stop_requested = true;
    decision.state.stop_collect_until_offset = input.offset + input.max_delay;
    decision.reason = reason;
}

inline bool IsSpeechTextEosCandidate(const SpeechDecodeStopInput& input) {
    return input.text_eos_token_id >= 0 && !input.text_provided &&
           SpeechDecodeReachedEffectiveFrames(input) &&
           input.sampled_text_token == input.text_eos_token_id;
}

inline void UpdateSpeechTextEosStreak(SpeechDecodeStopDecision& decision,
                                      const SpeechDecodeStopInput& input) {
    if (!IsSpeechTextEosCandidate(input)) {
        decision.state.text_eos_streak = 0;
        return;
    }

    ++decision.state.text_eos_streak;
    if (!decision.state.stop_requested &&
        decision.state.text_eos_streak >= kSpeechMinConsecutiveTextEos) {
        RequestSpeechDecodeStop(decision, input, SpeechDecodeStopReason::kTextEos);
    }
}

inline bool IsSpeechTextPadFallbackCandidate(const SpeechDecodeStopState& state,
                                             const SpeechDecodeStopInput& input) {
    return !state.stop_requested && input.extra_tail > 0 && !input.text_provided &&
           SpeechDecodeReachedEffectiveFrames(input) &&
           input.sampled_text_token == input.text_padding_id;
}

inline void UpdateSpeechTextPadStreak(SpeechDecodeStopDecision& decision,
                                      const SpeechDecodeStopInput& input) {
    if (!IsSpeechTextPadFallbackCandidate(decision.state, input)) {
        decision.state.text_pad_streak = 0;
        return;
    }

    ++decision.state.text_pad_streak;
    if (decision.state.text_pad_streak >= kSpeechMinConsecutiveTextPadAfterInput) {
        RequestSpeechDecodeStop(decision, input, SpeechDecodeStopReason::kTextPadFallback);
    }
}

inline bool IsSpeechContinuationCapCandidate(const SpeechDecodeStopState& state,
                                             const SpeechDecodeStopInput& input) {
    return !SpeechDecodeDrainComplete(state, input) && !state.stop_requested &&
           input.extra_tail > 0 &&
           input.target_pos >= (input.effective_frames + kSpeechMaxContinuationFramesAfterInput);
}

inline SpeechDecodeStopDecision UpdateSpeechDecodeStopState(SpeechDecodeStopState state,
                                                            const SpeechDecodeStopInput& input) {
    SpeechDecodeStopDecision decision;
    decision.state = state;

    UpdateSpeechTextEosStreak(decision, input);
    UpdateSpeechTextPadStreak(decision, input);

    if (IsSpeechContinuationCapCandidate(decision.state, input)) {
        RequestSpeechDecodeStop(decision, input, SpeechDecodeStopReason::kContinuationCap);
    }

    decision.should_break = SpeechDecodeDrainComplete(decision.state, input);
    return decision;
}

} // namespace trtmc

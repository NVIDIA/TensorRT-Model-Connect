// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-10
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Speech decode stop policy: EOS detection, pad fallback, continuation cap
// Preconditions:  SpeechDecodeStopInput with configured thresholds and token sequences
// Postconditions: EOS requires consecutive tokens, pad fallback needs tail+threshold, cap breaks
// correctly
// =============================================================================

#include "runtime/models/personaplex/speech_decode_stop_policy.h"

#include <iostream>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

trtmc::SpeechDecodeStopInput make_base_input() {
    trtmc::SpeechDecodeStopInput input;
    input.text_eos_token_id = 7;
    input.text_padding_id = 3;
    input.effective_frames = 10;
    input.extra_tail = 4;
    input.target_pos = 10;
    input.sampled_text_token = 1;
    input.offset = 20;
    input.max_delay = 2;
    input.text_provided = false;
    return input;
}

void test_text_eos_requires_two_consecutive_tokens_and_drains_delay() {
    trtmc::SpeechDecodeStopState state;
    auto input = make_base_input();
    input.sampled_text_token = input.text_eos_token_id;

    auto decision = trtmc::UpdateSpeechDecodeStopState(state, input);
    check(decision.reason == trtmc::SpeechDecodeStopReason::kNone,
          "eos threshold: first eos has no stop reason");
    check(decision.state.text_eos_streak == 1, "eos threshold: first eos increments streak");
    check(!decision.state.stop_requested, "eos threshold: first eos does not request stop");
    check(!decision.should_break, "eos threshold: first eos does not break");

    input.target_pos += 1;
    input.offset += 1;
    decision = trtmc::UpdateSpeechDecodeStopState(decision.state, input);
    check(decision.reason == trtmc::SpeechDecodeStopReason::kTextEos,
          "eos threshold: second eos requests stop");
    check(decision.state.stop_requested, "eos threshold: second eos marks stop requested");
    check(decision.state.stop_collect_until_offset == input.offset + input.max_delay,
          "eos threshold: second eos sets drain offset");
    check(!decision.should_break, "eos threshold: stop waits for drain");

    input.sampled_text_token = 0;
    input.target_pos += 1;
    input.offset = decision.state.stop_collect_until_offset;
    decision = trtmc::UpdateSpeechDecodeStopState(decision.state, input);
    check(decision.should_break, "eos threshold: stop breaks when drain offset reached");
}

void test_text_eos_reset_conditions() {
    trtmc::SpeechDecodeStopState state;
    state.text_eos_streak = 1;
    auto input = make_base_input();
    input.sampled_text_token = input.text_eos_token_id;
    input.target_pos = input.effective_frames - 1;

    auto decision = trtmc::UpdateSpeechDecodeStopState(state, input);
    check(decision.state.text_eos_streak == 0, "eos reset: before effective frames resets streak");
    check(!decision.state.stop_requested, "eos reset: before effective frames does not stop");

    state.text_eos_streak = 1;
    input = make_base_input();
    input.sampled_text_token = input.text_eos_token_id;
    input.text_provided = true;

    decision = trtmc::UpdateSpeechDecodeStopState(state, input);
    check(decision.state.text_eos_streak == 0, "eos reset: provided text resets streak");
    check(decision.reason == trtmc::SpeechDecodeStopReason::kNone,
          "eos reset: provided text has no stop reason");
}

void test_pad_fallback_requires_tail_and_threshold() {
    trtmc::SpeechDecodeStopState state;
    auto input = make_base_input();
    input.sampled_text_token = input.text_padding_id;

    for (int32_t i = 0; i < trtmc::kSpeechMinConsecutiveTextPadAfterInput - 1; ++i) {
        input.target_pos = input.effective_frames + i;
        input.offset = 30 + i;
        const auto decision = trtmc::UpdateSpeechDecodeStopState(state, input);
        check(decision.reason == trtmc::SpeechDecodeStopReason::kNone,
              "pad threshold: pre-threshold pad has no stop reason");
        check(!decision.state.stop_requested, "pad threshold: pre-threshold pad does not stop");
        state = decision.state;
    }

    input.target_pos = input.effective_frames + trtmc::kSpeechMinConsecutiveTextPadAfterInput - 1;
    input.offset = 30 + trtmc::kSpeechMinConsecutiveTextPadAfterInput - 1;
    const auto decision = trtmc::UpdateSpeechDecodeStopState(state, input);
    check(decision.reason == trtmc::SpeechDecodeStopReason::kTextPadFallback,
          "pad threshold: threshold pad requests stop");
    check(decision.state.text_pad_streak == trtmc::kSpeechMinConsecutiveTextPadAfterInput,
          "pad threshold: threshold pad records full streak");
    check(decision.state.stop_requested, "pad threshold: threshold pad marks stop requested");
    check(decision.state.stop_collect_until_offset == input.offset + input.max_delay,
          "pad threshold: threshold pad sets drain offset");
}

void test_pad_fallback_disabled_without_tail() {
    trtmc::SpeechDecodeStopState state;
    state.text_pad_streak = trtmc::kSpeechMinConsecutiveTextPadAfterInput - 1;
    auto input = make_base_input();
    input.extra_tail = 0;
    input.sampled_text_token = input.text_padding_id;

    const auto decision = trtmc::UpdateSpeechDecodeStopState(state, input);
    check(decision.state.text_pad_streak == 0, "pad disabled: zero tail resets streak");
    check(!decision.state.stop_requested, "pad disabled: zero tail prevents stop");
    check(decision.reason == trtmc::SpeechDecodeStopReason::kNone,
          "pad disabled: zero tail has no stop reason");
}

void test_continuation_cap_can_break_immediately() {
    trtmc::SpeechDecodeStopState state;
    auto input = make_base_input();
    input.extra_tail = 1;
    input.max_delay = 0;
    input.target_pos = input.effective_frames + trtmc::kSpeechMaxContinuationFramesAfterInput;
    input.offset = 99;

    const auto decision = trtmc::UpdateSpeechDecodeStopState(state, input);
    check(decision.reason == trtmc::SpeechDecodeStopReason::kContinuationCap,
          "continuation cap: threshold requests stop");
    check(decision.state.stop_requested, "continuation cap: threshold marks stop requested");
    check(decision.state.stop_collect_until_offset == input.offset,
          "continuation cap: zero delay drains immediately");
    check(decision.should_break, "continuation cap: zero delay breaks immediately");
}

} // namespace

int main() {
    test_text_eos_requires_two_consecutive_tokens_and_drains_delay();
    test_text_eos_reset_conditions();
    test_pad_fallback_requires_tail_and_threshold();
    test_pad_fallback_disabled_without_tail();
    test_continuation_cap_can_break_immediately();

    if (failures != 0)
        return 1;
    std::cout << "test_speech_decode_stop_policy: PASS\n";
    return 0;
}

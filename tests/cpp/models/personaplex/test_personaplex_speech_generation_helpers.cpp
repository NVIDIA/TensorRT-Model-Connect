/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-12
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Speech generation helpers: delay cache, waveform trim/normalize, postprocess
// safety Preconditions:  Delay cache with known codebook delays, waveform samples Postconditions:
// Delays generalize to num_codebooks, cache reads correct, waveform trimmed/normalized
// =============================================================================

#include "runtime/models/personaplex/speech_delay_cache.h"
#include "runtime/models/personaplex/speech_mimi_encode_plan.h"
#include "runtime/models/personaplex/speech_waveform_postprocess.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void check_close(float actual, float expected, float tolerance, const char* name) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

void test_default_speech_delays_generalize_to_num_codebooks() {
    const auto delays = trtmc::make_default_speech_delays(6);
    check(delays.size() == 7, "default delays size");
    check(delays[0] == 0, "default delays text stream");
    check(delays[1] == 0, "default delays first moshi stream");
    check(delays[4] == 0, "default delays first user stream");
    check(delays[2] == 1 && delays[3] == 1 && delays[5] == 1 && delays[6] == 1,
          "default delays remaining streams");
}

void test_delay_cache_reads_and_collects_outputs() {
    auto state = trtmc::make_delay_cache_state({0, 0, 0, 1, 1}, 4);
    check(state.total_k == 5, "delay cache total streams");
    check(state.max_delay == 1, "delay cache max delay");

    const std::vector<int32_t> codec_tokens = {
        101,
        102,
        201,
        202,
    };

    trtmc::seed_delay_offset_zero(state, 9000, 8000);
    trtmc::write_user_tokens_to_delay_cache(state, codec_tokens, 0, 2, 2, 2, 8000);
    trtmc::fill_initial_delay_tokens(state, 0, 9000, 8000);

    int32_t text_input = -1;
    std::vector<int32_t> moshi_input(2, -1);
    std::vector<int32_t> user_input(2, -1);
    trtmc::read_model_inputs_from_delay_cache(state, 0, 2, text_input, moshi_input, user_input);
    check(text_input == 9000, "delay cache reads text input");
    check(moshi_input == std::vector<int32_t>({8000, 8000}), "delay cache reads moshi input");
    check(user_input == std::vector<int32_t>({8000, 8000}), "delay cache reads delayed user input");

    std::vector<int32_t> target_audio_tokens(4, -1);
    std::vector<uint8_t> target_audio_provided(4, 0);
    trtmc::build_target_audio_arrays(state, 1, 4, 8000, target_audio_tokens, target_audio_provided);
    check(target_audio_tokens == std::vector<int32_t>({-2, -2, 101, 102}),
          "target audio exposes delayed user tokens");
    check(target_audio_provided == std::vector<uint8_t>({0, 0, 1, 1}),
          "target audio marks provided delayed streams");

    const std::vector<int32_t> frame_codes = {501, 502, 503, 504};
    trtmc::write_generated_tokens_to_delay_cache(state, 1, 9100, false, frame_codes, 4);
    std::vector<int32_t> output_codes;
    const bool collected =
        trtmc::collect_output_codes_from_delay_cache(state, 2, state.max_delay, 2, output_codes);
    check(collected, "delay cache collects output after max delay");
    check(output_codes == std::vector<int32_t>({501, 502}),
          "delay cache collects mimi codebooks only");
    int32_t output_text = -1;
    const bool text_collected =
        trtmc::collect_output_text_from_delay_cache(state, 2, state.max_delay, output_text);
    check(text_collected, "delay cache collects output text after max delay");
    check(output_text == 9100, "delay cache collects aligned output text token");
}

void test_teacher_trace_is_written_and_predictions_are_realigned() {
    auto state = trtmc::make_delay_cache_state({0, 0, 1, 1, 0, 1, 1}, 6);
    const std::vector<int32_t> teacher_text = {101, 102};
    const std::vector<int32_t> teacher_audio = {
        201, 202, 203, 211, 212, 213,
    };
    auto predictions = trtmc::make_speech_teacher_predictions(2, 3);

    trtmc::write_speech_teacher_frame_to_delay_cache(state, teacher_text, teacher_audio, 3, 0, 1);
    check(state.cache[trtmc::delay_cache_index(state, 0, 1)] == 101,
          "teacher trace writes text at its delayed position");
    check(state.cache[trtmc::delay_cache_index(state, 1, 1)] == 201,
          "teacher trace writes delay-zero audio");
    check(state.cache[trtmc::delay_cache_index(state, 2, 2)] == 202,
          "teacher trace writes delay-one audio");
    check(state.provided[trtmc::delay_cache_index(state, 3, 2)] == 1,
          "teacher trace marks forced audio as provided");
    check(state.provided[trtmc::delay_cache_index(state, 4, 1)] == 0,
          "teacher trace leaves user streams untouched");

    trtmc::capture_speech_teacher_predictions(state, teacher_text, teacher_audio, 3, 1,
                                              {101, 201, 999, 999}, predictions);
    trtmc::write_speech_teacher_frame_to_delay_cache(state, teacher_text, teacher_audio, 3, 1, 2);
    trtmc::capture_speech_teacher_predictions(state, teacher_text, teacher_audio, 3, 2,
                                              {999, 211, 202, 203}, predictions);
    trtmc::capture_speech_teacher_predictions(state, teacher_text, teacher_audio, 3, 3,
                                              {999, 999, 212, 213}, predictions);

    check(predictions.text == std::vector<int32_t>({101, 999}),
          "teacher text predictions align by logical frame");
    check(predictions.audio == std::vector<int32_t>({201, 202, 203, 211, 212, 213}),
          "teacher audio predictions undo codebook delays");
}

void test_waveform_trim_and_peak_normalize() {
    std::vector<float> waveform(20, 0.0F);
    const auto trim_result = trtmc::trim_speech_waveform_to_generated_frames(10, 2.0F, 3, waveform);
    check(trim_result.trimmed, "waveform trim applied");
    check(trim_result.expected_samples == 15, "waveform trim expected samples");
    check(waveform.size() == 15, "waveform trim resized output");

    waveform = {2.0F, -1.0F, 0.5F};
    const auto normalize_result = trtmc::peak_normalize_speech_waveform(waveform);
    check(normalize_result.normalized, "waveform normalize applied");
    check_close(normalize_result.peak, 2.0F, 1e-6F, "waveform normalize peak");
    check_close(normalize_result.scale, 0.475F, 1e-6F, "waveform normalize scale");
    check_close(waveform[0], 0.95F, 1e-6F, "waveform normalize sample 0");
    check_close(waveform[1], -0.475F, 1e-6F, "waveform normalize sample 1");
}

void test_waveform_postprocess_skips_invalid_or_safe_inputs() {
    std::vector<float> waveform = {0.1F, 0.2F, 0.3F};
    std::vector<float> empty_waveform;
    const auto no_trim_empty =
        trtmc::trim_speech_waveform_to_generated_frames(10, 2.0F, 2, empty_waveform);
    check(!no_trim_empty.trimmed && no_trim_empty.expected_samples == 0,
          "waveform trim skips empty waveform");

    const auto no_trim_bad_frames =
        trtmc::trim_speech_waveform_to_generated_frames(10, 0.0F, 2, waveform);
    check(!no_trim_bad_frames.trimmed, "waveform trim skips invalid frame rate");
    check(waveform.size() == 3, "waveform trim preserves waveform when skipped");

    const auto no_trim_expected_ge_size =
        trtmc::trim_speech_waveform_to_generated_frames(10, 2.0F, 1, waveform);
    check(!no_trim_expected_ge_size.trimmed,
          "waveform trim skips when expected samples do not shrink output");

    const auto no_norm_empty = trtmc::peak_normalize_speech_waveform(waveform, 0.8F);
    check(!no_norm_empty.normalized, "waveform normalize skips already safe waveform");
    check_close(no_norm_empty.peak, 0.3F, 1e-6F, "waveform normalize reports measured safe peak");

    const auto no_norm_zero = trtmc::peak_normalize_speech_waveform(empty_waveform);
    check(!no_norm_zero.normalized && no_norm_zero.peak == 0.0F,
          "waveform normalize skips empty waveform");
}

void test_mimi_encode_plan_keeps_only_the_causal_input_prefix() {
    const auto exact = trtmc::build_mimi_encode_plan(658560, 983040, 512);
    check(exact.input_fits, "Mimi long input fits declared engine capacity");
    check(exact.valid_frames == 343, "Mimi exact-hop input keeps 343 frames");

    const auto partial = trtmc::build_mimi_encode_plan(99844, 983040, 512);
    check(partial.input_fits, "Mimi short input fits declared engine capacity");
    check(partial.valid_frames == 53, "Mimi partial-hop input rounds up to 53 frames");

    const auto oversized = trtmc::build_mimi_encode_plan(983041, 983040, 512);
    check(!oversized.input_fits, "Mimi oversized input is rejected instead of truncated");
    check(oversized.valid_frames == 0, "Mimi oversized input exposes no encoded frames");
}

int count_allowed_mimi_keys(const trtmc::MimiRingAttentionInputs& inputs, int32_t query) {
    int allowed = 0;
    for (int32_t column = 0; column < trtmc::kMimiAttentionContext; ++column) {
        const auto index = static_cast<std::size_t>(query) * trtmc::kMimiAttentionContext +
                           static_cast<std::size_t>(column);
        allowed += inputs.mask[index] == 0.0F ? 1 : 0;
    }
    return allowed;
}

void test_mimi_ring_attention_matches_official_cache_wrap() {
    const auto first = trtmc::build_mimi_ring_attention_inputs(0);
    check(first.position_ids == std::array<int32_t, 2>{0, 1},
          "Mimi first chunk uses absolute positions zero and one");
    check(first.cache_indices == std::array<int32_t, 2>{0, 1},
          "Mimi first chunk writes the first two ring slots");
    check(count_allowed_mimi_keys(first, 0) == 1, "Mimi first query attends only to itself");
    check(count_allowed_mimi_keys(first, 1) == 2,
          "Mimi second query attends to both first-chunk tokens");

    const auto wrap = trtmc::build_mimi_ring_attention_inputs(124);
    check(wrap.position_ids == std::array<int32_t, 2>{248, 249},
          "Mimi wrap chunk keeps absolute positions");
    check(wrap.cache_indices == std::array<int32_t, 2>{248, 249},
          "Mimi wrap chunk fills the final ring slots");
    check(wrap.mask[0] == trtmc::kMimiAttentionMaskPenalty,
          "Mimi full-ring mask excludes the end pointer slot");
    check(count_allowed_mimi_keys(wrap, 0) == 248,
          "Mimi wrap first query matches official physical-ring visibility");
    check(count_allowed_mimi_keys(wrap, 1) == 249,
          "Mimi wrap second query matches official physical-ring visibility");

    const auto wrapped = trtmc::build_mimi_ring_attention_inputs(125);
    check(wrapped.cache_indices == std::array<int32_t, 2>{0, 1},
          "Mimi cache indices wrap to the first two slots");
    check(wrapped.mask[2] == trtmc::kMimiAttentionMaskPenalty,
          "Mimi wrapped mask excludes the new end pointer slot");
    check(count_allowed_mimi_keys(wrapped, 0) == 248,
          "Mimi wrapped first query retains official context");
    check(count_allowed_mimi_keys(wrapped, 1) == 249,
          "Mimi wrapped second query retains official context");
}

} // namespace

int main() {
    test_default_speech_delays_generalize_to_num_codebooks();
    test_delay_cache_reads_and_collects_outputs();
    test_teacher_trace_is_written_and_predictions_are_realigned();
    test_waveform_trim_and_peak_normalize();
    test_waveform_postprocess_skips_invalid_or_safe_inputs();
    test_mimi_encode_plan_keeps_only_the_causal_input_prefix();
    test_mimi_ring_attention_matches_official_cache_wrap();

    if (g_failures != 0) {
        std::cerr << g_failures << " speech generation helper test(s) failed\n";
        return 1;
    }
    return 0;
}

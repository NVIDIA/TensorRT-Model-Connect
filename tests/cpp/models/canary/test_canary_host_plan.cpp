/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-19
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Canary host plan: mel length, initial tokens, encoder planning, cross-KV apply
// Preconditions:  Canary config with valid mel/encoder parameters
// Postconditions: Mel length and tokens correct, encoder mask planned, cross-KV copy operations
// tracked
// =============================================================================

#include "runtime/models/canary/canary_cross_kv_apply.h"
#include "runtime/models/canary/canary_cross_kv_plan.h"
#include "runtime/models/canary/canary_host_plan.h"
#include "runtime/models/canary/canary_request.h"

#include <cstdint>
#include <cstring>
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

void test_expected_mel_length_and_initial_tokens() {
    trtmc::CanaryConfig cfg;
    cfg.max_source_positions = 1500;
    check(trtmc::resolve_canary_expected_mel_length(cfg) == 3000,
          "canary expected mel length defaults to max_source_positions * 2");

    cfg.mel_length = 4096;
    check(trtmc::resolve_canary_expected_mel_length(cfg) == 4096,
          "canary expected mel length honors explicit override");

    cfg.decoder_start_token_ids = {1, 2, 3};
    check(trtmc::make_canary_initial_decoder_tokens(cfg) == std::vector<int32_t>({1, 2, 3}),
          "canary initial tokens honor custom sequence");

    cfg.decoder_start_token_ids.clear();
    cfg.decoder_start_token_id = 10;
    cfg.language_token_id = 11;
    cfg.transcribe_token_id = 12;
    cfg.notimestamps_token_id = 13;
    check(trtmc::make_canary_initial_decoder_tokens(cfg) == std::vector<int32_t>({10, 11, 12, 13}),
          "canary initial tokens fall back to default special tokens");
}

void test_encoder_length_planning() {
    check(trtmc::count_canary_stride2_stages(3000, 1500) == 1,
          "canary stride stage count handles one stride-2 stage");
    check(trtmc::count_canary_stride2_stages(4000, 1000) == 2,
          "canary stride stage count handles repeated stride-2 stages");
    check(trtmc::apply_canary_stride2_subsampling(3000, 1) == 1500,
          "canary stride-2 subsampling halves even length");
    check(trtmc::apply_canary_stride2_subsampling(2999, 1) == 1500,
          "canary stride-2 subsampling rounds like convolution output");

    check(trtmc::compute_canary_actual_encoder_length(2000, 3000, 1500) == 1000,
          "canary actual encoder length subsamples partial mel length");
    check(trtmc::compute_canary_actual_encoder_length(3000, 3000, 1500) == 0,
          "canary actual encoder length returns zero for full-length mel");
    check(trtmc::compute_canary_actual_encoder_length(100, 0, 1500) == 0,
          "canary actual encoder length returns zero for invalid full mel length");
}

trtmc::CanaryConfig configurable_canary() {
    trtmc::CanaryConfig cfg;
    cfg.max_target_positions = 64;
    cfg.decoder_start_token_ids = {100, 101, 102, 103, 20, 20, 30, 31, 40, 41};
    cfg.supported_languages = {"en", "fr"};
    cfg.language_token_ids = {20, 21};
    cfg.punctuation_token_id = 30;
    cfg.no_punctuation_token_id = 32;
    cfg.timestamp_token_id = 42;
    cfg.no_timestamp_token_id = 40;
    return cfg;
}

void test_configurable_request_prompt() {
    const auto model = configurable_canary();
    trtmc::TranscriptionConfig request;
    request.max_output_tokens = 20;
    request.beam_size = 4;
    request.task = trtmc::TranscriptionTask::kTranslate;
    request.source_language = "en";
    request.target_language = "fr";
    request.punctuation = false;
    request.timestamps = true;

    const auto tokens = trtmc::make_canary_request_tokens(model, request, nullptr);
    check(tokens == std::vector<int32_t>({100, 101, 102, 103, 20, 21, 32, 31, 42, 41}),
          "Canary request replaces language and output control prompt slots");
}

void test_configurable_request_validation() {
    const auto model = configurable_canary();
    trtmc::TranscriptionConfig request;
    request.max_output_tokens = 20;

    auto rejects = [&model](const trtmc::TranscriptionConfig& candidate) {
        try {
            (void)trtmc::make_canary_request_tokens(model, candidate, nullptr);
            return false;
        } catch (const std::invalid_argument&) {
            return true;
        }
    };

    request.target_language = "fr";
    check(rejects(request), "Canary transcription rejects different target language");
    request.task = trtmc::TranscriptionTask::kTranslate;
    request.target_language = "en";
    check(rejects(request), "Canary translation rejects equal languages");
    request.target_language = "fr";
    request.source_language = "de";
    check(rejects(request), "Canary rejects language absent from bundle metadata");
    request.source_language = "en";
    request.beam_size = 17;
    check(rejects(request), "Canary rejects beam size above supported bound");
    request.beam_size = 1;
    request.beam_length_penalty = -1.0F;
    check(rejects(request), "Canary rejects negative beam length penalty");
    request.beam_length_penalty = 1.0F;
    request.max_output_tokens = 65;
    check(rejects(request), "Canary rejects output length above model bound");
    request.max_output_tokens = 20;
    request.segment_duration_seconds = -1.0F;
    check(rejects(request), "Canary rejects negative segment duration");
}

void test_mel_padding_and_truncation_are_row_major() {
    const float mel_data[] = {
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
    };
    const auto padded = trtmc::build_canary_padded_mel_input(mel_data, 2, 3, 5);
    check(padded.size() == 10, "canary padded mel buffer uses expected size");
    check(padded ==
              std::vector<float>({1.0F, 2.0F, 3.0F, 0.0F, 0.0F, 4.0F, 5.0F, 6.0F, 0.0F, 0.0F}),
          "canary padded mel buffer pads each mel row independently");

    const auto truncated = trtmc::build_canary_padded_mel_input(mel_data, 2, 3, 2);
    check(truncated == std::vector<float>({1.0F, 2.0F, 4.0F, 5.0F}),
          "canary padded mel buffer truncates each mel row independently");

    check(trtmc::build_canary_padded_mel_input(nullptr, 2, 3, 5).empty(),
          "canary padded mel buffer returns empty for null input");
}

void test_encoder_mask_and_cross_kv_plan() {
    const auto full_mask = trtmc::build_canary_encoder_mask_values(4, 4);
    check(full_mask == std::vector<float>({0.0F, 0.0F, 0.0F, 0.0F}),
          "canary encoder mask leaves valid sequence unmasked");

    const auto partial_mask = trtmc::build_canary_encoder_mask_values(5, 2);
    check(partial_mask == std::vector<float>({0.0F, 0.0F, -10000.0F, -10000.0F, -10000.0F}),
          "canary encoder mask marks padded encoder positions");

    const auto clamped_mask = trtmc::build_canary_encoder_mask_values(3, -2);
    check(clamped_mask == std::vector<float>({-10000.0F, -10000.0F, -10000.0F}),
          "canary encoder mask clamps negative actual length");

    const auto no_mask = trtmc::build_canary_encoder_mask_values(0, 0);
    check(no_mask.empty(), "canary encoder mask returns empty for invalid sequence length");

    const auto full_plan = trtmc::make_canary_cross_kv_plan(1500, 4, 0);
    check(full_plan.buffer_bytes ==
              static_cast<std::size_t>(1500 * 4 * static_cast<int>(sizeof(float))),
          "canary cross-kv plan computes full buffer size");
    check(!full_plan.zero_pad_encoder_output && full_plan.pad_bytes == 0,
          "canary cross-kv plan skips zero-padding when actual length is unknown");

    const auto partial_plan = trtmc::make_canary_cross_kv_plan(1500, 4, 1000);
    check(partial_plan.zero_pad_encoder_output,
          "canary cross-kv plan marks partial encoder output for zero-padding");
    check(partial_plan.valid_bytes ==
              static_cast<std::size_t>(1000 * 4 * static_cast<int>(sizeof(float))),
          "canary cross-kv plan computes valid byte span");
    check(partial_plan.pad_bytes == partial_plan.buffer_bytes - partial_plan.valid_bytes,
          "canary cross-kv plan computes padded byte span");

    const auto invalid_plan = trtmc::make_canary_cross_kv_plan(0, 4, 0);
    check(invalid_plan.buffer_bytes == 0 && !invalid_plan.zero_pad_encoder_output,
          "canary cross-kv plan returns empty plan for invalid shape");
}

void test_cross_kv_apply_tracks_zero_and_copy_operations() {
    const auto plan = trtmc::make_canary_cross_kv_plan(8, 4, 3);
    trtmc::CanaryCrossKvApplyStats stats;
    std::vector<std::size_t> zero_calls;
    std::vector<std::pair<std::size_t, trtmc::CanaryCrossKvBufferKind>> copy_calls;
    std::string error;

    const bool ok = trtmc::apply_canary_cross_kv_plan(
        plan, 2,
        [&zero_calls](std::size_t valid_bytes, std::size_t pad_bytes) {
            zero_calls.emplace_back(valid_bytes + pad_bytes);
            return true;
        },
        [&copy_calls](std::size_t layer, trtmc::CanaryCrossKvBufferKind kind, std::size_t bytes) {
            copy_calls.emplace_back(layer, kind);
            return bytes > 0;
        },
        error, &stats);

    check(ok, "canary cross-kv apply succeeds for valid plan");
    check(error.empty(), "canary cross-kv apply leaves error empty on success");
    check(zero_calls.size() == 1, "canary cross-kv apply zeroes encoder padding once");
    check(copy_calls.size() == 4, "canary cross-kv apply copies k and v per layer");
    check(stats.zero_ops == 1, "canary cross-kv apply counts zero operations");
    check(stats.copy_ops == 4, "canary cross-kv apply counts copy operations");
    check(copy_calls[0] == std::make_pair<std::size_t, trtmc::CanaryCrossKvBufferKind>(
                               0, trtmc::CanaryCrossKvBufferKind::K),
          "canary cross-kv apply copies cross_k first");
    check(copy_calls[1] == std::make_pair<std::size_t, trtmc::CanaryCrossKvBufferKind>(
                               0, trtmc::CanaryCrossKvBufferKind::V),
          "canary cross-kv apply copies cross_v second");
}

void test_cross_kv_apply_reports_failures() {
    const auto plan = trtmc::make_canary_cross_kv_plan(8, 4, 3);
    std::string error;

    bool ok = trtmc::apply_canary_cross_kv_plan(
        plan, 1, [](std::size_t, std::size_t) { return false; },
        [](std::size_t, trtmc::CanaryCrossKvBufferKind, std::size_t) { return true; }, error);
    check(!ok, "canary cross-kv apply fails when zeroing fails");
    check(error == "failed to zero canary encoder padding",
          "canary cross-kv apply reports zeroing failure");

    error.clear();
    ok = trtmc::apply_canary_cross_kv_plan(
        plan, 1, [](std::size_t, std::size_t) { return true; },
        [](std::size_t, trtmc::CanaryCrossKvBufferKind kind, std::size_t) {
            return kind == trtmc::CanaryCrossKvBufferKind::K;
        },
        error);
    check(!ok, "canary cross-kv apply fails when copy fails");
    check(error == "failed to copy canary cross_v", "canary cross-kv apply reports copy failure");

    error.clear();
    ok = trtmc::apply_canary_cross_kv_plan(
        trtmc::CanaryCrossKvPlan{}, 1, [](std::size_t, std::size_t) { return true; },
        [](std::size_t, trtmc::CanaryCrossKvBufferKind, std::size_t) { return true; }, error);
    check(!ok, "canary cross-kv apply rejects empty plan");
    check(error == "invalid canary cross-kv plan", "canary cross-kv apply reports invalid plan");
}

} // namespace

int main() {
    test_expected_mel_length_and_initial_tokens();
    test_encoder_length_planning();
    test_configurable_request_prompt();
    test_configurable_request_validation();
    test_mel_padding_and_truncation_are_row_major();
    test_encoder_mask_and_cross_kv_plan();
    test_cross_kv_apply_tracks_zero_and_copy_operations();
    test_cross_kv_apply_reports_failures();

    if (g_failures != 0) {
        std::cerr << g_failures << " canary host plan test(s) failed\n";
        return 1;
    }
    return 0;
}

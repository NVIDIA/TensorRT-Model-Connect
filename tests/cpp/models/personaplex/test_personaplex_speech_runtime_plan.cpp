/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-14
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Speech runtime plan: encoder shape fallback, prompt injection, generation
// settings Preconditions:  Speech config with valid/invalid cached shapes Postconditions: Encoder
// shape falls back correctly, prompt injected, generation settings applied
// =============================================================================

#include "runtime/models/personaplex/speech_runtime_plan.h"

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

void test_encoder_shape_falls_back_to_config_when_cache_shape_is_invalid() {
    trtmc::SpeechConfig cfg;
    cfg.num_codebooks = 8;

    const auto info = trtmc::resolve_encoder_shape_without_engine(cfg, 4, 2, 17);

    check(info.encode_codebooks == 4,
          "speech runtime plan keeps cached codebook count when present");
    check(info.num_frames == 4,
          "speech runtime plan recomputes frame count when cached shape is invalid");
}

void test_encoder_shape_uses_config_when_no_cached_shape_exists() {
    trtmc::SpeechConfig cfg;
    cfg.num_codebooks = 6;

    const auto info = trtmc::resolve_encoder_shape_without_engine(cfg, 0, 0, 18);

    check(info.encode_codebooks == 6, "speech runtime plan falls back to config codebook count");
    check(info.num_frames == 3,
          "speech runtime plan derives frames from token count and config codebooks");
}

void test_prompt_injection_and_generation_settings() {
    trtmc::SpeechConfig cfg;
    cfg.num_codebooks = 16;
    cfg.audio_initial_token_id = 2048;
    cfg.text_initial_token_id = 32000;
    cfg.text_padding_id = 3;
    cfg.mimi_decode_codebooks = 8;

    check(!trtmc::should_run_text_prompt_injection(cfg),
          "speech runtime plan skips prompt injection without pretokenized ids");
    cfg.text_prompt_ids = {1, 2, 3};
    check(trtmc::should_run_text_prompt_injection(cfg),
          "speech runtime plan enables prompt injection when pretokenized ids exist");

    const trtmc::EncoderShapeInfo shape{32, 10};
    const auto settings = trtmc::make_speech_generation_settings(cfg, 4096, shape);
    check(settings.hidden == 4096, "speech runtime plan forwards hidden size");
    check(settings.num_cb == 16 && settings.stream_cb == 8,
          "speech runtime plan derives stream split from codebook count");
    check(settings.encode_codebooks == 32 && settings.num_frames == 10,
          "speech runtime plan forwards encoder shape");
    check(settings.audio_bos == 2048 && settings.text_bos == 32000,
          "speech runtime plan forwards BOS tokens");
    check(settings.text_pad_id == 3 && settings.mimi_cb == 8,
          "speech runtime plan forwards pad token and Mimi decode count");
}

void test_encoder_shape_uses_fallback_codebooks_when_cached_shape_is_invalid() {
    trtmc::SpeechConfig cfg;
    cfg.num_codebooks = 16;

    const auto info = trtmc::resolve_encoder_shape_with_fallback_codebooks(cfg, 0, 0, 24, 8);

    check(info.encode_codebooks == 8, "speech runtime plan uses fallback encoder codebooks");
    check(info.num_frames == 3,
          "speech runtime plan derives frames from fallback encoder codebooks");
}

} // namespace

int main() {
    test_encoder_shape_falls_back_to_config_when_cache_shape_is_invalid();
    test_encoder_shape_uses_config_when_no_cached_shape_exists();
    test_prompt_injection_and_generation_settings();
    test_encoder_shape_uses_fallback_codebooks_when_cached_shape_is_invalid();

    if (g_failures != 0) {
        std::cerr << g_failures << " speech runtime plan test(s) failed\n";
        return 1;
    }
    return 0;
}

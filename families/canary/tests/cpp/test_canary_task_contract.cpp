/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/canary/runtime/canary_config.h"
#include "families/canary/runtime/canary_request.h"
#include "families/canary/runtime/pipeline.h"
#include "trtmc/internal/audio.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void test_canary_config_and_validation() {
    trtmc::CanaryConfig model;
    model.max_target_positions = 448;
    model.max_source_positions = 1500;
    model.supported_languages = {"en", "es", "de", "fr"};
    model.language_token_ids = {10, 11, 12, 13};
    model.translation_requires_english = true;
    model.punctuation_token_id = 100;
    model.no_punctuation_token_id = 101;
    model.timestamp_token_id = 102;
    model.no_timestamp_token_id = 103;
    model.source_language_position = 0;
    model.target_language_position = 1;
    model.punctuation_position = 2;
    model.timestamp_position = 3;
    model.decoder_start_token_ids = {1, 2, 3, 4};

    // Valid transcription request
    trtmc::TranscriptionConfig valid_cfg;
    valid_cfg.source_language = "en";
    valid_cfg.target_language = "en";
    valid_cfg.task = trtmc::TranscriptionTask::kTranscribe;
    valid_cfg.max_output_tokens = 50;
    valid_cfg.beam_size = 1;
    valid_cfg.length_penalty = 0.0F;

    bool threw = false;
    try {
        trtmc::validate_canary_request(model, valid_cfg);
    } catch (...) {
        threw = true;
    }
    check(!threw, "valid transcription request accepted");

    // Invalid beam size > 32
    trtmc::TranscriptionConfig bad_beam = valid_cfg;
    bad_beam.beam_size = 64;
    threw = false;
    try {
        trtmc::validate_canary_request(model, bad_beam);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "beam_size > 32 rejected");

    // Negative length penalty
    trtmc::TranscriptionConfig bad_lp = valid_cfg;
    bad_lp.length_penalty = -1.0F;
    threw = false;
    try {
        trtmc::validate_canary_request(model, bad_lp);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "negative length_penalty rejected");

    // Unsupported language
    trtmc::TranscriptionConfig bad_lang = valid_cfg;
    bad_lang.source_language = "xx";
    bad_lang.target_language = "xx";
    threw = false;
    try {
        trtmc::validate_canary_request(model, bad_lang);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "unsupported language rejected");

    // Transcribe with mismatched languages
    trtmc::TranscriptionConfig mismatch = valid_cfg;
    mismatch.source_language = "en";
    mismatch.target_language = "es";
    threw = false;
    try {
        trtmc::validate_canary_request(model, mismatch);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "transcribe with different source and target rejected");

    // Translate with same languages
    trtmc::TranscriptionConfig same_trans = valid_cfg;
    same_trans.task = trtmc::TranscriptionTask::kTranslate;
    same_trans.source_language = "es";
    same_trans.target_language = "es";
    threw = false;
    try {
        trtmc::validate_canary_request(model, same_trans);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "translate with same source and target rejected");

    // Translate between two non-English languages when English is required
    trtmc::TranscriptionConfig no_en_trans = valid_cfg;
    no_en_trans.task = trtmc::TranscriptionTask::kTranslate;
    no_en_trans.source_language = "es";
    no_en_trans.target_language = "de";
    threw = false;
    try {
        trtmc::validate_canary_request(model, no_en_trans);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "non-English translation rejected when English required");
}

void test_task_names() {
    check(trtmc::internal::ISpeechTranscription::kTask == "speech_transcription",
          "ISpeechTranscription kTask is speech_transcription");
    check(trtmc::internal::ISpeechTranslation::kTask == "speech_translation",
          "ISpeechTranslation kTask is speech_translation");
    check(trtmc::internal::IBatchSpeechTranscription::kTask == "batch_speech_transcription",
          "IBatchSpeechTranscription kTask is batch_speech_transcription");
    check(trtmc::internal::IBatchSpeechTranslation::kTask == "batch_speech_translation",
          "IBatchSpeechTranslation kTask is batch_speech_translation");
}

} // namespace

int main() {
    test_canary_config_and_validation();
    test_task_names();

    if (g_failures > 0) {
        std::cerr << g_failures << " tests failed\n";
        return 1;
    }
    std::cout << "All canary task contract tests passed\n";
    return 0;
}

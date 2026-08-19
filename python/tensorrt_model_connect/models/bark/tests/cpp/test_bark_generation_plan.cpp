/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-03
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Bark generation plan: coarse/fine step derivation, window context, codec input
// Preconditions:  BarkConfig with valid codebook and token parameters
// Postconditions: Step counts, window plans, and codec transposition are correct
// =============================================================================

#include "bark_generation_plan.h"

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

trtmc::BarkConfig make_config() {
    trtmc::BarkConfig cfg;
    cfg.semantic_pad_token = 10000;
    cfg.coarse_semantic_pad_token = 12048;
    cfg.coarse_infer_token = 12050;
    cfg.max_coarse_history = 630;
    cfg.max_coarse_input_length = 4;
    cfg.sliding_window_len = 3;
    cfg.semantic_rate_hz = 50.0F;
    cfg.coarse_rate_hz = 100;
    cfg.n_coarse_codebooks = 2;
    return cfg;
}

void test_coarse_plan_derives_total_steps_and_windows() {
    const trtmc::BarkConfig cfg = make_config();
    const std::vector<int32_t> semantic_tokens = {1, 2, cfg.semantic_pad_token, 4, 5};

    const auto plan = trtmc::make_bark_coarse_plan(semantic_tokens, cfg);

    check(plan.total_steps == 20, "bark coarse plan computes total steps");
    check(plan.num_windows == 7, "bark coarse plan computes number of windows");
    check(plan.remapped_semantic[2] == cfg.coarse_semantic_pad_token,
          "bark coarse plan remaps semantic pad token");
}

void test_coarse_window_plan_builds_context_and_history() {
    const trtmc::BarkConfig cfg = make_config();
    const auto coarse_plan = trtmc::make_bark_coarse_plan({10, 11, 12, 13, 14}, cfg);
    const std::vector<int32_t> generated_tokens = {20001, 20002, 20003, 20004};

    const auto window = trtmc::make_bark_coarse_window_plan(coarse_plan, generated_tokens, cfg);

    check(window.start_generated_count == 4, "bark window plan records generated count");
    check(window.generated_this_window == 3, "bark window plan caps work by sliding window");
    check(window.input_tokens.size() == 9, "bark window plan builds semantic plus history tokens");
    check(window.input_tokens[4] == cfg.coarse_infer_token,
          "bark window plan inserts infer token after semantic context");
    check(window.input_tokens.back() == 20004, "bark window plan appends generated history");
}

void test_coarse_window_semantic_alignment_counts_codebooks() {
    trtmc::BarkConfig cfg = make_config();
    cfg.max_coarse_history = 4;
    const auto coarse_plan = trtmc::make_bark_coarse_plan({10, 11, 12, 13, 14}, cfg);
    const std::vector<int32_t> generated_tokens = {
        20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008,
    };

    const auto window = trtmc::make_bark_coarse_window_plan(coarse_plan, generated_tokens, cfg);

    check(window.input_tokens[0] == 11,
          "bark coarse window converts interleaved codebook tokens to semantic time");
    check(window.input_tokens[1] == 12 && window.input_tokens[3] == 14,
          "bark coarse window preserves HF semantic context alignment");
}

void test_codec_plan_prefers_fine_codes_when_available() {
    const std::vector<int32_t> fine_codes(16, 7);
    const std::vector<int32_t> coarse_tokens(12, 3);

    const auto fine_plan = trtmc::make_bark_codec_plan(fine_codes, true, coarse_tokens, 2);
    check(fine_plan.use_fine_codes, "bark codec plan prefers fine codes when available");
    check(fine_plan.frame_count == 2, "bark codec plan derives fine frame count");

    const auto coarse_plan = trtmc::make_bark_codec_plan({}, false, coarse_tokens, 2);
    check(!coarse_plan.use_fine_codes, "bark codec plan falls back to coarse tokens");
    check(coarse_plan.frame_count == 6, "bark codec plan derives coarse frame count");
    check(trtmc::bark_coarse_codebook_index(5, make_config()) == 1,
          "bark codebook helper alternates by coarse codebook count");
}

void test_fine_plan_and_code_initialization() {
    trtmc::BarkConfig cfg = make_config();
    cfg.codebook_size = 1024;
    cfg.semantic_vocab_size = 10000;
    cfg.fine_seq_length = 3;

    const std::vector<int32_t> coarse_tokens = {
        10000, 11024, 10001, 11025, 10002, 11026, 10003, 11027,
    };

    const auto fine_plan = trtmc::make_bark_fine_plan(cfg, coarse_tokens.size(), true, true);
    check(fine_plan.n_frames_raw == 4, "bark fine plan derives raw frame count");
    check(fine_plan.n_frames == 3, "bark fine plan clamps by configured fine seq length");
    check(fine_plan.actual_frames == 3, "bark fine plan derives actual frame count");
    check(fine_plan.should_run_trt, "bark fine plan enables TRT when resources exist");

    const auto codes = trtmc::initialize_bark_fine_codes(coarse_tokens, fine_plan.n_frames, cfg);
    check(codes.size() == 24, "bark fine code init allocates eight codebooks");
    check(codes[0] == 0 && codes[1] == 1 && codes[2] == 2,
          "bark fine code init maps codebook zero tokens");
    check(codes[3] == 0 && codes[4] == 1 && codes[5] == 2,
          "bark fine code init maps codebook one tokens");
}

void test_fine_sampling_policy_matches_hf() {
    trtmc::BarkConfig cfg = make_config();
    cfg.fine_temperature = 0.5F;
    check(trtmc::bark_fine_uses_sampling(cfg),
          "bark fine samples when HF temperature differs from one");

    cfg.fine_temperature = 1.0F;
    check(!trtmc::bark_fine_uses_sampling(cfg), "bark fine uses argmax at HF temperature one");

    cfg.fine_temperature = 0.5F;
    cfg.greedy = true;
    check(!trtmc::bark_fine_uses_sampling(cfg), "bark global greedy disables fine sampling");
}

void test_request_seed_overrides_session_seed() {
    check(trtmc::resolve_bark_seed(42, 0) == 0,
          "bark request seed overrides configured session seed");
    check(trtmc::resolve_bark_seed(42, -1) == 42,
          "bark configured session seed remains the fallback");
    check(trtmc::resolve_bark_seed(-1, -1) == -1,
          "bark remains nondeterministic when neither seed is configured");
}

void test_fine_input_embeddings_use_hf_padding_code() {
    constexpr int32_t n_frames = 1;
    constexpr int32_t actual_frames = 1;
    constexpr int32_t max_seq = 2;
    constexpr int32_t hidden = 1;
    constexpr int32_t codebook_size = 4;
    constexpr int32_t fine_cb_size = 5;
    const std::vector<int32_t> codes = {
        1, 2, 4, 4, 4, 4, 4, 4,
    };
    std::vector<float> embeddings(8 * fine_cb_size * hidden, 0.0F);
    for (int32_t cb = 0; cb < 8; ++cb) {
        for (int32_t code = 0; code < fine_cb_size; ++code) {
            embeddings[static_cast<std::size_t>(cb) * fine_cb_size + code] =
                static_cast<float>(10 * cb + code);
        }
    }
    const std::vector<float> positions = {100.0F, 200.0F};
    std::vector<float> output(max_seq * hidden, 0.0F);

    trtmc::build_bark_fine_input_embeddings(output, codes, 2, n_frames, actual_frames, max_seq,
                                            hidden, fine_cb_size, codebook_size, embeddings,
                                            positions);

    check(output[0] == 137.0F, "bark fine embeddings use generated codes for real frames");
    check(output[1] == 242.0F, "bark fine embeddings use HF padding code for padded frames");
}

void test_codec_input_builder_transposes_codebooks() {
    const std::vector<int32_t> codes_flat = {
        10, 11, 12, 20, 21, 22, 30, 31, 32,
    };
    const auto input_codes = trtmc::make_bark_codec_input_codes(codes_flat, 3, 3, 4, 5, 2);

    check(input_codes.size() == 20, "bark codec input builder uses padded target size");
    check(input_codes[0] == 10 && input_codes[1] == 11,
          "bark codec input builder copies first codebook frames");
    check(input_codes[5] == 20 && input_codes[6] == 21,
          "bark codec input builder transposes second codebook frames");
    check(input_codes[10] == 30 && input_codes[11] == 31,
          "bark codec input builder transposes third codebook frames");
    check(input_codes[15] == 0 && input_codes.back() == 0,
          "bark codec input builder leaves missing codebooks padded");
}

} // namespace

int main() {
    test_coarse_plan_derives_total_steps_and_windows();
    test_coarse_window_plan_builds_context_and_history();
    test_coarse_window_semantic_alignment_counts_codebooks();
    test_codec_plan_prefers_fine_codes_when_available();
    test_fine_plan_and_code_initialization();
    test_fine_sampling_policy_matches_hf();
    test_request_seed_overrides_session_seed();
    test_fine_input_embeddings_use_hf_padding_code();
    test_codec_input_builder_transposes_codebooks();

    if (g_failures != 0) {
        std::cerr << g_failures << " bark generation plan test(s) failed\n";
        return 1;
    }
    return 0;
}

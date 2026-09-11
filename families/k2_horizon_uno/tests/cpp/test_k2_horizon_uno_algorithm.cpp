/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/k2_horizon_uno/runtime/algorithm.h"
#include "families/k2_horizon_uno/runtime/kv_cache.h"
#include "trtmc/task.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

template <typename Callable>
bool rejects(Callable&& callable) {
    try {
        callable();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

template <typename Callable>
bool accepts(Callable&& callable) {
    try {
        callable();
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

void test_generation_modes_and_bounds() {
    trtmc::TextGenerationConfig config;
    trtmc::K2HorizonUnoResolvedGenerateConfig resolved;
    check(accepts([&] { resolved = trtmc::k2_horizon_uno_resolve_generate_config(config); }),
          "default generation config is accepted");
    check(resolved.mode == trtmc::K2HorizonUnoGenerationMode::kLinearPsiSpec,
          "auto selects linear Psi-Spec");
    check(resolved.block_length == 8, "auto uses publisher block length eight");

    for (const std::string mode :
         {"uno", "linear_psi_spec", "linear-psi-spec", "linear_spec_lora"}) {
        config.text_generation_mode = mode;
        config.block_length = 4;
        check(accepts([&] { resolved = trtmc::k2_horizon_uno_resolve_generate_config(config); }),
              "Uno generation aliases are accepted");
        check(resolved.mode == trtmc::K2HorizonUnoGenerationMode::kLinearPsiSpec,
              "Uno mode alias resolves to Psi-Spec");
        check(resolved.block_length == 4, "explicit block length is retained");
    }

    config = {};
    config.text_generation_mode = "ar";
    check(accepts([&] { resolved = trtmc::k2_horizon_uno_resolve_generate_config(config); }),
          "AR generation config is accepted");
    check(resolved.mode == trtmc::K2HorizonUnoGenerationMode::kAutoregressive,
          "AR control resolves independently");
    check(resolved.block_length == 1, "AR always uses one row");

    config = {};
    config.block_length = 9;
    check(rejects([&] { (void)trtmc::k2_horizon_uno_resolve_generate_config(config); }),
          "block length above engine profile is rejected");
    config.text_generation_mode = "ar";
    config.block_length = 2;
    check(rejects([&] { (void)trtmc::k2_horizon_uno_resolve_generate_config(config); }),
          "AR rejects multi-row block request");
    config = {};
    config.text_generation_mode = "diffusion";
    check(rejects([&] { (void)trtmc::k2_horizon_uno_resolve_generate_config(config); }),
          "unrelated diffusion mode is rejected");
}

void test_request_surface_fails_closed() {
    trtmc::TextGenerationConfig config;
    config.top_k = 2;
    check(rejects([&] { (void)trtmc::k2_horizon_uno_resolve_generate_config(config); }),
          "stochastic sampling is rejected");

    config = {};
    config.top_p = 0.9F;
    check(rejects([&] { (void)trtmc::k2_horizon_uno_resolve_generate_config(config); }),
          "active nucleus sampling overrides the default top-k one");

    config = {};
    config.top_k = 1;
    config.top_p = 1.0F;
    check(accepts([&] { (void)trtmc::k2_horizon_uno_resolve_generate_config(config); }),
          "default top-k one is accepted");

    config = {};
    config.temperature = 0.0F;
    config.top_k = 50;
    config.top_p = 0.9F;
    check(accepts([&] { (void)trtmc::k2_horizon_uno_resolve_generate_config(config); }),
          "temperature-zero greedy config is accepted");

    config = {};
    config.top_k = 50;
    config.top_p = 0.0F;
    check(accepts([&] { (void)trtmc::k2_horizon_uno_resolve_generate_config(config); }),
          "top-p-zero greedy config is accepted");

    config = {};
    config.lora_adapter_id = "external";
    check(rejects([&] { (void)trtmc::k2_horizon_uno_resolve_generate_config(config); }),
          "external LoRA selection is rejected");
    config = {};
    config.seed = 0;
    check(rejects([&] { (void)trtmc::k2_horizon_uno_resolve_generate_config(config); }),
          "deterministic noise rejects a silent seed override");
    config = {};
    config.confidence_threshold = 0.9F;
    check(rejects([&] { (void)trtmc::k2_horizon_uno_resolve_generate_config(config); }),
          "confidence-transfer decoding is rejected");
    config = {};
    config.use_chat_template = true;
    config.enable_thinking = false;
    check(rejects([&] { (void)trtmc::k2_horizon_uno_resolve_generate_config(config); }),
          "non-high chat reasoning is rejected");
}

void test_row_gate_is_exact() {
    const auto mask = trtmc::k2_horizon_uno_draft_lora_mask(8);
    check(mask.size() == 8 && mask.front() == 0.0F, "causal seed has LoRA disabled");
    bool noise_enabled = true;
    for (std::size_t index = 1; index < mask.size(); ++index)
        noise_enabled = noise_enabled && mask[index] == 1.0F;
    check(noise_enabled, "all noise rows have LoRA enabled");
}

void test_official_deterministic_uniform_golden() {
    // Golden values were computed independently with BigInt arithmetic from
    // ifm-ai/uno noise.py's explicit modulo-2^64 formula.
    const std::vector<int32_t> prompt{0, 250018, 2672, 200, 3749};
    const auto noise = trtmc::k2_horizon_uno_deterministic_uniform_noise(
        prompt, /*completion_token_count=*/5, /*seed_token=*/12345,
        /*total_sequence_length=*/10, /*count=*/7, /*vocab_size=*/250624);
    check(noise == std::vector<int32_t>({92094, 225933, 112692, 93512, 180152, 155240, 168818}),
          "deterministic_uniform matches the independent official-formula golden");

    const auto small = trtmc::k2_horizon_uno_deterministic_uniform_noise(
        {1, 2, 3}, /*completion_token_count=*/1, /*seed_token=*/7,
        /*total_sequence_length=*/4, /*count=*/3, /*vocab_size=*/32);
    check(small == std::vector<int32_t>({29, 20, 29}),
          "deterministic_uniform preserves modulo arithmetic at small vocabulary");
    check(rejects([] {
              (void)trtmc::k2_horizon_uno_deterministic_uniform_noise(
                  {1, 2, 3}, 1, 7, /*inconsistent total=*/5, 3, 32);
          }),
          "deterministic_uniform rejects inconsistent sequence accounting");
}

void test_greedy_verification_commits_correction_or_lookahead() {
    auto decision = trtmc::k2_horizon_uno_decide_greedy_commit({10, 20, 30}, {21, 31, 41});
    check(decision.tokens == std::vector<int32_t>({10, 21}),
          "first rejection commits clean plus correction");
    check(decision.accepted_draft_tokens == 0 && !decision.all_draft_tokens_accepted,
          "first rejection reports zero accepted future drafts");

    decision = trtmc::k2_horizon_uno_decide_greedy_commit({10, 20, 30}, {20, 31, 41});
    check(decision.tokens == std::vector<int32_t>({10, 20, 31}),
          "accepted prefix precedes verified correction");
    check(decision.accepted_draft_tokens == 1 && !decision.all_draft_tokens_accepted,
          "one accepted future draft is counted");

    decision = trtmc::k2_horizon_uno_decide_greedy_commit({10, 20, 30}, {20, 30, 40});
    check(decision.tokens == std::vector<int32_t>({10, 20, 30, 40}),
          "full acceptance appends verifier lookahead");
    check(decision.accepted_draft_tokens == 2 && decision.all_draft_tokens_accepted,
          "full future draft acceptance is reported");
}

void test_commit_truncation_and_retained_kv_frontier() {
    std::vector<int32_t> tokens{10, 20, 1, 40};
    trtmc::k2_horizon_uno_truncate_commit(tokens, 4, {1, 250019});
    check(tokens == std::vector<int32_t>({10, 20, 1}), "EOS is retained and terminates commit");
    check(trtmc::k2_horizon_uno_retained_kv_position(12, tokens.size()) == 15,
          "KV retains prior seed and all committed tokens except the last");

    tokens = {10, 20, 30};
    trtmc::k2_horizon_uno_truncate_commit(tokens, 2, {});
    check(tokens == std::vector<int32_t>({10, 20}), "remaining generation budget truncates commit");
}

void test_eos_set_validation() {
    trtmc::k2_horizon_uno_validate_eos_token_ids({1, 31}, 32);
    check(rejects([] { trtmc::k2_horizon_uno_validate_eos_token_ids({}, 32); }),
          "empty EOS set is rejected");
    check(rejects([] { trtmc::k2_horizon_uno_validate_eos_token_ids({32}, 32); }),
          "request EOS at vocabulary upper bound is rejected");
    check(rejects([] { trtmc::k2_horizon_uno_validate_eos_token_ids({1, 1}, 32); }),
          "duplicate EOS IDs are rejected");
}

void test_cache_cursor_rollback_contract() {
    trtmc::K2HorizonUnoCacheCursor cursor(32);
    cursor.advance(8);
    check(cursor.position() == 8, "draft block advances cursor");
    cursor.rollback(1);
    check(cursor.position() == 1, "draft noise rollback retains seed row");
    cursor.advance(3);
    check(cursor.position() == 4, "verify block advances from retained seed");
    cursor.rollback(3);
    check(cursor.position() == 3, "commit rollback leaves final token uncached");
    check(rejects([&] { cursor.rollback(4); }), "rollback cannot move frontier forward");
    cursor.reset();
    check(cursor.position() == 0, "cursor reset clears logical state");
    check(rejects([&] { cursor.advance(9); }), "cursor rejects blocks above engine maximum");

    // Three prompt rows plus five completion tokens fill capacity exactly.
    // The first completion seed remains uncached: draft reaches 7, then
    // retaining its seed and verifying four rows reaches 8, not 9.
    trtmc::K2HorizonUnoCacheCursor boundary(8);
    boundary.advance(3);
    boundary.advance(4);
    boundary.rollback(4);
    boundary.advance(4);
    check(boundary.position() == 8, "linear verify can reach exact KV capacity");
}

void test_argmax_and_receipt_contract() {
    const float logits[] = {0.0F, 2.0F, 2.0F, 1.0F, -1.0F, 0.0F, 4.0F, 3.0F};
    check(trtmc::k2_horizon_uno_argmax_rows(logits, 2, 4) == std::vector<int32_t>({1, 2}),
          "row argmax uses lowest-index ties");
    const float invalid[] = {0.0F, std::numeric_limits<float>::quiet_NaN()};
    check(rejects([&] { (void)trtmc::k2_horizon_uno_argmax_rows(invalid, 1, 2); }),
          "non-finite logits fail closed");

    check(trtmc::k2_horizon_uno_format_decode_receipt(
              trtmc::K2HorizonUnoGenerationMode::kLinearPsiSpec, 8, "deterministic_uniform", 6, 13,
              2) == "[trtmc.k2_horizon_uno.decode] mode=linear_spec_lora block_length=8 "
                    "noise_mode=deterministic_uniform forwards=6 committed_tokens=13 lookaheads=2",
          "decode receipt has one stable aggregate-only format");
}

} // namespace

int main() {
    test_generation_modes_and_bounds();
    test_request_surface_fails_closed();
    test_row_gate_is_exact();
    test_official_deterministic_uniform_golden();
    test_greedy_verification_commits_correction_or_lookahead();
    test_commit_truncation_and_retained_kv_frontier();
    test_eos_set_validation();
    test_cache_cursor_rollback_contract();
    test_argmax_and_receipt_contract();
    if (failures != 0) {
        std::cerr << failures << " K2-Horizon-Uno algorithm test(s) failed\n";
        return 1;
    }
    return 0;
}

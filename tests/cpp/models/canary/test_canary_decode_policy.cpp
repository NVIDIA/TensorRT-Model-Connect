/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-18
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Canary decode policy: EOT stopping, prefill/decode failure reporting, zero
// budget handling Preconditions:  Canary decode policy with configured EOT token Postconditions:
// Loop stops on EOT, failures reported at prefill and decode stages, zero budget handled
// =============================================================================

#include "runtime/models/canary/canary_decode_policy.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void test_decode_loop_stops_on_eot() {
    int step_calls = 0;
    const auto result = trtmc::run_canary_decode_loop(
        {11, 12}, 4, 99,
        [&step_calls](int32_t token, std::vector<float>& logits, std::string& error) {
            ++step_calls;
            error.clear();
            logits.assign(4, -5.0F);
            if (token == 12) {
                logits[1] = 7.0F;
            } else {
                logits[3] = 6.0F;
            }
            return true;
        },
        [](const std::vector<float>& logits) {
            int32_t best = 0;
            for (int32_t i = 1; i < static_cast<int32_t>(logits.size()); ++i) {
                if (logits[static_cast<std::size_t>(i)] > logits[static_cast<std::size_t>(best)]) {
                    best = i;
                }
            }
            return best == 1 ? 99 : 17;
        });

    check(!result.prefill_failed, "canary decode loop keeps successful prefill clear");
    check(!result.decode_failed, "canary decode loop stops cleanly on eot");
    check(result.output_ids == std::vector<int32_t>({99}),
          "canary decode loop emits eot token once and stops");
    check(step_calls == 2, "canary decode loop skips decode step after eot");
}

void test_decode_loop_reports_prefill_failure() {
    int step_calls = 0;
    const auto result = trtmc::run_canary_decode_loop(
        {21, 22}, 3, 99,
        [&step_calls](int32_t token, std::vector<float>& logits, std::string& error) {
            ++step_calls;
            if (token == 22) {
                error = "prefill-fail";
                return false;
            }
            logits.assign(2, 0.0F);
            return true;
        },
        [](const std::vector<float>&) { return 0; });

    check(result.prefill_failed, "canary decode loop reports prefill failure");
    check(!result.decode_failed,
          "canary decode loop does not mark decode failure on prefill error");
    check(result.error == "prefill-fail", "canary decode loop forwards prefill error");
    check(result.output_ids.empty(), "canary decode loop emits no tokens on prefill error");
    check(step_calls == 2, "canary decode loop stops at failing prefill token");
}

void test_decode_loop_reports_decode_failure_after_emitting_token() {
    int step_calls = 0;
    const auto result = trtmc::run_canary_decode_loop(
        {31}, 3, 99,
        [&step_calls](int32_t token, std::vector<float>& logits, std::string& error) {
            ++step_calls;
            if (step_calls == 2) {
                error = "decode-fail";
                return false;
            }
            logits.assign(3, -3.0F);
            logits[2] = 8.0F;
            return true;
        },
        [](const std::vector<float>&) { return 55; });

    check(!result.prefill_failed, "canary decode loop keeps prefill success clear");
    check(result.decode_failed, "canary decode loop reports decode failure");
    check(result.error == "decode-fail", "canary decode loop forwards decode error");
    check(result.output_ids == std::vector<int32_t>({55}),
          "canary decode loop preserves emitted token before decode failure");
    check(step_calls == 2, "canary decode loop stops immediately after decode failure");
}

void test_decode_loop_handles_zero_budget_and_empty_logits() {
    const auto zero_budget = trtmc::run_canary_decode_loop(
        {41}, 0, 99,
        [](int32_t, std::vector<float>& logits, std::string&) {
            logits.assign(1, 1.0F);
            return true;
        },
        [](const std::vector<float>&) { return 0; });
    check(zero_budget.output_ids.empty(),
          "canary decode loop emits nothing with zero token budget");

    const auto empty_logits = trtmc::run_canary_decode_loop(
        {}, 2, 99, [](int32_t, std::vector<float>&, std::string&) { return true; },
        [](const std::vector<float>&) { return 0; });
    check(empty_logits.output_ids.empty(), "canary decode loop emits nothing without logits");
    check(!empty_logits.prefill_failed && !empty_logits.decode_failed,
          "canary decode loop treats empty-logit case as clean no-op");
}

void test_beam_search_recovers_better_sequence_than_greedy_prefix() {
    int prefill_calls = 0;
    int advance_calls = 0;
    const auto result = trtmc::run_canary_beam_search(
        {9}, 2, 2, 2, 0.0F,
        [&prefill_calls](const std::vector<int32_t>& prefix, std::vector<float>& logits,
                         std::string&) {
            ++prefill_calls;
            if (prefix == std::vector<int32_t>({9}))
                logits = {0.0F, -0.1F, -10.0F};
            return true;
        },
        [&advance_calls](int32_t generation, int32_t parent_slot, int32_t child_slot, int32_t token,
                         std::vector<float>& logits, std::string&) {
            ++advance_calls;
            check(generation == 0, "Canary beam search advances only the required generation");
            check(parent_slot == 0, "Canary first beam generation branches from root state");
            check(child_slot >= 0 && child_slot < 2,
                  "Canary beam search assigns bounded child state slots");
            if (token == 1) {
                logits = {-10.0F, -10.0F, 2.0F};
            } else {
                logits = {0.0F, 0.0F, 0.0F};
            }
            return true;
        });
    check(!result.decode_failed, "Canary beam search succeeds");
    check(result.output_ids == std::vector<int32_t>({1, 2}),
          "Canary beam search selects highest sequence probability and stops on EOT");
    check(prefill_calls == 1, "Canary beam search prefills the prompt once");
    check(advance_calls == 2, "Canary beam search advances each retained branch once");
}

void test_beam_search_reports_prefill_failure() {
    const auto result = trtmc::run_canary_beam_search(
        {9}, 2, 2, 2, 1.0F,
        [](const std::vector<int32_t>&, std::vector<float>&, std::string& error) {
            error = "prefill-fail";
            return false;
        },
        [](int32_t, int32_t, int32_t, int32_t, std::vector<float>&, std::string&) { return true; });
    check(result.prefill_failed, "Canary beam search reports prompt prefill failure");
    check(!result.decode_failed, "Canary prefill failure is not a decode failure");
    check(result.error == "prefill-fail", "Canary beam search forwards prefill error");
}

void test_beam_search_reports_branch_advance_failure() {
    const auto result = trtmc::run_canary_beam_search(
        {9}, 2, 2, 2, 1.0F,
        [](const std::vector<int32_t>&, std::vector<float>& logits, std::string&) {
            logits = {1.0F, 0.0F, -10.0F};
            return true;
        },
        [](int32_t, int32_t, int32_t, int32_t, std::vector<float>&, std::string& error) {
            error = "advance-fail";
            return false;
        });
    check(result.decode_failed, "Canary beam search reports branch advance failure");
    check(result.error == "advance-fail", "Canary beam search forwards branch error");
}

void test_beam_search_applies_default_length_normalization() {
    check(trtmc::CanaryDefaultBeamLengthPenalty == 1.0F,
          "Canary beam search keeps average-log-probability scoring as its default");
    auto decode = [](float length_penalty) {
        return trtmc::run_canary_beam_search(
            {9}, 2, 1, 2, length_penalty,
            [](const std::vector<int32_t>&, std::vector<float>& logits, std::string&) {
                logits = {-0.1F, 0.0F};
                return true;
            },
            [](int32_t, int32_t, int32_t, int32_t, std::vector<float>& logits, std::string&) {
                logits = {-10.0F, 0.0F};
                return true;
            });
    };

    const auto raw = decode(0.0F);
    const auto normalized = decode(trtmc::CanaryDefaultBeamLengthPenalty);
    check(raw.output_ids == std::vector<int32_t>({1}),
          "Canary raw beam score favors the shorter sequence");
    check(normalized.output_ids == std::vector<int32_t>({0, 1}),
          "Canary length-normalized beam score can retain the longer sequence");
}

} // namespace

int main() {
    test_decode_loop_stops_on_eot();
    test_decode_loop_reports_prefill_failure();
    test_decode_loop_reports_decode_failure_after_emitting_token();
    test_decode_loop_handles_zero_budget_and_empty_logits();
    test_beam_search_recovers_better_sequence_than_greedy_prefix();
    test_beam_search_reports_prefill_failure();
    test_beam_search_reports_branch_advance_failure();
    test_beam_search_applies_default_length_normalization();

    if (g_failures != 0) {
        std::cerr << g_failures << " canary decode policy test(s) failed\n";
        return 1;
    }
    return 0;
}

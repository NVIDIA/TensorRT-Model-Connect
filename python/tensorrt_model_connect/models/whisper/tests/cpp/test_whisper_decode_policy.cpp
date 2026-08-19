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
// Intent:         Whisper decode policy: EOT stopping, prefill/decode failure reporting, zero
// budget handling Preconditions:  Whisper decode policy with configured EOT token Postconditions:
// Loop stops on EOT, failures reported at prefill and decode stages, zero budget handled
// =============================================================================

#include "whisper_decode_policy.h"

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
    const auto result = trtmc::run_whisper_decode_loop(
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

    check(!result.prefill_failed, "whisper decode loop keeps successful prefill clear");
    check(!result.decode_failed, "whisper decode loop stops cleanly on eot");
    check(result.output_ids == std::vector<int32_t>({99}),
          "whisper decode loop emits eot token once and stops");
    check(step_calls == 2, "whisper decode loop skips decode step after eot");
}

void test_decode_loop_reports_prefill_failure() {
    int step_calls = 0;
    const auto result = trtmc::run_whisper_decode_loop(
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

    check(result.prefill_failed, "whisper decode loop reports prefill failure");
    check(!result.decode_failed,
          "whisper decode loop does not mark decode failure on prefill error");
    check(result.error == "prefill-fail", "whisper decode loop forwards prefill error");
    check(result.output_ids.empty(), "whisper decode loop emits no tokens on prefill error");
    check(step_calls == 2, "whisper decode loop stops at failing prefill token");
}

void test_decode_loop_reports_decode_failure_after_emitting_token() {
    int step_calls = 0;
    const auto result = trtmc::run_whisper_decode_loop(
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

    check(!result.prefill_failed, "whisper decode loop keeps prefill success clear");
    check(result.decode_failed, "whisper decode loop reports decode failure");
    check(result.error == "decode-fail", "whisper decode loop forwards decode error");
    check(result.output_ids == std::vector<int32_t>({55}),
          "whisper decode loop preserves emitted token before decode failure");
    check(step_calls == 2, "whisper decode loop stops immediately after decode failure");
}

void test_decode_loop_handles_zero_budget_and_empty_logits() {
    const auto zero_budget = trtmc::run_whisper_decode_loop(
        {41}, 0, 99,
        [](int32_t, std::vector<float>& logits, std::string&) {
            logits.assign(1, 1.0F);
            return true;
        },
        [](const std::vector<float>&) { return 0; });
    check(zero_budget.output_ids.empty(),
          "whisper decode loop emits nothing with zero token budget");

    const auto empty_logits = trtmc::run_whisper_decode_loop(
        {}, 2, 99, [](int32_t, std::vector<float>&, std::string&) { return true; },
        [](const std::vector<float>&) { return 0; });
    check(empty_logits.output_ids.empty(), "whisper decode loop emits nothing without logits");
    check(!empty_logits.prefill_failed && !empty_logits.decode_failed,
          "whisper decode loop treats empty-logit case as clean no-op");
}

} // namespace

int main() {
    test_decode_loop_stops_on_eot();
    test_decode_loop_reports_prefill_failure();
    test_decode_loop_reports_decode_failure_after_emitting_token();
    test_decode_loop_handles_zero_budget_and_empty_logits();

    if (g_failures != 0) {
        std::cerr << g_failures << " whisper decode policy test(s) failed\n";
        return 1;
    }
    return 0;
}

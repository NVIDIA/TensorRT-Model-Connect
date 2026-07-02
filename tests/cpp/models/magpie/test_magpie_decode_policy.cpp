/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-05
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Magpie decode policy: greedy/sampling decode with audio range and EOS detection
// Preconditions:  Decode policy with configured audio token range and EOS token
// Postconditions: Greedy uses audio range, sampling uses sampler, EOS stops generation
// =============================================================================

#include "runtime/models/magpie/magpie_decode_policy.h"

#include <cmath>
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

void test_greedy_decode_uses_audio_range_but_still_detects_eos() {
    const int32_t num_cb = 2;
    const int32_t cb_size = trtmc::kMagpieEosToken + 1;
    std::vector<float> logits(static_cast<std::size_t>(num_cb) * cb_size, -10.0F);

    logits[10] = 4.0F;
    logits[trtmc::kMagpieEosToken] = 5.0F;

    const auto cb1_offset = static_cast<std::size_t>(cb_size);
    logits[cb1_offset + 7] = 3.0F;

    int sampler_calls = 0;
    const auto result =
        trtmc::decode_magpie_frame_codes(logits, num_cb, cb_size, true, 0.8F, 80,
                                         [&sampler_calls](const float*, int32_t, float, int32_t) {
                                             ++sampler_calls;
                                             return -1;
                                         });

    check(result.eos, "greedy decode detects eos from full argmax");
    check(result.frame_codes == std::vector<int32_t>({10, 7}),
          "greedy decode samples from audio range only");
    check(sampler_calls == 0, "greedy decode skips sampler");
}

void test_sampling_decode_uses_sampler() {
    const int32_t num_cb = 2;
    const int32_t cb_size = trtmc::kMagpieEosToken + 1;
    std::vector<float> logits(static_cast<std::size_t>(num_cb) * cb_size, -5.0F);
    logits[trtmc::kMagpieEosToken] = 2.0F;

    int sampler_calls = 0;
    const auto result = trtmc::decode_magpie_frame_codes(
        logits, num_cb, cb_size, false, 0.6F, 32,
        [&sampler_calls, cb_size](const float*, int32_t vocab_size, float temperature,
                                  int32_t top_k) {
            ++sampler_calls;
            check(vocab_size == cb_size, "sampler path uses full vocab for eos detection");
            check(std::fabs(temperature - 0.6F) < 1e-6F, "sampler path forwards temperature");
            check(top_k == 32, "sampler path forwards top-k");
            return 40 + sampler_calls;
        });

    check(result.eos, "sampling decode still detects eos");
    check(result.frame_codes == std::vector<int32_t>({41, 42}),
          "sampling decode returns sampler-selected ids");
    check(sampler_calls == 2, "sampling decode invokes sampler per codebook");
}

void test_sampling_decode_stops_on_sampled_eos() {
    const int32_t num_cb = 2;
    const int32_t cb_size = trtmc::kMagpieEosToken + 1;
    std::vector<float> logits(static_cast<std::size_t>(num_cb) * cb_size, -5.0F);
    logits[11] = 3.0F;
    logits[cb_size + 7] = 3.0F;

    int sampler_calls = 0;
    const auto result = trtmc::decode_magpie_frame_codes(
        logits, num_cb, cb_size, false, 0.6F, 32,
        [&sampler_calls](const float*, int32_t, float, int32_t) {
            ++sampler_calls;
            return sampler_calls == 1 ? trtmc::kMagpieEosToken : 17;
        });

    check(result.eos, "sampling decode treats sampled eos as stop");
    check(result.frame_codes == std::vector<int32_t>({0, 17}),
          "sampled eos frame is not emitted as audio");
    check(sampler_calls == 2, "sampled eos still evaluates all codebooks");
}

void test_stop_rule_helpers() {
    check(trtmc::should_run_magpie_periodic_check(15, 4, 16),
          "periodic helper triggers on configured interval");
    check(!trtmc::should_run_magpie_periodic_check(3, 4, 16), "periodic helper skips early frames");
    check(trtmc::should_stop_magpie_on_eos(true, trtmc::kMagpieMinFrames, trtmc::kMagpieMinFrames),
          "eos helper stops after minimum frames");
    check(!trtmc::should_stop_magpie_on_eos(true, 1, trtmc::kMagpieMinFrames),
          "eos helper keeps short outputs alive");
}

} // namespace

int main() {
    test_greedy_decode_uses_audio_range_but_still_detects_eos();
    test_sampling_decode_uses_sampler();
    test_sampling_decode_stops_on_sampled_eos();
    test_stop_rule_helpers();

    if (g_failures != 0) {
        std::cerr << g_failures << " Magpie decode policy test(s) failed\n";
        return 1;
    }
    return 0;
}

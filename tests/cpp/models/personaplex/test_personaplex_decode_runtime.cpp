/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-DEC-CPP-01
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-TRT-DEC-01
// Intent:         Argmax token selection, attention mask building
// Preconditions:  TRT headers available
// Postconditions: Argmax selects correct token, mask shape and values correct
// =============================================================================

// =============================================================================
// Test suite: PersonaPlex decode runtime CPU-side helper functions
// =============================================================================
//
// Purpose:
//   Validates the CPU-side utility functions from trt_decode_runtime.cpp that
//   support the autoregressive decoding loop: token selection (argmax, top-k)
//   and causal attention mask construction. These functions are exercised
//   without a GPU; they operate on CPU vectors and return CPU results.
//
// Dependencies:
//   - runtime/models/personaplex/decode_runtime.h (select_argmax_token, select_topk_tokens,
//                                        build_attention_mask)
//
// Approach:
//   All tests construct small input vectors, call the target function, and
//   verify the output against expected values. The test groups are:
//
//   1. Argmax tests — verify select_argmax_token returns the index of the
//      maximum logit value, including edge cases (single element, empty input,
//      all negatives, ties).
//
//   2. Top-k tests — verify select_topk_tokens returns the k indices with
//      highest logit values in descending order, including edge cases (k > size,
//      k=0, empty input).
//
//   3. Attention mask tests — verify build_attention_mask produces correct
//      causal masks for various cache occupancy levels (empty, partial, full,
//      with/without current-token slot).
//
// Environment:
//   No GPU execution; tests only exercise CPU logic.
// =============================================================================

#include "runtime/models/personaplex/decode_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

// -----------------------------------------------------------------------------
// Intention:  Verify basic argmax — selecting the index of the highest logit
//             from a 3-element vector.
// Setup:      logits = {0.1, 0.9, 0.3}. Maximum is at index 1.
// Mechanism:  Calls select_argmax_token and asserts the result is 1.
// -----------------------------------------------------------------------------
bool test_argmax_basic() {
    const std::vector<float> logits = {0.1F, 0.9F, 0.3F};
    const int32_t result = trtmc::personaplex_select_argmax_token(logits);
    if (result != 1) {
        std::cerr << "argmax_basic: got " << result << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify argmax with a single-element vector — the only valid
//             index is 0.
// Setup:      logits = {5.0}. Only one element, so index must be 0.
// Mechanism:  Calls select_argmax_token and asserts the result is 0.
// -----------------------------------------------------------------------------
bool test_argmax_single() {
    const std::vector<float> logits = {5.0F};
    const int32_t result = trtmc::personaplex_select_argmax_token(logits);
    if (result != 0) {
        std::cerr << "argmax_single: got " << result << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify argmax with an empty vector — should return 0 as a safe
//             default rather than crashing.
// Setup:      An empty logits vector.
// Mechanism:  Calls select_argmax_token and asserts the result is 0.
// -----------------------------------------------------------------------------
bool test_argmax_empty() {
    const std::vector<float> logits;
    const int32_t result = trtmc::personaplex_select_argmax_token(logits);
    if (result != 0) {
        std::cerr << "argmax_empty: got " << result << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify argmax when all logit values are negative — the function
//             should still return the index of the maximum (least negative).
// Setup:      logits = {-3.0, -1.0, -5.0, -2.0}. Maximum is -1.0 at index 1.
// Mechanism:  Calls select_argmax_token and asserts the result is 1.
// -----------------------------------------------------------------------------
bool test_argmax_all_negative() {
    const std::vector<float> logits = {-3.0F, -1.0F, -5.0F, -2.0F};
    const int32_t result = trtmc::personaplex_select_argmax_token(logits);
    if (result != 1) {
        std::cerr << "argmax_all_negative: got " << result << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify argmax tie-breaking behavior — when multiple elements
//             share the maximum value, the function should return the first
//             occurrence (consistent with std::max_element).
// Setup:      logits = {1.0, 5.0, 5.0, 2.0}. Tie at indices 1 and 2.
// Mechanism:  Calls select_argmax_token and asserts the result is 1 (the first
//             occurrence of the maximum).
// -----------------------------------------------------------------------------
bool test_argmax_tie() {
    // std::max_element returns first occurrence
    const std::vector<float> logits = {1.0F, 5.0F, 5.0F, 2.0F};
    const int32_t result = trtmc::personaplex_select_argmax_token(logits);
    if (result != 1) {
        std::cerr << "argmax_tie: got " << result << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify basic top-k selection — the 2 indices with the highest
//             logit values should be returned in descending order of value.
// Setup:      logits = {0.1, 0.9, 0.3, 0.7}. Top-2: index 1 (0.9), index 3
//             (0.7).
// Mechanism:  Calls select_topk_tokens with k=2, asserts the result has 2
//             elements, and verifies the indices are [1, 3].
// -----------------------------------------------------------------------------
bool test_topk_basic() {
    const std::vector<float> logits = {0.1F, 0.9F, 0.3F, 0.7F};
    const auto result = trtmc::personaplex_select_topk_tokens(logits, 2);
    if (result.size() != 2) {
        std::cerr << "topk_basic: size=" << result.size() << std::endl;
        return false;
    }
    // Top-2 by value: indices 1 (0.9) and 3 (0.7)
    if (result[0] != 1 || result[1] != 3) {
        std::cerr << "topk_basic: got [" << result[0] << ", " << result[1] << "]" << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify top-k behavior when k exceeds the vector size — the
//             function should return all elements (clamped to vector size)
//             rather than crashing.
// Setup:      logits = {0.1, 0.9} with k=5.
// Mechanism:  Calls select_topk_tokens and asserts the result has exactly 2
//             elements (the full vector).
// -----------------------------------------------------------------------------
bool test_topk_k_greater_than_size() {
    const std::vector<float> logits = {0.1F, 0.9F};
    const auto result = trtmc::personaplex_select_topk_tokens(logits, 5);
    if (result.size() != 2) {
        std::cerr << "topk_k_greater: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify top-k behavior with k=0 — should return an empty vector.
// Setup:      logits = {0.1, 0.9} with k=0.
// Mechanism:  Calls select_topk_tokens and asserts the result is empty.
// -----------------------------------------------------------------------------
bool test_topk_k_zero() {
    const std::vector<float> logits = {0.1F, 0.9F};
    const auto result = trtmc::personaplex_select_topk_tokens(logits, 0);
    if (!result.empty()) {
        std::cerr << "topk_k_zero: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify top-k behavior with an empty logits vector — should return
//             an empty result without crashing.
// Setup:      An empty logits vector with k=3.
// Mechanism:  Calls select_topk_tokens and asserts the result is empty.
// -----------------------------------------------------------------------------
bool test_topk_empty() {
    const std::vector<float> logits;
    const auto result = trtmc::personaplex_select_topk_tokens(logits, 3);
    if (!result.empty()) {
        std::cerr << "topk_empty: size=" << result.size() << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify sample_token_topk returns 0 for empty logits and exits
//             without touching RNG state.
// Setup:      Empty logits, any temperature/top_k, seeded rng_state.
// Mechanism:  Calls sample_token_topk and checks result=0 and unchanged RNG.
// -----------------------------------------------------------------------------
bool test_sample_topk_empty_logits_returns_zero() {
    const std::vector<float> logits;
    uint64_t rng_state = 0x123456789ABCDEF0ULL;
    const uint64_t rng_before = rng_state;
    const int32_t result = trtmc::personaplex_sample_token_topk(logits, 1.0F, 4, rng_state);
    if (result != 0) {
        std::cerr << "sample_topk_empty: got " << result << std::endl;
        return false;
    }
    if (rng_state != rng_before) {
        std::cerr << "sample_topk_empty: rng mutated from " << rng_before << " to " << rng_state
                  << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify near-zero temperature takes the argmax fallback path.
// Setup:      Non-empty logits, temperature < 1e-6, top_k > 1, seeded RNG.
// Mechanism:  Calls sample_token_topk and checks argmax index with unchanged
//             RNG state.
// -----------------------------------------------------------------------------
bool test_sample_topk_near_zero_temperature_uses_argmax() {
    const std::vector<float> logits = {0.2F, 2.7F, 1.5F, 2.1F};
    uint64_t rng_state = 0x0FEDCBA987654321ULL;
    const uint64_t rng_before = rng_state;
    const int32_t result = trtmc::personaplex_sample_token_topk(logits, 1.0e-8F, 4, rng_state);
    if (result != 1) {
        std::cerr << "sample_topk_near_zero_temp: got " << result << std::endl;
        return false;
    }
    if (rng_state != rng_before) {
        std::cerr << "sample_topk_near_zero_temp: rng mutated from " << rng_before << " to "
                  << rng_state << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify top_k=1 is deterministic argmax, independent of RNG seed.
// Setup:      Same logits sampled twice with different rng_state values.
// Mechanism:  Calls sample_token_topk(top_k=1) and checks both results are the
//             argmax index.
// -----------------------------------------------------------------------------
bool test_sample_topk_k_one_is_deterministic_argmax() {
    const std::vector<float> logits = {0.4F, 0.9F, 1.8F, -0.1F};
    uint64_t rng_a = 0x1111111111111111ULL;
    uint64_t rng_b = 0x2222222222222222ULL;
    const int32_t result_a = trtmc::personaplex_sample_token_topk(logits, 0.7F, 1, rng_a);
    const int32_t result_b = trtmc::personaplex_sample_token_topk(logits, 0.7F, 1, rng_b);
    if (result_a != 2 || result_b != 2) {
        std::cerr << "sample_topk_k1: got [" << result_a << ", " << result_b << "]" << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify top_k<=0 is clamped to 1, so sampling reduces to argmax.
// Setup:      Non-empty logits, temperature > 0, top_k is negative.
// Mechanism:  Calls sample_token_topk(top_k=-7) and checks argmax index.
// -----------------------------------------------------------------------------
bool test_sample_topk_non_positive_k_clamps_to_one() {
    const std::vector<float> logits = {-1.0F, 3.0F, 0.5F};
    uint64_t rng_state = 0xABCDEF0011223344ULL;
    const int32_t result = trtmc::personaplex_sample_token_topk(logits, 1.0F, -7, rng_state);
    if (result != 1) {
        std::cerr << "sample_topk_k_non_positive: got " << result << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify the sampling path mutates rng_state via xorshift updates.
// Setup:      Non-empty logits, temperature > 0, top_k > 1, non-zero seed.
// Mechanism:  Calls sample_token_topk and checks rng_state changed.
// -----------------------------------------------------------------------------
bool test_sample_topk_mutates_rng_state() {
    const std::vector<float> logits = {0.1F, 0.8F, 1.6F, -0.5F};
    uint64_t rng_state = 0x0123456789ABCDEFULL;
    const uint64_t rng_before = rng_state;
    const int32_t sampled = trtmc::personaplex_sample_token_topk(logits, 0.8F, 2, rng_state);
    if (sampled != 1 && sampled != 2) {
        std::cerr << "sample_topk_rng_mutates: sampled=" << sampled << std::endl;
        return false;
    }
    if (rng_state == rng_before) {
        std::cerr << "sample_topk_rng_mutates: rng unchanged=" << rng_state << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify non-positive attention-mask width returns an empty vector.
// Setup:      Cases where width=max_cache_length+(include_current?1:0) <= 0.
// Mechanism:  Calls build_attention_mask and checks both results are empty.
// -----------------------------------------------------------------------------
bool test_mask_non_positive_width_returns_empty() {
    const auto zero_width = trtmc::personaplex_build_attention_mask(0, 0, false);
    if (!zero_width.empty()) {
        std::cerr << "mask_non_positive_width: zero_width size=" << zero_width.size() << std::endl;
        return false;
    }

    const auto negative_width = trtmc::personaplex_build_attention_mask(2, -1, true);
    if (!negative_width.empty()) {
        std::cerr << "mask_non_positive_width: negative_width size=" << negative_width.size()
                  << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify negative cache_length with include_current=true keeps only
//             the appended current-token slot visible.
// Setup:      cache_length=-3, max_cache=3, include_current=true.
// Mechanism:  Calls build_attention_mask and checks mask[0..2] masked, mask[3]
//             (current slot) visible.
// -----------------------------------------------------------------------------
bool test_mask_negative_cache_with_current_slot() {
    const auto mask = trtmc::personaplex_build_attention_mask(-3, 3, true);
    if (mask.size() != 4) {
        std::cerr << "mask_negative_cache_current: size=" << mask.size() << std::endl;
        return false;
    }
    for (int32_t i = 0; i < 3; ++i) {
        if (mask[static_cast<std::size_t>(i)] >= 0.0F) {
            std::cerr << "mask_negative_cache_current: [" << i
                      << "]=" << mask[static_cast<std::size_t>(i)] << std::endl;
            return false;
        }
    }
    if (mask[3] != 0.0F) {
        std::cerr << "mask_negative_cache_current: [3]=" << mask[3] << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify the causal attention mask when the cache is empty
//             (cache_length=0) and include_current is false. Only the first
//             position should be visible (0.0); the rest should be masked
//             (large negative).
// Setup:      cache_length=0, max_cache=4, include_current=false.
// Mechanism:  Calls build_attention_mask and asserts: size is 4, mask[0]=0.0
//             (visible), mask[1..3] < 0.0 (masked).
// -----------------------------------------------------------------------------
bool test_mask_cache0_no_current() {
    // cache_length=0, max=4, include_current=false
    // First position visible (else clause), rest masked
    const auto mask = trtmc::personaplex_build_attention_mask(0, 4, false);
    if (mask.size() != 4) {
        std::cerr << "mask_cache0: size=" << mask.size() << std::endl;
        return false;
    }
    if (mask[0] != 0.0F) {
        std::cerr << "mask_cache0: [0]=" << mask[0] << std::endl;
        return false;
    }
    for (int i = 1; i < 4; ++i) {
        if (mask[static_cast<std::size_t>(i)] >= 0.0F) {
            std::cerr << "mask_cache0: [" << i << "]=" << mask[static_cast<std::size_t>(i)]
                      << std::endl;
            return false;
        }
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify the causal attention mask when 3 of 4 cache slots are
//             occupied and include_current is false. The first 3 positions
//             should be visible; the last should be masked.
// Setup:      cache_length=3, max_cache=4, include_current=false.
// Mechanism:  Calls build_attention_mask and asserts: size is 4, mask[0..2]=0.0,
//             mask[3] < 0.0.
// -----------------------------------------------------------------------------
bool test_mask_cache3_no_current() {
    // cache_length=3, max=4, include_current=false -> 3 visible, 1 masked
    const auto mask = trtmc::personaplex_build_attention_mask(3, 4, false);
    if (mask.size() != 4) {
        std::cerr << "mask_cache3: size=" << mask.size() << std::endl;
        return false;
    }
    for (int i = 0; i < 3; ++i) {
        if (mask[static_cast<std::size_t>(i)] != 0.0F) {
            std::cerr << "mask_cache3: [" << i << "]=" << mask[static_cast<std::size_t>(i)]
                      << std::endl;
            return false;
        }
    }
    if (mask[3] >= 0.0F) {
        std::cerr << "mask_cache3: [3]=" << mask[3] << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify that include_current=true appends an extra slot to the
//             mask (for the current token being decoded), and that this slot is
//             visible (0.0).
// Setup:      cache_length=0, max_cache=4, include_current=true.
// Mechanism:  Calls build_attention_mask and asserts: size is 5 (4 cache + 1
//             current), and mask[4] (the current-token slot) equals 0.0.
// -----------------------------------------------------------------------------
bool test_mask_with_current_slot() {
    // cache_length=0, max=4, include_current=true -> width=5, last slot visible
    const auto mask = trtmc::personaplex_build_attention_mask(0, 4, true);
    if (mask.size() != 5) {
        std::cerr << "mask_current: size=" << mask.size() << std::endl;
        return false;
    }
    // Last slot (current) should be visible
    if (mask[4] != 0.0F) {
        std::cerr << "mask_current: [4]=" << mask[4] << std::endl;
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Intention:  Verify the attention mask when cache_length >= max_cache_length —
//             all positions should be visible (the cache is fully occupied).
// Setup:      cache_length=5, max_cache=4, include_current=false. The cache
//             length exceeds max, so all 4 positions should be unmasked.
// Mechanism:  Calls build_attention_mask and asserts: size is 4, every element
//             equals 0.0.
// -----------------------------------------------------------------------------
bool test_mask_full_cache() {
    // cache_length >= max -> all positions visible
    const auto mask = trtmc::personaplex_build_attention_mask(5, 4, false);
    if (mask.size() != 4) {
        std::cerr << "mask_full: size=" << mask.size() << std::endl;
        return false;
    }
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (mask[i] != 0.0F) {
            std::cerr << "mask_full: [" << i << "]=" << mask[i] << std::endl;
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    bool all_passed = true;
    std::cout << "test_personaplex_decode_runtime:" << std::endl;

    const auto run = [&](const char* name, bool (*fn)()) {
        const bool ok = fn();
        std::cout << "  " << name << ": " << (ok ? "PASS" : "FAIL") << std::endl;
        all_passed &= ok;
    };

    run("argmax_basic", test_argmax_basic);
    run("argmax_single", test_argmax_single);
    run("argmax_empty", test_argmax_empty);
    run("argmax_all_negative", test_argmax_all_negative);
    run("argmax_tie", test_argmax_tie);
    run("topk_basic", test_topk_basic);
    run("topk_k_greater", test_topk_k_greater_than_size);
    run("topk_k_zero", test_topk_k_zero);
    run("topk_empty", test_topk_empty);
    run("sample_topk_empty_logits_zero", test_sample_topk_empty_logits_returns_zero);
    run("sample_topk_near_zero_temp_argmax", test_sample_topk_near_zero_temperature_uses_argmax);
    run("sample_topk_k_one_argmax", test_sample_topk_k_one_is_deterministic_argmax);
    run("sample_topk_non_positive_k_clamps", test_sample_topk_non_positive_k_clamps_to_one);
    run("sample_topk_rng_mutates", test_sample_topk_mutates_rng_state);
    run("mask_non_positive_width_empty", test_mask_non_positive_width_returns_empty);
    run("mask_negative_cache_current_slot", test_mask_negative_cache_with_current_slot);
    run("mask_cache0_no_current", test_mask_cache0_no_current);
    run("mask_cache3_no_current", test_mask_cache3_no_current);
    run("mask_with_current_slot", test_mask_with_current_slot);
    run("mask_full_cache", test_mask_full_cache);

    if (all_passed) {
        std::cout << "test_personaplex_decode_runtime passed" << std::endl;
        return 0;
    }
    std::cerr << "test_personaplex_decode_runtime FAILED" << std::endl;
    return 1;
}

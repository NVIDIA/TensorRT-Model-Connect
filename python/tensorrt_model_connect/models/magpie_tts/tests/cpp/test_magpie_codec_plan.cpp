/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-04
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Magpie codec plan: size computation, sample validation, input transposition
// Preconditions:  MagpieConfig with valid codec parameters
// Postconditions: Sizes and transposed inputs match expected values
// =============================================================================

#include "magpie_codec_plan.h"

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

void test_codec_plan_computes_sizes_and_valid_samples() {
    const auto plan = trtmc::make_magpie_codec_plan(6, 8, 4);
    check(plan.max_codec_frames == 4, "codec plan max frames");
    check(plan.padded_frames == 4, "codec plan padded frames");
    check(plan.input_len == 4, "codec plan input len");
    check(plan.input_elems == 32, "codec plan input elems");
    check(plan.valid_samples == 4096, "codec plan valid samples");
}

void test_codec_input_transposes_and_sanitizes_codes() {
    const auto plan = trtmc::make_magpie_codec_plan(3, 2, 4);
    const std::vector<int32_t> codes = {
        1, trtmc::kMagpieCodecInputLimit, 3, 4, 5, 6,
    };
    const auto codec_input = trtmc::build_magpie_codec_input(codes, 2, plan);
    const std::vector<int32_t> expected = {
        1, 3, 5, 0, 0, 4, 6, 0,
    };
    check(codec_input == expected, "codec input is transposed and sanitized");
}

} // namespace

int main() {
    test_codec_plan_computes_sizes_and_valid_samples();
    test_codec_input_transposes_and_sanitizes_codes();

    if (g_failures != 0) {
        std::cerr << g_failures << " magpie codec plan test(s) failed\n";
        return 1;
    }
    return 0;
}

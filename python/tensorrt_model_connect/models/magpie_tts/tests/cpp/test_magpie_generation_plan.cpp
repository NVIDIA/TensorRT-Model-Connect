/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-08
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-01
// Intent:         Magpie request generation settings override session defaults
// Preconditions:  Session and request seed values
// Postconditions: Request seed wins when present; session seed remains the fallback
// =============================================================================

#include "magpie_generation_plan.h"

#include <iostream>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void test_request_seed_overrides_session_seed() {
    check(trtmc::resolve_magpie_seed(42, 0) == 0,
          "magpie request seed overrides configured session seed");
    check(trtmc::resolve_magpie_seed(42, -1) == 42,
          "magpie configured session seed remains the fallback");
    check(trtmc::resolve_magpie_seed(-1, -1) == -1,
          "magpie remains nondeterministic when neither seed is configured");
}

} // namespace

int main() {
    test_request_seed_overrides_session_seed();
    if (g_failures != 0) {
        std::cerr << g_failures << " magpie generation plan test(s) failed\n";
        return 1;
    }
    return 0;
}

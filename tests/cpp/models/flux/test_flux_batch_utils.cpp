/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Unit test for Flux-owned image batch chunk planning.

#include "runtime/models/flux/flux_batch_utils.h"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

void test_flux_chunk_planning() {
    auto even = trtmc::flux_batch::plan_chunks(/*total=*/8, /*cap=*/4);
    check(even == std::vector<int>{4, 4}, "flux 8/4 splits evenly");
    auto rem = trtmc::flux_batch::plan_chunks(/*total=*/9, /*cap=*/4);
    check(rem == std::vector<int>{4, 4, 1}, "flux 9/4 leaves remainder as final chunk");

    bool threw = false;
    try {
        (void)trtmc::flux_batch::plan_chunks(/*total=*/0, /*cap=*/4);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "flux rejects zero total");
}

} // namespace

int main() {
    test_flux_chunk_planning();
    if (failures > 0) {
        std::cerr << failures << " Flux batch utility test(s) FAILED\n";
        return 1;
    }
    return 0;
}

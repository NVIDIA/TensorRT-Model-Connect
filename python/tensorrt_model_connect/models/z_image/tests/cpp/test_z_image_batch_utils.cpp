/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Unit test for Z-Image-owned per-sample seed and chunk-planning helpers.

#include "z_image_batch_utils.h"

#include <cstdint>
#include <iostream>
#include <set>
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

void test_z_image_batch_contracts() {
    auto a = trtmc::z_image_batch::derive_per_sample_seeds(/*global_seed=*/42, /*count=*/4);
    auto b = trtmc::z_image_batch::derive_per_sample_seeds(/*global_seed=*/42, /*count=*/4);
    check(a == b && a.size() == 4, "z-image global seed reproduces same sequence of 4");
    std::set<std::uint32_t> unique(a.begin(), a.end());
    check(unique.size() == 4, "z-image per-sample seeds are distinct");

    bool seed_threw = false;
    try {
        (void)trtmc::z_image_batch::derive_per_sample_seeds(/*global_seed=*/42, /*count=*/0);
    } catch (const std::invalid_argument&) {
        seed_threw = true;
    }
    check(seed_threw, "z-image rejects zero seed count");

    auto even = trtmc::z_image_batch::plan_chunks(/*total=*/8, /*cap=*/4);
    check(even == std::vector<int>{4, 4}, "z-image 8/4 splits evenly");
    auto rem = trtmc::z_image_batch::plan_chunks(/*total=*/9, /*cap=*/4);
    check(rem == std::vector<int>{4, 4, 1}, "z-image 9/4 leaves remainder as final chunk");
}

} // namespace

int main() {
    test_z_image_batch_contracts();
    if (failures > 0) {
        std::cerr << failures << " Z-Image batch utility test(s) FAILED\n";
        return 1;
    }
    return 0;
}

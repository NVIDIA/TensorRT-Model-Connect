/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Unit test for Qwen Image-owned image batch chunk planning.

#include "runtime/models/qwen_image/qwen_image_batch_utils.h"

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

void test_qwen_image_chunk_planning() {
    auto even = trtmc::qwen_image_batch::plan_chunks(/*total=*/8, /*cap=*/4);
    check(even == std::vector<int>{4, 4}, "qwen image 8/4 splits evenly");
    auto rem = trtmc::qwen_image_batch::plan_chunks(/*total=*/9, /*cap=*/4);
    check(rem == std::vector<int>{4, 4, 1}, "qwen image 9/4 leaves remainder as final chunk");

    bool threw = false;
    try {
        (void)trtmc::qwen_image_batch::plan_chunks(/*total=*/8, /*cap=*/0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "qwen image rejects zero cap");
}

} // namespace

int main() {
    test_qwen_image_chunk_planning();
    if (failures > 0) {
        std::cerr << failures << " Qwen Image batch utility test(s) FAILED\n";
        return 1;
    }
    return 0;
}

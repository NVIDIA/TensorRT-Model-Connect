/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dataset_answer.h"

#include <iostream>
#include <optional>
#include <string>

namespace {

using trtmc::examples::dataset_benchmark::extract_answer;

int failures = 0;

void check(const std::string& text, const std::optional<std::string>& expected, const char* name) {
    const auto actual = extract_answer(text);
    if (actual != expected) {
        std::cerr << "FAIL: " << name << " expected=" << expected.value_or("<none>")
                  << " actual=" << actual.value_or("<none>") << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    check("Final answer: 3,000", "3000", "final answer keeps the complete comma number");
    check(R"(The derivation gives \boxed{1,234}. Final answer: 9)", "1234",
          "boxed answer has priority");
    check(R"(The invalid result is \boxed{not-a-number}. Final answer: 9)", "9",
          "invalid boxed content continues to the next rule");
    check("After checking, the answer is -42.", "-42", "answer phrase");
    check("Thus, the total is 1,024.", "1024", "discourse quantity");
    check("The requested value is m + n = 77.", "77", "m plus n phrase");
    check("The intermediate values are 2, 5, and 19.", "19", "last integer");
    check("There is no numeric answer.", std::nullopt, "missing answer");
    return failures;
}

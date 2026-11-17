/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/hunyuan_vl/runtime/request.h"

#include <iostream>
#include <limits>
#include <vector>

namespace {
using trtmc::internal::ConfigEntry;
using trtmc::internal::ConfigError;
int failures = 0;
void expect(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << '\n';
        ++failures;
    }
}
bool penalty_rejected(double penalty) {
    const std::vector<ConfigEntry> supplied{{"repetition_penalty", penalty}};
    try {
        trtmc::hunyuan_vl::read_generation({supplied.data(), supplied.size()});
    } catch (const ConfigError&) {
        return true;
    }
    return false;
}
bool penalty_overflows(std::vector<float> logits, const std::vector<bool>& seen, float penalty) {
    try {
        trtmc::hunyuan_vl::apply_repetition_penalty(logits, seen, penalty);
    } catch (const ConfigError&) {
        return true;
    }
    return false;
}
} // namespace

int main() {
    constexpr float smallest = std::numeric_limits<float>::min();
    constexpr float largest = std::numeric_limits<float>::max();
    // The penalty must stay a positive normal float after the conversion to float.
    expect(trtmc::hunyuan_vl::read_generation({}).repetition_penalty == 1.0, "default penalty");
    for (const double accepted : {1.05, 0.5, double(smallest), double(largest)})
        expect(!penalty_rejected(accepted), "positive normal float penalty accepted");
    for (const double rejected :
         {0.0, -1.0, 1e-300, 1e-40, 1e300, std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::quiet_NaN()})
        expect(penalty_rejected(rejected), "penalty outside the positive normal floats rejected");

    // Negative logits are multiplied, the others divided; unseen tokens keep their logits.
    std::vector<float> logits{2.0F, -2.0F, 3.0F, 0.0F};
    trtmc::hunyuan_vl::apply_repetition_penalty(logits, {true, true, false, true}, 2.0F);
    expect(logits == std::vector<float>{1.0F, -4.0F, 3.0F, 0.0F}, "penalty direction");
    expect(penalty_overflows({10.0F}, {true}, smallest), "division overflow rejected");
    expect(penalty_overflows({-10.0F}, {true}, largest), "multiplication overflow rejected");
    expect(!penalty_overflows({10.0F}, {false}, smallest), "unseen tokens are not penalized");
    return failures == 0 ? 0 : 1;
}

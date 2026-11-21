/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The expected bins are numpy.digitize(x, linspace(-1, 1, 257)[:-1]) - 1, which is what
// LeRobot's Pi05PrepareStateTokenizerProcessorStep computes.

#include "families/lerobot_pi05/runtime/prompt.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template <class Function>
bool throws_invalid(Function function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    using namespace trtmc::lerobot_pi05;

    const struct {
        float value;
        int bin;
    } bins[] = {{-5.0F, -1},       {-1.5F, -1}, {-1.0F, 0},  {-0.9999F, 0},
                {-0.5F, 64},       {0.0F, 128}, {0.5F, 192}, {0.99F, 254},
                {0.9921875F, 255}, {1.0F, 255}, {7.0F, 255}};
    for (const auto& expected : bins)
        check(discretize_state_value(expected.value) == expected.bin,
              "bin for " + std::to_string(expected.value));

    check(clean_prompt("  pick_up the\ncup \t") == "pick up the cup", "strip and replace");
    check(clean_prompt("\xE3\x80\x80 move \xE3\x80\x80") == "move",
          "unicode whitespace is stripped");
    check(clean_prompt("   \n\t ").empty(), "whitespace-only prompt is empty");
    check(clean_prompt("a__b") == "a  b", "every underscore is replaced");
    check(throws_invalid([] { clean_prompt("bad \xFF byte"); }), "invalid UTF-8 is rejected");

    check(format_prompt("pick_up the cup", {0.0F, -1.0F, 1.0F}) ==
              "Task: pick up the cup, State: 128 0 255;\nAction: ",
          "prompt layout");
    check(format_prompt("go", {}) == "Task: go, State: ;\nAction: ", "empty state");

    if (failures == 0)
        std::cout << "lerobot_pi05 prompt tests passed\n";
    return failures == 0 ? 0 : 1;
}

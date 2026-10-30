/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Formalisation follows LeRobot's prepare_n1_7_language_batch (str.lower() and a filter keeping
// isalnum(), isspace() and '_'); positions follow Qwen3VLModel.get_rope_index.

#include "families/lerobot_groot/runtime/prompt.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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
    using namespace trtmc::lerobot_groot;

    check(formalize_instruction("Pick up the Black bowl.", true) == "pick up the black bowl",
          "lower and drop '.'");
    check(formalize_instruction("a_b-c, (d)!", true) == "a_bc d", "keep alnum, space and '_'");
    check(formalize_instruction("tab\tnew\nline\x1c", true) == "tab\tnew\nline\x1c",
          "Python isspace() set");
    check(formalize_instruction("", true) == "perform the task", "LeRobot default for empty text");
    check(formalize_instruction("!!!", true) == "Perform the task.",
          "nothing left: default verbatim");
    check(formalize_instruction("", false) == "Perform the task.", "default kept verbatim");
    check(formalize_instruction("Keep. Case!", false) == "Keep. Case!", "no formalisation");
    check(throws_invalid([] { formalize_instruction("caf\xc3\xa9", true); }),
          "non-ASCII is rejected");
    check(throws_invalid([] { formalize_instruction("thin\xe2\x80\x89space", false); }),
          "non-ASCII is rejected without formalisation too");

    TemplateTokens tokens{1, 2, 3, 4, 9, {5, 6}, {6}};
    const auto prompt = build_prompt({7, 8}, 2, 2, tokens);
    const std::vector<std::int32_t> ids = {1, 5, 6, 3, 9, 9, 9, 9, 4, 3, 9, 9, 9, 9, 4, 7, 8, 2, 6};
    check(prompt.token_ids == ids, "template layout");
    check(prompt.image_index ==
              std::vector<std::int32_t>{0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 7, 0, 0, 0, 0, 0},
          "image token numbers");
    check(prompt.image_mask ==
              std::vector<std::int32_t>{0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0},
          "image mask");
    // Text counts up; an image keeps its start for t, counts rows for h and columns for w, then
    // the count moves on by the grid side.
    const std::vector<std::int32_t> t = {0, 1, 2, 3, 4,  4,  4,  4,  6, 7,
                                         8, 8, 8, 8, 10, 11, 12, 13, 14};
    const std::vector<std::int32_t> h = {0, 1, 2, 3, 4,  4,  5,  5,  6, 7,
                                         8, 8, 9, 9, 10, 11, 12, 13, 14};
    const std::vector<std::int32_t> w = {0, 1, 2, 3, 4,  5,  4,  5,  6, 7,
                                         8, 9, 8, 9, 10, 11, 12, 13, 14};
    std::vector<std::int32_t> positions(t);
    positions.insert(positions.end(), h.begin(), h.end());
    positions.insert(positions.end(), w.begin(), w.end());
    check(prompt.positions == positions, "multimodal rotary positions");

    if (failures == 0)
        std::cout << "lerobot_groot prompt tests passed\n";
    return failures == 0 ? 0 : 1;
}

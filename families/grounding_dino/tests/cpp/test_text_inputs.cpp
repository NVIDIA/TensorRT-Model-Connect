/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The phrase structure is derived from the token ids alone, and getting it
// wrong produces a same-shaped wrong answer: letting the whole prompt attend
// to itself scored 0.84 at the first text layer. These cases pin the exact
// pattern measured from the reference for "a cat. a remote control.".

#include "families/grounding_dino/runtime/text_inputs.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

// [CLS] a cat . a remote control . [SEP]
const std::vector<std::int32_t> kPrompt{101, 1037, 4937, 1012, 1037, 6556, 2491, 1012, 102};

void test_position_ids() {
    std::vector<std::uint8_t> block;
    std::vector<std::int32_t> positions;
    trtmc::grounding_dino_phrase_structure(kPrompt, block, positions);
    const std::vector<std::int32_t> expected{0, 0, 1, 0, 0, 1, 2, 0, 0};
    check(positions == expected, "per-phrase position ids restart at each delimiter");
}

void test_block_mask_is_phrase_local() {
    std::vector<std::uint8_t> block;
    std::vector<std::int32_t> positions;
    trtmc::grounding_dino_phrase_structure(kPrompt, block, positions);
    const auto length = static_cast<std::int32_t>(kPrompt.size());
    const auto at = [&](int row, int column) {
        return block[static_cast<std::size_t>(row) * length + column] != 0;
    };
    // "a cat" (1,2) is one phrase; "a remote control" (4,5,6) is another.
    check(at(1, 2) && at(2, 1), "tokens inside a phrase attend to each other");
    check(!at(1, 5) && !at(5, 1), "tokens do not attend across phrases");
    check(at(4, 6) && at(6, 4), "the second phrase is connected");
    check(!at(2, 4), "the first phrase does not reach the second");
    // [CLS] and [SEP] attend to themselves only.
    for (int column = 0; column < length; ++column) {
        check(at(0, column) == (column == 0), "[CLS] attends to itself only");
        check(at(length - 1, column) == (column == length - 1), "[SEP] attends to itself only");
    }
}

void test_padding_is_masked_but_never_fully() {
    const auto inputs = trtmc::grounding_dino_text_inputs(kPrompt, 16, 8);
    check(inputs.real_length == 9, "real length is the untruncated prompt");
    for (int i = 0; i < 16; ++i) {
        const bool real = i < 9;
        check((inputs.key_mask[static_cast<std::size_t>(i)] == 0.0F) == real,
              "padded keys are masked and real keys are not");
        // A fully masked softmax row is NaN, so the diagonal always stays open.
        check(inputs.self_attention_mask[static_cast<std::size_t>(i) * 16 + i] == 0.0F,
              "every row keeps its own diagonal");
    }
    check(inputs.self_attention_mask[9 * 16 + 0] == trtmc::kGroundingDinoMaskFill,
          "a padded row does not reach the prompt");
    check(inputs.self_attention_mask[0 * 16 + 9] == trtmc::kGroundingDinoMaskFill,
          "the prompt does not reach a padded column");
}

void test_sine_positions_interleave() {
    const std::vector<std::int32_t> positions{0, 1};
    const auto embedding = trtmc::grounding_dino_sine_positions(positions, 4);
    check(embedding.size() == 8, "one row of features per position");
    // Position zero: sin(0) = 0 in the even slots, cos(0) = 1 in the odd ones.
    check(embedding[0] == 0.0F && embedding[2] == 0.0F, "even slots are sine");
    check(embedding[1] == 1.0F && embedding[3] == 1.0F, "odd slots are cosine");
    const float two_pi = 6.2831853F;
    check(std::fabs(embedding[4] - std::sin(two_pi)) < 1e-5F,
          "the first frequency is the raw angle");
}

void test_rejects_an_overlong_prompt() {
    bool threw = false;
    try {
        trtmc::grounding_dino_text_inputs(kPrompt, 4, 8);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a prompt longer than max_text_len is rejected, not truncated");
}

} // namespace

int main() {
    test_position_ids();
    test_block_mask_is_phrase_local();
    test_padding_is_masked_but_never_fully();
    test_sine_positions_interleave();
    test_rejects_an_overlong_prompt();
    if (failures != 0) {
        std::fprintf(stderr, "%d grounding_dino text-input checks failed\n", failures);
        return EXIT_FAILURE;
    }
    std::printf("grounding_dino text inputs: all checks passed\n");
    return EXIT_SUCCESS;
}

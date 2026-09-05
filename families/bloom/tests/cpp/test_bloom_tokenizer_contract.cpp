/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/bloom/runtime/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void check_ids(const std::vector<int32_t>& actual, const std::vector<int32_t>& expected,
               const char* name) {
    if (actual == expected)
        return;
    std::cerr << "FAIL: " << name << " got";
    for (const int32_t token : actual)
        std::cerr << ' ' << token;
    std::cerr << '\n';
    ++failures;
}

} // namespace

int main() {
    const std::string tokenizer_json = R"({
      "model": {
        "type": "BPE",
        "vocab": {
          "h": 0, "e": 1, "l": 2, "o": 3, "w": 4, "r": 5, "d": 6,
          "\u0120": 7, "!": 8, ".": 9, ",": 10,
          "he": 11, "ll": 12, "lo": 13, "hell": 19, "hel": 20,
          "\u0120w": 14, "or": 15, "ld": 16, "orld": 21,
          "hello": 17, "\u0120world": 18,
          "\u010a": 22, "\u010a\u010a": 23, ".\u010a\u010a": 24,
          "|": 25, "|.": 26
        },
        "merges": [
          "h e", "l l", "l o", "he ll", "hel lo",
          "\u0120 w", "o r", "l d", "or ld",
          "\u0120w orld", "hello \u0120world",
          "\u010a \u010a", ". \u010a\u010a", "| ."
        ]
      },
      "pre_tokenizer": {
        "type": "Sequence",
        "pretokenizers": [
          {
            "type": "Split",
            "pattern": {"Regex": " ?[^(\\s|[.,!?])]+"},
            "behavior": "Isolated",
            "invert": false
          },
          {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false}
        ]
      }
    })";

    auto tokenizer = trtmc::CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), false);
    check(tokenizer != nullptr, "word_separator_create");
    if (tokenizer == nullptr)
        return 1;

    const auto punctuation_newlines = tokenizer->encode(".\n\nworld");
    check_ids(punctuation_newlines, {24, 4, 21},
              "word_separator_preserves_unmatched_punctuation_newlines");

    const auto regex_literals = tokenizer->encode("|.world");
    check_ids(regex_literals, {26, 4, 21}, "word_separator_preserves_unmatched_regex_literals");

    return failures == 0 ? 0 : 1;
}

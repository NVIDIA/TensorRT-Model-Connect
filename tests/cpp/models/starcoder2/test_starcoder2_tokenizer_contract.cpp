/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// StarCoder2's tokenizer_config declares GPT2Tokenizer while its published
// tokenizer.json wraps ByteLevel in Sequence[Digits, ByteLevel]. Verify that
// the native runtime already applies the effective GPT-2 token boundaries.

#include "trtmc/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    const std::string tokenizer_json = R"({
      "model": {
        "type": "BPE",
        "vocab": {
          "\u010a": 0,
          "\u0120": 1,
          "0": 2,
          ".": 3,
          "5": 4,
          "\u010a\u0120": 5,
          "\u010a\u0120\u0120": 6,
          "\u010a\u0120\u0120\u0120": 7,
          "\u010a\u0120\u0120\u0120\u0120": 8
        },
        "merges": [
          "\u010a \u0120",
          "\u010a\u0120 \u0120",
          "\u010a\u0120\u0120 \u0120",
          "\u010a\u0120\u0120\u0120 \u0120"
        ]
      },
      "pre_tokenizer": {
        "type": "Sequence",
        "pretokenizers": [
          {"type": "Digits", "individual_digits": true},
          {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
          }
        ]
      },
      "decoder": {
        "type": "ByteLevel",
        "add_prefix_space": true,
        "trim_offsets": true,
        "use_regex": true
      }
    })";

    auto tokenizer =
        trtmc::CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), false);
    if (!tokenizer) {
        std::cerr << "FAIL: native StarCoder2 tokenizer was not created\n";
        return 1;
    }

    const std::vector<int32_t> expected{7, 1, 2, 3, 4};
    const auto actual = tokenizer->encode("\n    0.5");
    if (actual != expected) {
        std::cerr << "FAIL: effective GPT-2 token boundaries differ\n";
        return 1;
    }
    return 0;
}

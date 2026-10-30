/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// detect_split_variant() previously returned on the first Split step found
// in a Sequence pre_tokenizer, even when that step's regex classifies as
// the generic kLlama default. Checkpoints that isolate digit-grouping
// (\p{N}{1,3}) as its own Split step ahead of the step that actually
// identifies the variant (e.g. Qwen3's `[^\r\n...` signature) were
// misclassified as kLlama, silently disabling variant-specific
// pre-tokenization behavior (trailing-newline attachment, grouped digit
// runs) for the whole checkpoint.
//
// The first two checks below use the pre_tokenizer field exactly as
// published in openbmb/MiniCPM5-2B's tokenizer.json (a real, public
// checkpoint that triggers this misclassification), so a passing result
// here reflects the real checkpoint's own regex shapes, not a synthetic
// stand-in. The third check is a synthetic fixture verifying the
// ambiguous-classification case (two Split steps that each classify to a
// different, non-kLlama variant) throws rather than silently picking one.

#include "families/llama/runtime/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check_ids(const std::vector<int32_t>& actual, const std::vector<int32_t>& expected,
               const char* name) {
    if (actual == expected)
        return;
    std::cerr << "FAIL: " << name << " expected";
    for (const int32_t token : expected)
        std::cerr << ' ' << token;
    std::cerr << " got";
    for (const int32_t token : actual)
        std::cerr << ' ' << token;
    std::cerr << '\n';
    ++failures;
}

// MiniCPM5-2B's actual pre_tokenizer field (openbmb/MiniCPM5-2B,
// tokenizer.json): a digit-grouping Split step first, then the real
// classifying (Qwen3-shaped) Split step second, then ByteLevel.
constexpr const char* kMiniCPM5PreTokenizer = R"(
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {
        "type": "Split",
        "pattern": {"Regex": "\\p{N}{1,3}"},
        "behavior": "Isolated",
        "invert": false
      },
      {
        "type": "Split",
        "pattern": {
          "Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        },
        "behavior": "Isolated",
        "invert": false
      },
      {
        "type": "ByteLevel",
        "add_prefix_space": false,
        "trim_offsets": true,
        "use_regex": false
      }
    ]
  })";

// Trailing newlines after punctuation should stay attached to the same
// pre-token (Qwen3-variant behavior) rather than split off as their own
// whitespace token. Only reachable if the second Split step's regex is
// actually used to classify the variant, not the first.
void check_newline_attachment() {
    const std::string tokenizer_json = std::string(R"({
      "model": {
        "type": "BPE",
        "vocab": {
          ".": 0,
          "Ċ": 1,
          "ĊĊ": 2,
          ".ĊĊ": 3
        },
        "merges": ["Ċ Ċ", ". ĊĊ"]
      },)") + kMiniCPM5PreTokenizer + R"(
    })";

    auto tokenizer = trtmc::CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), false);
    if (!tokenizer) {
        std::cerr << "FAIL: check_newline_attachment: tokenizer was not created\n";
        ++failures;
        return;
    }
    check_ids(tokenizer->encode(".\n\n"), {3}, "check_newline_attachment");
}

// Multi-digit runs should be grouped into chunks of up to 3 digits
// (\p{N}{1,3}), each its own pre-token, not left as one unbounded digit
// run (the kLlama-misclassified behavior) and not split into individual
// one-digit pre-tokens (the failure mode of a fix that scans for the
// variant but couples digit-group size to the same winning step, which
// does not carry the {1,3} grouping in this checkpoint's real regex).
void check_digit_grouping() {
    const std::string tokenizer_json = std::string(R"({
      "model": {
        "type": "BPE",
        "vocab": {
          "1": 0,
          "5": 1,
          "0": 2,
          "15": 3,
          "150": 4,
          "1500": 5
        },
        "merges": ["1 5", "15 0", "150 0"]
      },)") + kMiniCPM5PreTokenizer + R"(
    })";

    auto tokenizer = trtmc::CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), false);
    if (!tokenizer) {
        std::cerr << "FAIL: check_digit_grouping: tokenizer was not created\n";
        ++failures;
        return;
    }
    // Correct (grouped by 3): ["150", "0"] -> ids {4, 2}.
    // kLlama misclassification (unpatched bug): one ungrouped word "1500" -> {5}.
    // Coupled/naive fix (digit_group taken from the same step as the
    // variant, which for this checkpoint's real second step is 0): four
    // one-digit words -> {0, 1, 2, 2}.
    check_ids(tokenizer->encode("1500"), {4, 2}, "check_digit_grouping");
}

// Synthetic fixture: two Split steps that each classify to a different,
// non-kLlama variant. Not a real checkpoint shape -- this exercises the
// ambiguous-classification guard rather than reproducing a reported bug.
void check_ambiguous_variants_throw() {
    const std::string tokenizer_json = R"({
      "model": {
        "type": "BPE",
        "vocab": {"a": 0},
        "merges": []
      },
      "pre_tokenizer": {
        "type": "Sequence",
        "pretokenizers": [
          {
            "type": "Split",
            "pattern": {"Regex": "[^(\\s)]+"},
            "behavior": "Isolated",
            "invert": false
          },
          {
            "type": "Split",
            "pattern": {"Regex": "[^\\r\\n\\p{L}\\p{N}]?\\p{L}+"},
            "behavior": "Isolated",
            "invert": false
          }
        ]
      }
    })";

    bool threw = false;
    try {
        trtmc::CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), false);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (!threw) {
        std::cerr
            << "FAIL: check_ambiguous_variants_throw: expected std::runtime_error, none thrown\n";
        ++failures;
    }
}

} // namespace

int main() {
    check_newline_attachment();
    check_digit_grouping();
    check_ambiguous_variants_throw();
    return failures == 0 ? 0 : 1;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen/runtime/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check_ids(const std::vector<std::int32_t>& actual, const std::vector<std::int32_t>& expected,
               const char* name) {
    if (actual == expected)
        return;
    std::cerr << "FAIL: " << name << " got";
    for (const std::int32_t token : actual)
        std::cerr << ' ' << token;
    std::cerr << '\n';
    ++failures;
}

// Byte-level vocabulary where "'T" and "'\n" are single tokens; Ċ is the
// byte-level spelling of '\n'.
std::string tokenizer_json(const std::string& pre_tokenizer) {
    return R"({
      "model": {
        "type": "BPE",
        "vocab": {"'": 0, "T": 1, "'T": 2, "Ċ": 3, "'Ċ": 4},
        "merges": ["' T", "' Ċ"]
      },
      "pre_tokenizer": )" +
           pre_tokenizer + "}";
}

// The Split regex of the pinned Qwen3 checkpoints.
const std::string qwen3_pre_tokenizer = R"({
  "type": "Sequence",
  "pretokenizers": [
    {
      "type": "Split",
      "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
      "behavior": "Isolated",
      "invert": false
    },
    {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
  ]
})";

// GPT-2's built-in ByteLevel regex matches contractions case-sensitively.
const std::string gpt2_pre_tokenizer =
    R"({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true})";

} // namespace

int main() {
    const auto qwen3_json = tokenizer_json(qwen3_pre_tokenizer);
    const auto qwen3 = trtmc::CreateBpeTokenizer(qwen3_json.data(), qwen3_json.size(), false);
    check_ids(qwen3->encode("'T"), {2}, "qwen3 contractions are case-insensitive");
    check_ids(qwen3->encode("'\n"), {4}, "qwen3 apostrophe keeps trailing newlines");
    check_ids(qwen3->encode("'T'\n"), {2, 4}, "qwen3 contraction then apostrophe newline");

    const auto gpt2_json = tokenizer_json(gpt2_pre_tokenizer);
    const auto gpt2 = trtmc::CreateBpeTokenizer(gpt2_json.data(), gpt2_json.size(), false);
    check_ids(gpt2->encode("'T"), {0, 1}, "gpt2 contractions stay case-sensitive");
    check_ids(gpt2->encode("'\n"), {0, 3}, "gpt2 apostrophe does not take newlines");

    if (failures != 0)
        std::cerr << failures << " Qwen tokenizer test(s) failed\n";
    return failures == 0 ? 0 : 1;
}

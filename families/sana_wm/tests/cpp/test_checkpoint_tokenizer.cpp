/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The SANA-WM factory must frame prompts with the checkpoint tokenizer's
// post-processor, as the official Gemma text encoders do: every prompt,
// including an empty negative prompt, starts with <bos>.

#include "families/sana_wm/runtime/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::int32_t kBosId = 2;

// Minimal Gemma-style tokenizer.json: special tokens, a BPE merge and the
// TemplateProcessing post-processor "<bos> $A" used by both SANA-WM tokenizers.
const char* kTokenizerJson = R"json({
  "added_tokens": [
    {"id": 0, "content": "<pad>", "special": true},
    {"id": 1, "content": "<eos>", "special": true},
    {"id": 2, "content": "<bos>", "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<bos>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}}
    ],
    "pair": [],
    "special_tokens": {"<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]}}
  },
  "decoder": null,
  "model": {
    "type": "BPE",
    "vocab": {"<pad>": 0, "<eos>": 1, "<bos>": 2, "a": 3, "b": 4, "ab": 5},
    "merges": ["a b"],
    "byte_fallback": false
  }
})json";

int g_failures = 0;

void check(bool condition, const std::string& name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

} // namespace

int main() {
    const std::string json = kTokenizerJson;
    const auto tokenizer = trtmc::CreateSanaWmCheckpointTokenizer(json.data(), json.size());
    check(tokenizer != nullptr, "checkpoint tokenizer loads");
    if (tokenizer) {
        check(tokenizer->encode("ab") == std::vector<std::int32_t>{kBosId, 5},
              "prompt starts with <bos> and keeps its BPE tokens");
        check(tokenizer->encode("") == std::vector<std::int32_t>{kBosId},
              "empty prompt encodes to <bos> instead of an all-padding sequence");
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " SANA-WM checkpoint tokenizer check(s) failed\n";
        return 1;
    }
    std::cout << "SANA-WM checkpoint tokenizer tests passed\n";
    return 0;
}

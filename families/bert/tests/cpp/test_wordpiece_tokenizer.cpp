/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/bert/runtime/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check_ids(const std::vector<int32_t>& actual, const std::vector<int32_t>& expected,
               const char* name) {
    if (actual == expected)
        return;
    std::cerr << "FAIL: " << name << " got";
    for (const auto id : actual)
        std::cerr << ' ' << id;
    std::cerr << '\n';
    ++failures;
}

} // namespace

int main() {
    const std::string tokenizer_json = R"({
      "model": {
        "type": "WordPiece", "unk_token": "[UNK]",
        "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
        "vocab": {
          "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
          "hello": 5, "world": 6, "[": 7, "]": 8, "mask": 9,
          "cls": 10, "sep": 11, "pad": 12, "unk": 13, ".": 14,
          ",": 15, "cafe": 16, "play": 17, "##ing": 18
        }
      },
      "normalizer": {"type": "BertNormalizer", "clean_text": true,
                     "handle_chinese_chars": true, "strip_accents": null, "lowercase": true},
      "pre_tokenizer": {"type": "BertPreTokenizer"},
      "post_processor": {"type": "BertProcessing", "cls": ["[CLS]", 2], "sep": ["[SEP]", 3]},
      "added_tokens": [
        {"id": 0, "content": "[PAD]", "special": true, "normalized": false,
         "single_word": false, "lstrip": false, "rstrip": false},
        {"id": 1, "content": "[UNK]", "special": true, "normalized": false,
         "single_word": false, "lstrip": false, "rstrip": false},
        {"id": 2, "content": "[CLS]", "special": true, "normalized": false,
         "single_word": false, "lstrip": false, "rstrip": false},
        {"id": 3, "content": "[SEP]", "special": true, "normalized": false,
         "single_word": false, "lstrip": false, "rstrip": false},
        {"id": 4, "content": "[MASK]", "special": true, "normalized": false,
         "single_word": false, "lstrip": false, "rstrip": false}
      ]
    })";

    for (bool add_special : {false, true}) {
        auto tokenizer = trtmc::CreateWordPieceTokenizer(tokenizer_json.data(),
                                                         tokenizer_json.size(), add_special);
        auto check = [&](const std::string& text, std::vector<int32_t> expected, const char* name) {
            if (add_special) {
                expected.insert(expected.begin(), 2);
                expected.push_back(3);
            }
            check_ids(tokenizer->encode(text), expected, name);
        };
        check("[PAD][UNK][CLS][SEP][MASK]", {0, 1, 2, 3, 4}, "all explicit special tokens");
        check("hello [SEP] world", {5, 3, 6}, "separator between words");
        check("hello[MASK]world", {5, 4, 6}, "special token adjacent to words");
        check("[MASK][MASK]", {4, 4}, "repeated special tokens");
        check("[CLS] hello [SEP]", {2, 5, 3}, "explicit framing tokens");
        check("[MASK], [MASK].", {4, 15, 4, 14}, "punctuation around special tokens");
        check("[mask]", {7, 9, 8}, "special token matching is case sensitive");
        check("HELLO, world.", {5, 15, 6, 14}, "ordinary normalization and punctuation");
        check("CAF\xc3\x89 [MASK] playing", {16, 4, 17, 18}, "normalization on both sides");
        check("playing", {17, 18}, "ordinary wordpieces");
        check("unrecognized", {1}, "unknown word");
        check("", {}, "empty input");
        check(" \t\n ", {}, "whitespace input");
    }

    return failures == 0 ? 0 : 1;
}

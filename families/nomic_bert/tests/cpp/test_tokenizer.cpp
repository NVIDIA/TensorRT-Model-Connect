/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * What this family actually relies on the tokenizer for.
 *
 * The pipeline prepends a task prefix, pads to a fixed window and keeps the
 * closing [SEP] when it truncates. All three depend on the tokenizer framing
 * every input with [CLS] ... [SEP] and on the prefix tokenizing as ordinary
 * words rather than as something special - so those are what is checked
 * here, rather than wordpiece in general.
 */

#include "families/nomic_bert/runtime/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(const std::vector<std::int32_t>& actual, const std::vector<std::int32_t>& expected,
           const char* name) {
    if (actual == expected)
        return;
    std::cerr << "FAIL: " << name << " got";
    for (const auto id : actual)
        std::cerr << ' ' << id;
    std::cerr << "\n           expected";
    for (const auto id : expected)
        std::cerr << ' ' << id;
    std::cerr << '\n';
    ++failures;
}

void expect(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    // The real vocabulary is 30522 entries; this is the slice the checks below
    // need, with the same normalizer and post-processor the checkpoint ships.
    // The ids are renumbered to run from zero because the loader sizes its id
    // table by the number of entries, so a 20-word excerpt cannot keep the
    // real ids. Nothing here depends on the particular numbers.
    const std::string tokenizer_json = R"({
      "model": {
        "type": "WordPiece", "unk_token": "[UNK]",
        "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
        "vocab": {
          "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
          "search": 5, "_": 6, "document": 7, "query": 8, ":": 9,
          "paris": 10, "the": 11, "capital": 12, "of": 13, "france": 14,
          "is": 15, ".": 16, "play": 17, "##ing": 18, "cafe": 19
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
         "single_word": false, "lstrip": false, "rstrip": false}
      ]
    })";

    auto tokenizer = trtmc::CreateWordPieceTokenizer(tokenizer_json.data(), tokenizer_json.size(),
                                                     /*add_special_tokens=*/true);
    expect(tokenizer != nullptr, "tokenizer loads");
    if (tokenizer == nullptr)
        return 1;

    // Every input is framed. The pipeline's truncation keeps the last id on
    // the assumption that it is the [SEP], and the padding assumes the real
    // tokens are a prefix of the window.
    const auto framed = tokenizer->encode("paris");
    check(framed, {2, 10, 3}, "[CLS] and [SEP] frame the input");

    // The task prefix is ordinary text. If "search_document:" tokenized as a
    // special token the prefix would not reach the encoder as the words the
    // checkpoint was trained on.
    check(tokenizer->encode("search_document: paris"), {2, 5, 6, 7, 9, 10, 3},
          "the document prefix is ordinary words");
    check(tokenizer->encode("search_query: paris"), {2, 5, 6, 8, 9, 10, 3},
          "the query prefix is ordinary words");

    // Lowercasing is part of this checkpoint's normalizer, not an assumption.
    check(tokenizer->encode("PARIS"), {2, 10, 3}, "input is lowercased");
    // strip_accents is null, which means it follows lowercase, so it is on.
    check(tokenizer->encode("CAF\xc3\x89"), {2, 19, 3}, "accents are stripped");
    check(tokenizer->encode("playing"), {2, 17, 18, 3}, "wordpieces continue with ##");
    check(tokenizer->encode("zzzzzz"), {2, 1, 3}, "an unknown word becomes [UNK]");

    // An empty input is still framed, so the pipeline never sees an empty id
    // list from a caller who passed an empty string with a prefix on it.
    check(tokenizer->encode(""), {2, 3}, "empty input is still framed");

    const auto sentence = tokenizer->encode("The capital of France is Paris.");
    expect(sentence.front() == 2, "a sentence opens with [CLS]");
    expect(sentence.back() == 3, "a sentence closes with [SEP]");

    return failures == 0 ? 0 : 1;
}

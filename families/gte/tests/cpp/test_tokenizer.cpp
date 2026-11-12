/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * What this family actually relies on the tokenizer for.
 *
 * The pipeline pads to a fixed window, keeps the closing [SEP] when it
 * truncates, and then reads **row 0** as the embedding. That last one is
 * what makes the framing load-bearing here rather than cosmetic: if the
 * opening [CLS] were missing, the pooled vector would be the first word's
 * row and would still look like an embedding. So the checks below are about
 * framing and normalisation, not about wordpiece in general.
 *
 * This checkpoint's post_processor is TemplateProcessing rather than the
 * BertProcessing the rotary encoder next door ships, so the excerpt below
 * uses the template form the real tokenizer.json carries.
 */

#include "families/gte/runtime/tokenizer.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
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

void check_unicode_normalization() {
    const auto path = std::filesystem::path(__FILE__).parent_path().parent_path() /
                      "fixtures/unicode_tokenizer.json";
    std::ifstream input(path);
    const auto fixture = nlohmann::json::parse(input);
    auto definition = fixture.at("tokenizer");
    for (const auto& sample : fixture.at("cases")) {
        definition["normalizer"] = sample.at("normalizer");
        const auto data = definition.dump();
        auto tokenizer = trtmc::CreateWordPieceTokenizer(data.data(), data.size(), true);
        const auto text = sample.at("input").get<std::string>();
        const auto expected = sample.at("ids").get<std::vector<std::int32_t>>();
        check(tokenizer->encode(text), expected, text.c_str());
    }
}

} // namespace

int main() {
    check_unicode_normalization();
    // The real vocabulary is 30522 entries; this is the slice the checks
    // below need, with the same normalizer and post-processor the checkpoint
    // ships. The ids are renumbered to run from zero because the loader
    // sizes its id table by the number of entries, so a short excerpt cannot
    // keep the real ids. Nothing here depends on the particular numbers.
    const std::string tokenizer_json = R"({
      "model": {
        "type": "WordPiece", "unk_token": "[UNK]",
        "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
        "vocab": {
          "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
          "paris": 5, "the": 6, "capital": 7, "of": 8, "france": 9,
          "is": 10, ".": 11, "play": 12, "##ing": 13, "cafe": 14
        }
      },
      "normalizer": {"type": "BertNormalizer", "clean_text": true,
                     "handle_chinese_chars": true, "strip_accents": null, "lowercase": true},
      "pre_tokenizer": {"type": "BertPreTokenizer"},
      "post_processor": {
        "type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                   {"Sequence": {"id": "A", "type_id": 0}},
                   {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
        "special_tokens": {
          "[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
          "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]}
        }
      },
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

    // Row 0 is the embedding, so the opening [CLS] is the pooled token.
    const auto framed = tokenizer->encode("paris");
    check(framed, {2, 5, 3}, "[CLS] and [SEP] frame the input");

    // Lowercasing is part of this checkpoint's normalizer, not an assumption.
    check(tokenizer->encode("PARIS"), {2, 5, 3}, "input is lowercased");
    // strip_accents is null, which means it follows lowercase, so it is on.
    check(tokenizer->encode("CAF\xc3\x89"), {2, 14, 3}, "accents are stripped");
    check(tokenizer->encode("playing"), {2, 12, 13, 3}, "wordpieces continue with ##");
    check(tokenizer->encode("zzzzzz"), {2, 1, 3}, "an unknown word becomes [UNK]");

    // An empty input is still framed, so the pipeline never reads row 0 of a
    // sequence that has no [CLS] in it.
    check(tokenizer->encode(""), {2, 3}, "empty input is still framed");

    const auto sentence = tokenizer->encode("The capital of France is Paris.");
    expect(sentence.front() == 2, "a sentence opens with [CLS]");
    expect(sentence.back() == 3, "a sentence closes with [SEP]");

    return failures == 0 ? 0 : 1;
}

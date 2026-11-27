/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/deberta_v2/runtime/tokenizer.h"

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
    // The real vocabulary is 128000 Unigram pieces; this is the slice these
    // checks need, with the same normalizer and pre-tokenizer the converted
    // tokenizer.json carries. Scores are all equal, so the longest piece
    // wins, which is enough to exercise the pieces below.
    const std::string tokenizer_json = R"({
      "model": {
        "type": "Unigram", "unk_id": 0,
        "vocab": [
          ["<unk>", 0.0], ["▁the", -1.0], ["▁capital", -1.0],
          ["▁of", -1.0], ["▁france", -1.0], ["▁is", -1.0],
          ["▁paris", -1.0], [".", -1.0], ["▁a", -1.0], ["▁b", -1.0],
          ["▁é", -1.0], ["▁가", -1.0], ["▁각", -1.0], ["▁Å", -1.0],
          ["▁Ǻ", -1.0], ["▁ộ", -1.0], ["▁क़", -1.0], ["▁Ώ", -1.0],
          ["▁①", -1.0], ["▁ﬀ", -1.0], ["▁é̀", -1.0]
        ]
      },
      "normalizer": {"type": "Sequence", "normalizers": [
        {"type": "Replace", "pattern": {"Regex": "\\s{2,}|[\\n\\r\\t]"}, "content": " "},
        {"type": "NFC"},
        {"type": "Strip", "strip_left": false, "strip_right": true}
      ]},
      "pre_tokenizer": {"type": "Metaspace", "replacement": "▁",
                        "prepend_scheme": "always", "split": true},
      "added_tokens": []
    })";

    auto tokenizer = trtmc::CreateDebertaTokenizer(tokenizer_json.data(), tokenizer_json.size(),
                                                   /*add_special_tokens=*/false);
    expect(tokenizer != nullptr, "tokenizer loads");
    if (tokenizer == nullptr)
        return 1;

    const auto plain = tokenizer->encode("the capital of france is paris.");
    check(plain, {1, 2, 3, 4, 5, 6, 7}, "an ordinary sentence");

    // The body only: this checkpoint's post-processor declares no special
    // tokens, so the [CLS]/[SEP] framing is the pipeline's job and must not
    // appear here.
    expect(plain.front() == 1, "no [CLS] is prepended by the tokenizer");
    expect(plain.back() == 7, "no [SEP] is appended by the tokenizer");

    // Replace: runs of two or more collapse, and a lone tab or newline
    // becomes one space, so all of these normalise to the same sentence.
    for (const char* spaced :
         {"the  capital   of france is paris.", "the\tcapital\nof france is paris.",
          "the \n capital \t of france is paris."}) {
        check(tokenizer->encode(spaced), plain, "whitespace runs collapse");
    }

    // Strip is right-only: a trailing space goes, a leading one is kept and
    // is what Metaspace turns into the word-initial mark.
    check(tokenizer->encode("the capital of france is paris.   "), plain,
          "trailing whitespace is stripped");

    const auto check_input = [&](const std::string& text, const std::vector<std::int32_t>& expected,
                                 const char* name) {
        try {
            check(tokenizer->encode(text), expected, name);
        } catch (const std::exception& error) {
            std::cerr << "FAIL: " << name << ": " << error.what() << '\n';
            ++failures;
        }
    };
    check_input("é a", {10, 8}, "composed accent");
    check_input("é a", {10, 8}, "decomposed accent");
    check_input("가 a", {11, 8}, "Hangul LV composition");
    check_input("각 a", {12, 8}, "Hangul LVT composition");
    check_input("Å a", {13, 8}, "canonical singleton");
    check_input("Å a", {13, 8}, "ring composition");
    check_input("Ǻ a", {14, 8}, "recursive composition");
    check_input("ộ a", {15, 8}, "canonical mark reordering");
    check_input("ộ a", {15, 8}, "ordered canonical marks");
    check_input("क़ a", {16, 8}, "composition exclusion");
    check_input("Ώ a", {17, 8}, "Greek composition");
    check_input("① a", {18, 8}, "circled digit remains compatible-only");
    check_input("ﬀ a", {19, 8}, "ligature remains compatible-only");
    check_input("é̀ a", {20, 8}, "uncomposed trailing mark is retained");

    for (const std::string& text :
         {std::string("\xc3"), std::string("\xc0\xaf"), std::string("\xed\xa0\x80"),
          std::string("\u0378"), std::string("\U00011380")}) {
        bool refused = false;
        try {
            tokenizer->encode(text);
        } catch (const std::exception&) {
            refused = true;
        }
        expect(refused, "invalid UTF-8 or out-of-profile scalar is refused");
    }

    const std::string close_scores = R"({
      "model": {"type": "Unigram", "unk_id": 0, "vocab": [
        ["<unk>", 0.0], ["▁", 0.0], ["a", -0.1], ["aa", -0.200000001]
      ]},
      "pre_tokenizer": {"type": "Metaspace", "replacement": "▁",
                        "prepend_scheme": "always", "split": true}
    })";
    const auto precise = trtmc::CreateDebertaTokenizer(close_scores.data(), close_scores.size());
    check(precise->encode("aa"), {1, 2, 2}, "nearby Unigram scores retain their ordering");

    return failures == 0 ? 0 : 1;
}

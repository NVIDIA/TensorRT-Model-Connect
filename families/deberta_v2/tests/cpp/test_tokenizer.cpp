/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * The three normalizer steps this checkpoint declares, and the one it cannot
 * do.
 *
 * DeBERTa's converted tokenizer.json asks for Replace(\s{2,}|[\n\r\t] -> " "),
 * then NFC, then a right-only Strip. The first and third are implemented
 * exactly. NFC is not: composing Unicode in full is a large table, and
 * getting it subtly wrong would move tokenization silently, so input that
 * NFC would actually have changed is refused instead. These are the checks
 * for all three.
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
          ["▁paris", -1.0], [".", -1.0], ["▁a", -1.0], ["▁b", -1.0]
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

    // NFC: composed text passes through untouched...
    check(tokenizer->encode("é a"), tokenizer->encode("é a"), "composed text is stable");
    // ...and decomposed text is refused rather than silently mis-tokenised.
    bool refused = false;
    try {
        tokenizer->encode("é a");
    } catch (const std::exception&) {
        refused = true;
    }
    expect(refused, "decomposed text is refused rather than passed through");

    return failures == 0 ? 0 : 1;
}

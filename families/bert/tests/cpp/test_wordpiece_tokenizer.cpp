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
          ",": 15, "cafe": 16, "play": 17, "##ing": 18,
          "istanbul": 19, "dvorak": 20, "cesky": 21, "łodz": 22,
          "skoda": 23, "i̇": 24, "hēllō": 25, "HELLO": 26, "I": 27,
          "αθηνα": 28, "москва": 29, "иога": 30, "елка": 31,
          "αθήνα": 32, "йога": 33, "ёлка": 34, "ΑΘΗΝΑ": 35,
          "МОСКВА": 36, "ИОГА": 37, "ЕЛКА": 38, "ι": 39, "υ": 40, "ΐ": 41, "ΰ": 42, "ʹ": 43, "ʹ": 44,
          "¡": 45, "،": 46, "।": 47, "‿": 48, "hello⁒world": 49, "※": 50,
          "〃": 51, "｡": 52, "hello〆world": 53, "hello々world": 54,
          "hello﹢world": 55, "hello＋world": 56, "hello＄world": 57,
          "hello｀world": 58, "hello〱world": 59, "hello⁄world": 60,
          "hello〿world": 61
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
        check("HĒLLŌ", {5}, "Latin Extended-A uppercase accents");
        check("he\xcc\x84llo\xcc\x84", {5}, "decomposed Latin Extended-A accents");
        check("İstanbul Dvořák ČESKÝ Łódź ŠKODA", {19, 20, 21, 22, 23}, "Latin Extended-A names");
        check("ΑΘΉΝΑ ΜΟΣΧΑ МОСКВА ЙОГА ЁЛКА", {28, 1, 29, 30, 31},
              "Greek and Cyrillic case and canonical accents");
        check("Αθήνα йога ёлка", {28, 30, 31}, "decomposed Greek and Cyrillic accents");
        check("[MASK]Αθήνα[MASK]Москва", {4, 28, 4, 29},
              "Greek and Cyrillic around preserved special tokens");
        check("ΐ ΰ", {39, 40}, "recursive Greek canonical decomposition");
        check("ʹ", {43}, "Greek singleton canonical decomposition");
        check("CAF\xc3\x89 [MASK] playing", {16, 4, 17, 18}, "normalization on both sides");
        check("playing", {17, 18}, "ordinary wordpieces");
        check("hello¡world", {5, 45, 6}, "Latin punctuation splits words");
        check("hello،world", {5, 46, 6}, "Arabic punctuation splits words");
        check("hello।world", {5, 47, 6}, "Devanagari punctuation splits words");
        check("hello‿world", {5, 48, 6}, "connector punctuation splits words");
        check("hello⁒world", {49}, "commercial minus symbol remains inside a word");
        check("hello※world", {5, 50, 6}, "reference mark punctuation splits words");
        check("hello〃world", {5, 51, 6}, "CJK punctuation splits words");
        check("hello｡world", {5, 52, 6}, "halfwidth punctuation splits words");
        check("hello〆world", {53}, "CJK letter remains inside a word");
        check("hello々world", {54}, "CJK iteration modifier remains inside a word");
        check("hello﹢world", {55}, "small mathematical symbol remains inside a word");
        check("hello＋world", {56}, "fullwidth mathematical symbol remains inside a word");
        check("hello＄world", {57}, "fullwidth currency symbol remains inside a word");
        check("hello｀world", {58}, "fullwidth modifier remains inside a word");
        check("hello〱world", {59}, "vertical iteration modifier remains inside a word");
        check("hello⁄world", {60}, "fraction slash symbol remains inside a word");
        check("hello〿world", {61}, "CJK symbol remains inside a word");
        check("hello$world", {5, 1, 6}, "ASCII currency retains reference punctuation behavior");
        // Expected token IDs were checked with Hugging Face tokenizers 0.22.2.
        for (const std::string& control :
             {std::string("\xc2\xad"), std::string("\xd8\x80"), std::string("\xe2\x80\x8b"),
              std::string("\xe2\x80\x8e"), std::string("\xe2\x80\xae"), std::string("\xe2\x81\xa0"),
              std::string("\xee\x80\x80"), std::string("\xef\xbb\xbf"), std::string("\xef\xbf\xbb"),
              std::string("\xf0\x91\x82\xbd"), std::string("\xf0\x9b\xb2\xa0"),
              std::string("\xf3\xa0\x80\xa0"), std::string("\xf3\xb0\x80\x80"),
              std::string("\xf4\x80\x80\x80")}) {
            check("hel" + control + "lo", {5}, "clean Unicode control inside word");
            check(control + "hello" + control + "[MASK]" + control + "world" + control, {5, 4, 6},
                  "clean controls around explicit special token");
        }
        check("hello\tworld\nhello\rworld", {5, 6, 5, 6}, "keep whitespace separators");
        check("hel\xcd\xb8lo", {1}, "unassigned U+0378 is not removed");
        check("hel\xf3\xbf\xbf\xbelo", {1}, "noncharacter U+FFFFE is not removed");
        check("unrecognized", {1}, "unknown word");
        check("", {}, "empty input");
        check(" \t\n ", {}, "whitespace input");
    }

    std::string unclean_json = tokenizer_json;
    const auto clean_flag = unclean_json.find("\"clean_text\": true");
    unclean_json.replace(clean_flag, std::string("\"clean_text\": true").size(),
                         "\"clean_text\": false");
    auto unclean = trtmc::CreateWordPieceTokenizer(unclean_json.data(), unclean_json.size(), false);
    check_ids(unclean->encode("hel\xc2\xadlo"), {1}, "clean_text false preserves soft hyphen");

    std::string accented_json = tokenizer_json;
    const auto accent_flag = accented_json.find("\"strip_accents\": null");
    accented_json.replace(accent_flag, std::string("\"strip_accents\": null").size(),
                          "\"strip_accents\": false");
    auto accented =
        trtmc::CreateWordPieceTokenizer(accented_json.data(), accented_json.size(), false);
    check_ids(accented->encode("İ HĒLLŌ"), {24, 25},
              "lowercase preserves accents when stripping is disabled");

    check_ids(accented->encode("ΑΘΉΝΑ МОСКВА ЙОГА ЁЛКА"), {32, 29, 33, 34},
              "Greek and Cyrillic accents remain when stripping is disabled");

    check_ids(accented->encode("ΐ ΰ"), {41, 42},
              "recursive Greek accents remain when stripping is disabled");

    check_ids(accented->encode("ʹ"), {44}, "Greek singleton remains without NFD stripping");

    std::string cased_json = tokenizer_json;
    cased_json.replace(cased_json.find("\"lowercase\": true"),
                       std::string("\"lowercase\": true").size(), "\"lowercase\": false");
    cased_json.replace(cased_json.find("\"strip_accents\": null"),
                       std::string("\"strip_accents\": null").size(), "\"strip_accents\": true");
    auto cased = trtmc::CreateWordPieceTokenizer(cased_json.data(), cased_json.size(), false);
    check_ids(cased->encode("HĒLLŌ İ"), {26, 27},
              "accent stripping preserves case when lowercasing is disabled");

    check_ids(cased->encode("ΑΘΉΝΑ МОСКВА ЙОГА ЁЛКА"), {35, 36, 37, 38},
              "Greek and Cyrillic accent stripping preserves uppercase");

    return failures == 0 ? 0 : 1;
}

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

void test_unicode_normalization() {
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
          "hello〿world": 61, "viet": 62, "nam": 63, "عربي": 64, "שלום": 65, "क": 66, "a": 67, "b": 68, "ạ": 69, "Ệ": 70, "ω": 71, "k": 72, "աբգ": 73, "한글": 74, "عَرَبِيّ": 75, "a\u093eb": 76, "a\u20ddb": 77, "a\u08d4b": 78, "ა": 79, "𞤢": 80, "𐓘": 81, "𐖗": 82, "ɤ": 83, "𖺻": 84, "a\u1b44\u302e": 85, "A\u1b44\u302e": 86, "a\u302e\u1b44": 87, "A\u302e\u1b44": 88,
          "a\u1b44\u08d4": 89, "a\u08d4\u1b44": 90, "a\u302f\u302e": 91,
          "\u1b44\u302e": 92, "a\u302eb\u1b44": 93
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
        auto check_normalized = [&](const std::string& text, std::vector<int32_t> expected,
                                    const char* name) {
            if (add_special) {
                expected.insert(expected.begin(), 2);
                expected.push_back(3);
            }
            check(tokenizer->encode(text), expected, name);
        };
        check_normalized("[PAD][UNK][CLS][SEP][MASK]", {0, 1, 2, 3, 4},
                         "all explicit special tokens");
        check_normalized("hello [SEP] world", {5, 3, 6}, "separator between words");
        check_normalized("hello[MASK]world", {5, 4, 6}, "special token adjacent to words");
        check_normalized("[MASK][MASK]", {4, 4}, "repeated special tokens");
        check_normalized("[CLS] hello [SEP]", {2, 5, 3}, "explicit framing tokens");
        check_normalized("[MASK], [MASK].", {4, 15, 4, 14}, "punctuation around special tokens");
        check_normalized("[mask]", {7, 9, 8}, "special token matching is case sensitive");
        check_normalized("HELLO, world.", {5, 15, 6, 14}, "ordinary normalization and punctuation");
        check_normalized("HĒLLŌ", {5}, "Latin Extended-A uppercase accents");
        check_normalized("he\xcc\x84llo\xcc\x84", {5}, "decomposed Latin Extended-A accents");
        check_normalized("İstanbul Dvořák ČESKÝ Łódź ŠKODA", {19, 20, 21, 22, 23},
                         "Latin Extended-A names");
        check_normalized("ΑΘΉΝΑ ΜΟΣΧΑ МОСКВА ЙОГА ЁЛКА", {28, 1, 29, 30, 31},
                         "Greek and Cyrillic case and canonical accents");
        check_normalized("Αθήνα йога ёлка", {28, 30, 31}, "decomposed Greek and Cyrillic accents");
        check_normalized("[MASK]Αθήνα[MASK]Москва", {4, 28, 4, 29},
                         "Greek and Cyrillic around preserved special tokens");
        check_normalized("ΐ ΰ", {39, 40}, "recursive Greek canonical decomposition");
        check_normalized("ʹ", {43}, "Greek singleton canonical decomposition");
        check_normalized("CAF\xc3\x89 [MASK] playing", {16, 4, 17, 18},
                         "normalization on both sides");
        check_normalized("playing", {17, 18}, "ordinary wordpieces");
        check_normalized("Ω K Å", {71, 72, 67}, "canonical normalization precedes lowercasing");
        check_normalized("ԱԲԳ 한글", {73, 74},
                         "Armenian lowercase and Hangul canonical decomposition");
        check_normalized("Ა 𞤀 𐒰 𐕰 Ɤ 𖺠", {79, 80, 81, 82, 83, 84},
                         "lowercase mappings added after Unicode 8");
        check_normalized("A\u302e\u1b44", {85}, "NFD reorders retained spacing marks");
        check_normalized("A\u1b44\u302e", {85}, "NFD preserves ordered spacing marks");
        check_normalized("A\u08d4\u1b44", {89},
                         "Unicode 9 ordering retains a mark outside Unicode 8 categories");
        check_normalized("A\u302f\u302e", {91}, "equal combining classes retain input order");
        check_normalized("\u302e\u1b44", {92}, "NFD reorders leading combining marks");
        check_normalized("A\u302eB\u1b44", {93}, "NFD ordering stops at the next starter");
        check_normalized("[MASK]A\u302e\u1b44[MASK]", {4, 85, 4},
                         "NFD ordering preserves special token boundaries");
        check_normalized("VIỆT NAM", {62, 63}, "Vietnamese canonical accents and lowercase");
        check_normalized("عَرَبِيّ שָׁלוֹם क़", {64, 65, 66}, "nonspacing marks across scripts");
        check_normalized("a\u05b0", {67},
                         "Hebrew mark stripped outside combining diacritic blocks");
        check_normalized("a\u093eb", {76}, "spacing mark is preserved");
        check_normalized("a\u20ddb", {77}, "enclosing mark is preserved");
        check_normalized("a\u08d4b", {78}, "post Unicode 8 mark remains for the pinned oracle");
        check_normalized("hello¡world", {5, 45, 6}, "Latin punctuation splits words");
        check_normalized("hello،world", {5, 46, 6}, "Arabic punctuation splits words");
        check_normalized("hello।world", {5, 47, 6}, "Devanagari punctuation splits words");
        check_normalized("hello‿world", {5, 48, 6}, "connector punctuation splits words");
        check_normalized("hello⁒world", {49}, "commercial minus symbol remains inside a word");
        check_normalized("hello※world", {5, 50, 6}, "reference mark punctuation splits words");
        check_normalized("hello〃world", {5, 51, 6}, "CJK punctuation splits words");
        check_normalized("hello｡world", {5, 52, 6}, "halfwidth punctuation splits words");
        check_normalized("hello〆world", {53}, "CJK letter remains inside a word");
        check_normalized("hello々world", {54}, "CJK iteration modifier remains inside a word");
        check_normalized("hello﹢world", {55}, "small mathematical symbol remains inside a word");
        check_normalized("hello＋world", {56},
                         "fullwidth mathematical symbol remains inside a word");
        check_normalized("hello＄world", {57}, "fullwidth currency symbol remains inside a word");
        check_normalized("hello｀world", {58}, "fullwidth modifier remains inside a word");
        check_normalized("hello〱world", {59}, "vertical iteration modifier remains inside a word");
        check_normalized("hello⁄world", {60}, "fraction slash symbol remains inside a word");
        check_normalized("hello〿world", {61}, "CJK symbol remains inside a word");
        check_normalized("hello$world", {5, 1, 6},
                         "ASCII currency retains reference punctuation behavior");
        for (const std::string& control :
             {std::string("\xc2\xad"), std::string("\xd8\x80"), std::string("\xe2\x80\x8b"),
              std::string("\xe2\x80\x8e"), std::string("\xe2\x80\xae"), std::string("\xe2\x81\xa0"),
              std::string("\xee\x80\x80"), std::string("\xef\xbb\xbf"), std::string("\xef\xbf\xbb"),
              std::string("\xf0\x91\x82\xbd"), std::string("\xf0\x9b\xb2\xa0"),
              std::string("\xf3\xa0\x80\xa0"), std::string("\xf3\xb0\x80\x80"),
              std::string("\xf4\x80\x80\x80")}) {
            check_normalized("hel" + control + "lo", {5}, "clean Unicode control inside word");
            check_normalized(control + "hello" + control + "[MASK]" + control + "world" + control,
                             {5, 4, 6}, "clean controls around explicit special token");
        }
        check_normalized("hello\tworld\nhello\rworld", {5, 6, 5, 6}, "keep whitespace separators");
        check_normalized("hel\xcd\xb8lo", {1}, "unassigned U+0378 is not removed");
        check_normalized("hel\xf3\xbf\xbf\xbelo", {1}, "noncharacter U+FFFFE is not removed");
        check_normalized("unrecognized", {1}, "unknown word");
        check_normalized("", {}, "empty input");
        check_normalized(" \t\n ", {}, "whitespace input");
    }

    std::string unclean_json = tokenizer_json;
    const auto clean_flag = unclean_json.find("\"clean_text\": true");
    unclean_json.replace(clean_flag, std::string("\"clean_text\": true").size(),
                         "\"clean_text\": false");
    auto unclean = trtmc::CreateWordPieceTokenizer(unclean_json.data(), unclean_json.size(), false);
    check(unclean->encode("hel\xc2\xadlo"), {1}, "clean_text false preserves soft hyphen");

    std::string accented_json = tokenizer_json;
    const auto accent_flag = accented_json.find("\"strip_accents\": null");
    accented_json.replace(accent_flag, std::string("\"strip_accents\": null").size(),
                          "\"strip_accents\": false");
    auto accented =
        trtmc::CreateWordPieceTokenizer(accented_json.data(), accented_json.size(), false);
    check(accented->encode("Ა 𞤀 𐒰 𐕰 Ɤ 𖺠"), {79, 80, 81, 82, 83, 84},
          "modern lowercase without accent stripping");
    check(accented->encode("A\u302e\u1b44"), {87}, "strip_accents false preserves mark order");
    check(accented->encode("A\u08d4\u1b44"), {90},
          "strip_accents false preserves Unicode 9 mark order");
    check(accented->encode("عَرَبِيّ"), {75}, "nonspacing marks remain without stripping");
    check(accented->encode("Ạ"), {69}, "Vietnamese lowercase without stripping");
    check(accented->encode("İ HĒLLŌ"), {24, 25},
          "lowercase preserves accents when stripping is disabled");

    check(accented->encode("ΑΘΉΝΑ МОСКВА ЙОГА ЁЛКА"), {32, 29, 33, 34},
          "Greek and Cyrillic accents remain when stripping is disabled");

    check(accented->encode("ΐ ΰ"), {41, 42},
          "recursive Greek accents remain when stripping is disabled");

    check(accented->encode("ʹ"), {44}, "Greek singleton remains without NFD stripping");

    std::string cased_json = tokenizer_json;
    cased_json.replace(cased_json.find("\"lowercase\": true"),
                       std::string("\"lowercase\": true").size(), "\"lowercase\": false");
    cased_json.replace(cased_json.find("\"strip_accents\": null"),
                       std::string("\"strip_accents\": null").size(), "\"strip_accents\": true");
    auto cased = trtmc::CreateWordPieceTokenizer(cased_json.data(), cased_json.size(), false);
    check(cased->encode("A\u302e\u1b44"), {86}, "NFD ordering does not require lowercase");
    check(cased->encode("HĒLLŌ İ"), {26, 27},
          "accent stripping preserves case when lowercasing is disabled");

    check(cased->encode("ΑΘΉΝΑ МОСКВА ЙОГА ЁЛКА"), {35, 36, 37, 38},
          "Greek and Cyrillic accent stripping preserves uppercase");
}

} // namespace

int main() {
    test_unicode_normalization();
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

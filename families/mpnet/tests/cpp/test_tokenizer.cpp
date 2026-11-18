/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/mpnet/runtime/tokenizer.h"

#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace {

int failures = 0;

std::unique_ptr<trtmc::ITokenizer> tokenizer(bool clean_text, bool add_special_tokens,
                                             bool lowercase = false, bool strip_accents = false) {
    const nlohmann::json config = {
        {"model",
         {{"type", "WordPiece"},
          {"unk_token", "<unk>"},
          {"vocab",
           {{"<s>", 0},
            {"<pad>", 1},
            {"</s>", 2},
            {"<unk>", 3},
            {"ab", 4},
            {"a", 5},
            {"b", 6},
            {u8"a\uE000b", 7},
            {u8"a\u00ADb", 8},
            {u8"a\u0378b", 9},
            {u8"Σ", 10},
            {u8"σ", 11},
            {u8"АБВ", 12},
            {u8"абв", 13},
            {"i", 14},
            {u8"i\u0307", 15},
            {u8"İ", 16},
            {u8"ΟΣ", 17},
            {u8"οσ", 18},
            {"cafe", 19},
            {u8"CAFÉ", 20},
            {u8"\u060C", 21},
            {u8"\u00B7", 22},
            {u8"a\u2044b", 23},
            {u8"a\u3012b", 24},
            {"+", 25},
            {",", 26},
            {u8"a\u00A9b", 27},
            {u8"\U00010100", 28},
            {";", 29},
            {u8"\u3008", 30},
            {u8"\u3009", 31},
            {u8"\u03B1", 32},
            {"viet", 33},
            {u8"\u1100\u1161", 34},
            {u8"\uAC00", 35},
            {u8"\u1EBF", 36},
            {"e", 37},
            {u8"a\u1B44\u302Eb", 38},
            {u8"a\u302E\u1B44b", 39},
            {u8"a\u1AFFb", 40},
            {u8"\u4E3D", 41},
            {u8"\U0001D157\U0001D165", 42}}}}},
        {"normalizer",
         {{"type", "BertNormalizer"},
          {"clean_text", clean_text},
          {"handle_chinese_chars", false},
          {"strip_accents", strip_accents},
          {"lowercase", lowercase}}},
        {"pre_tokenizer", {{"type", "BertPreTokenizer"}}},
        {"post_processor",
         {{"type", "RobertaProcessing"}, {"cls", {"<s>", 0}}, {"sep", {"</s>", 2}}}},
    };
    const auto data = config.dump();
    return trtmc::CreateWordPieceTokenizer(data.data(), data.size(), add_special_tokens);
}

void check(const trtmc::ITokenizer& subject, const std::string& input,
           std::vector<std::int32_t> expected, bool add_special_tokens, const char* name) {
    if (add_special_tokens) {
        expected.insert(expected.begin(), 0);
        expected.push_back(2);
    }
    if (subject.encode(input) != expected) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void test_clean_text(bool add_special_tokens) {
    const auto subject = tokenizer(true, add_special_tokens);
    const std::vector<std::string> controls = {
        u8"\u00AD",     u8"\u0600",     u8"\u061C",     u8"\u06DD",     u8"\u070F",
        u8"\u180E",     u8"\u200D",     u8"\u202E",     u8"\u2060",     u8"\u206F",
        u8"\uE000",     u8"\uFEFF",     u8"\uFFF9",     u8"\U000110BD", u8"\U0001BCA0",
        u8"\U0001D173", u8"\U000E0001", u8"\U000E0020", u8"\U000F0000", u8"\U00100000",
    };
    for (const auto& control : controls)
        check(*subject, "a" + control + "b", {4}, add_special_tokens,
              "clean_text removes format and private-use characters before WordPiece");
    check(*subject, std::string("a\0b", 3), {4}, add_special_tokens, "NUL is removed");
    check(*subject, u8"a\uFFFDb", {4}, add_special_tokens, "replacement character is removed");
    check(*subject,
          "a\x7F"
          "b",
          {4}, add_special_tokens, "ASCII control is removed");
    check(*subject, "a\tb", {5, 6}, add_special_tokens, "tab remains a word boundary");
    check(*subject, "a\nb", {5, 6}, add_special_tokens, "newline remains a word boundary");
    check(*subject, "a\rb", {5, 6}, add_special_tokens, "carriage return remains a word boundary");
    check(*subject, u8"a\u0378b", {9}, add_special_tokens, "unassigned scalar is retained");
}

void test_clean_text_disabled(bool add_special_tokens) {
    const auto subject = tokenizer(false, add_special_tokens);
    check(*subject, u8"a\uE000b", {7}, add_special_tokens, "disabled clean_text keeps private use");
    check(*subject, u8"a\u00ADb", {8}, add_special_tokens, "disabled clean_text keeps format");
}

void test_lowercase(bool add_special_tokens) {
    const auto lower = tokenizer(true, add_special_tokens, true, false);
    check(*lower, u8"Σ", {11}, add_special_tokens, "Greek uppercase is lowered");
    check(*lower, u8"АБВ", {13}, add_special_tokens, "Cyrillic uppercase is lowered");
    check(*lower, u8"İ", {15}, add_special_tokens, "dotted I expands before accent stripping");
    check(*lower, u8"ΟΣ", {18}, add_special_tokens, "Greek sigma uses scalar lowercase");

    const auto strip = tokenizer(true, add_special_tokens, true, true);
    check(*strip, u8"İ", {14}, add_special_tokens, "accent stripping removes the expanded dot");
    check(*strip, u8"CAFÉ", {19}, add_special_tokens, "Latin-1 accent normalization is preserved");

    const auto unchanged = tokenizer(true, add_special_tokens, false, false);
    check(*unchanged, u8"Σ", {10}, add_special_tokens, "disabled lowercase preserves Greek");
    check(*unchanged, u8"АБВ", {12}, add_special_tokens, "disabled lowercase preserves Cyrillic");
    check(*unchanged, u8"İ", {16}, add_special_tokens, "disabled lowercase preserves dotted I");
}

void test_punctuation(bool add_special_tokens) {
    const auto subject = tokenizer(true, add_special_tokens);
    check(*subject, u8"a\u060Cb", {5, 21, 6}, add_special_tokens,
          "Arabic comma isolates a word boundary");
    check(*subject, u8"a\u00B7b", {5, 22, 6}, add_special_tokens,
          "middle dot isolates a word boundary");
    check(*subject, u8"a\U00010100b", {5, 28, 6}, add_special_tokens,
          "supplementary punctuation isolates a word boundary");
    check(*subject, u8"a\u2044b", {23}, add_special_tokens,
          "fraction slash symbol remains inside a word");
    check(*subject, u8"a\u3012b", {24}, add_special_tokens, "postal symbol remains inside a word");
    check(*subject, u8"a\u00A9b", {27}, add_special_tokens,
          "copyright symbol remains inside a word");
    check(*subject, "a+b", {5, 25, 6}, add_special_tokens,
          "ASCII symbol keeps the reference punctuation boundary");
    check(*subject, "a,b", {5, 26, 6}, add_special_tokens, "ASCII comma keeps its boundary");
    const auto no_clean = tokenizer(false, add_special_tokens);
    check(*no_clean, u8"a\u0085b", {5, 6}, add_special_tokens,
          "next-line whitespace separates words when cleaning is disabled");
}

void test_strip_accents(bool add_special_tokens) {
    const auto strip = tokenizer(true, add_special_tokens, false, true);
    check(*strip, u8"\u037E", {29}, add_special_tokens, "Greek question mark decomposes");
    check(*strip, u8"\u0387", {22}, add_special_tokens, "Greek ano teleia decomposes");
    check(*strip, u8"\u2329\u232A", {30, 31}, add_special_tokens,
          "canonical angle brackets decompose before punctuation splitting");
    check(*strip, u8"\u1F00", {32}, add_special_tokens, "Greek accent is removed");
    check(*strip, u8"vi\u1EC7t", {33}, add_special_tokens, "Vietnamese accents are removed");
    check(*strip, u8"\u1EBF", {37}, add_special_tokens, "recursive decomposition removes accents");
    check(*strip, u8"\uAC00", {34}, add_special_tokens, "Hangul decomposes to Jamo");
    check(*strip, u8"\U0002F800", {41}, add_special_tokens,
          "supplementary compatibility ideograph decomposes");
    check(*strip, u8"\U0001D15E", {42}, add_special_tokens,
          "spacing marks survive musical symbol decomposition");
    check(*strip, u8"a\u302Ab", {4}, add_special_tokens,
          "nonspacing mark outside the original blocks is removed");
    check(*strip, u8"a\u1AFFb", {40}, add_special_tokens, "unassigned character is retained");
    check(*strip, u8"a\u302E\u1B44b", {38}, add_special_tokens,
          "retained marks are canonically ordered");
    check(*strip, u8"a\u302E\u034F\u1B44b", {39}, add_special_tokens,
          "removed starter prevents reordering across its boundary");
    check(*strip, u8"a\u302E\uFE0F\u1B44b", {39}, add_special_tokens,
          "removed variation selector preserves its ordering boundary");
    const auto unchanged = tokenizer(true, add_special_tokens, false, false);
    check(*unchanged, u8"\uAC00", {35}, add_special_tokens, "disabled stripping keeps Hangul");
    check(*unchanged, u8"\u1EBF", {36}, add_special_tokens, "disabled stripping keeps accents");
}

} // namespace

int main() {
    for (const bool add_special_tokens : {false, true}) {
        test_clean_text(add_special_tokens);
        test_clean_text_disabled(add_special_tokens);
        test_lowercase(add_special_tokens);
        test_punctuation(add_special_tokens);
        test_strip_accents(add_special_tokens);
    }
    if (failures != 0) {
        std::cerr << "mpnet tokenizer: " << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "mpnet tokenizer: all checks passed\n";
    return 0;
}

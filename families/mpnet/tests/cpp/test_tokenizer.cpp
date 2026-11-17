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
          {"vocab", {{"<s>", 0},        {"<pad>", 1},      {"</s>", 2},  {"<unk>", 3},
                     {"ab", 4},         {"a", 5},          {"b", 6},     {u8"a\uE000b", 7},
                     {u8"a\u00ADb", 8}, {u8"a\u0378b", 9}, {u8"Σ", 10},  {u8"σ", 11},
                     {u8"АБВ", 12},     {u8"абв", 13},     {"i", 14},    {u8"i\u0307", 15},
                     {u8"İ", 16},       {u8"ΟΣ", 17},      {u8"οσ", 18}, {"cafe", 19},
                     {u8"CAFÉ", 20}}}}},
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

} // namespace

int main() {
    for (const bool add_special_tokens : {false, true}) {
        test_clean_text(add_special_tokens);
        test_clean_text_disabled(add_special_tokens);
        test_lowercase(add_special_tokens);
    }
    if (failures != 0) {
        std::cerr << "mpnet tokenizer: " << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "mpnet tokenizer: all checks passed\n";
    return 0;
}

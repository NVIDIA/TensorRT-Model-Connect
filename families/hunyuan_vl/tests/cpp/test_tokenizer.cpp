/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/hunyuan_vl/runtime/tokenizer.h"

#include <iostream>
#include <nlohmann/json.hpp>
#include <unicode/unistr.h>

namespace {
int failures = 0;
void expect(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << '\n';
        ++failures;
    }
}
void expect_ids(const std::vector<int32_t>& actual, const std::vector<int32_t>& expected,
                const char* what) {
    expect(actual == expected, what);
    if (actual != expected) {
        std::cerr << "  actual:";
        for (const auto id : actual)
            std::cerr << ' ' << id;
        std::cerr << '\n';
    }
}
std::string byte_char(int b) {
    int extra = 256;
    for (int i = 0; i < b; ++i)
        if (!((i >= 33 && i <= 126) || (i >= 161 && i <= 172) || i >= 174))
            ++extra;
    const bool literal = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
    std::string result;
    icu::UnicodeString(UChar32(literal ? b : extra)).toUTF8String(result);
    return result;
}
std::string serialized() {
    nlohmann::json vocab = nlohmann::json::object();
    for (int b = 0; b < 256; ++b)
        if (b != 13 && b != 'q') // Like the released vocabulary, some bytes have no token.
            vocab[byte_char(b)] = b;
    vocab["12"] = 256;
    vocab["123"] = 257;
    vocab["a" + byte_char(11)] = 258;
    vocab["ab"] = 259;
    nlohmann::json split_digits = {{"type", "Split"},
                                   {"pattern", {{"Regex", "\\p{N}{1,3}"}}},
                                   {"behavior", "Isolated"},
                                   {"invert", false}};
    nlohmann::json split_spaces = {{"type", "Split"},
                                   {"pattern", {{"Regex", "\\s+"}}},
                                   {"behavior", "Isolated"},
                                   {"invert", false}};
    return nlohmann::json{
        {"normalizer", {{"type", "Sequence"}, {"normalizers", nlohmann::json::array()}}},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers",
           {split_digits,
            split_spaces,
            {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"use_regex", false}}}}}},
        {"decoder", {{"type", "ByteLevel"}}},
        {"added_tokens",
         nlohmann::json::array({nlohmann::json{
             {"id", 300}, {"content", "<|x|>"}, {"special", true}, {"normalized", false}}})},
        {"model",
         {{"type", "BPE"},
          {"dropout", nullptr},
          {"vocab", vocab},
          {"merges",
           nlohmann::json::array(
               {nlohmann::json::array({"1", "2"}), nlohmann::json::array({"12", "3"}),
                nlohmann::json::array({"a", byte_char(11)}), nlohmann::json::array({"a", "b"})})}}}}
        .dump();
}
} // namespace

int main() {
    const trtmc::hunyuan_vl::Tokenizer plain(serialized());
    // Digits isolate in groups of three before the whitespace stage and the merges run.
    expect_ids(plain.encode("a 12345"), {97, 32, 257, 52, 53}, "split stages");
    // U+000B is whitespace for the reference's regex engine, so no merge crosses it.
    expect_ids(plain.encode("a\x0B"
                            "b"),
               {97, 11, 98}, "Unicode whitespace");
    expect_ids(plain.encode("a<|x|>b"), {97, 300, 98}, "added token");
    // Characters outside the vocabulary are dropped before merges, so the merge across them
    // applies.
    expect_ids(plain.encode("aqb"), {259}, "unknown bytes are dropped");
    expect_ids(plain.encode("a\r\n"), {97, 10}, "carriage return has no token");
    expect(plain.decode({97, 300, 98}) == "ab", "special tokens are skipped");
    expect(plain.decode(plain.encode("Hi .")) == "Hi .", "spaces before punctuation are kept");
    return failures == 0 ? 0 : 1;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::ideogram4::test {

// Word, punctuation, and Unicode split coverage; the digit split is applied first.
inline constexpr const char* kSplitRegex =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|"
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|"
    "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

inline std::string encode_code_point(std::uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

// GPT-2 byte-to-unicode alphabet.
inline std::vector<std::string> byte_units() {
    std::vector<std::string> units(256);
    std::uint32_t extra = 256;
    for (int value = 0; value < 256; ++value) {
        const bool printable =
            (value >= 33 && value <= 126) || (value >= 161 && value <= 172) || value >= 174;
        units[static_cast<std::size_t>(value)] =
            encode_code_point(printable ? static_cast<std::uint32_t>(value) : extra++);
    }
    return units;
}

struct Fixture {
    nlohmann::json vocab = nlohmann::json::object();
    std::int32_t next_id = 0;

    std::int32_t add(const std::string& token) {
        if (!vocab.contains(token))
            vocab[token] = next_id++;
        return vocab[token].get<std::int32_t>();
    }
};

inline std::string tokenizer_json(Fixture& fixture) {
    const auto unk = fixture.add("<unk>");
    const auto bos = fixture.add("<s>");
    const auto eos = fixture.add("</s>");
    const auto begin = fixture.add("\\begin{");
    for (const auto& unit : byte_units())
        fixture.add(unit);
    const auto space = byte_units()[' '];
    const std::vector<std::pair<std::string, std::string>> merges{
        {space, "w"}, {"h", "e"}, {"l", "l"}, {"he", "ll"}, {"hell", "o"}};
    nlohmann::json merge_list = nlohmann::json::array();
    for (const auto& [left, right] : merges) {
        fixture.add(left + right);
        merge_list.push_back({left, right});
    }
    fixture.add("World"); // reachable only through ignore_merges
    auto added = [](std::int32_t id, const std::string& content, bool special) {
        return nlohmann::json{{"id", id},          {"content", content}, {"single_word", false},
                              {"lstrip", false},   {"rstrip", false},    {"normalized", false},
                              {"special", special}};
    };
    nlohmann::json tokenizer{
        {"version", "1.0"},
        {"added_tokens",
         {added(unk, "<unk>", true), added(bos, "<s>", true), added(eos, "</s>", true),
          added(begin, "\\begin{", false)}},
        {"normalizer", {{"type", "NFC"}}},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers",
           {{{"type", "Split"},
             {"pattern", {{"Regex", kSplitRegex}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", true},
             {"use_regex", false}}}}}},
        {"post_processor", nullptr},
        {"decoder", {{"type", "ByteLevel"}, {"add_prefix_space", true}, {"use_regex", true}}},
        {"model",
         {{"type", "BPE"},
          {"dropout", nullptr},
          {"unk_token", nullptr},
          {"continuing_subword_prefix", ""},
          {"end_of_word_suffix", ""},
          {"fuse_unk", false},
          {"byte_fallback", false},
          {"ignore_merges", true},
          {"vocab", fixture.vocab},
          {"merges", merge_list}}},
    };
    tokenizer["pre_tokenizer"]["pretokenizers"].insert(
        tokenizer["pre_tokenizer"]["pretokenizers"].begin(),
        nlohmann::json{{"type", "Split"},
                       {"pattern", {{"Regex", "[0-9]"}}},
                       {"behavior", "Isolated"},
                       {"invert", false}});
    return tokenizer.dump();
}

} // namespace trtmc::ideogram4::test

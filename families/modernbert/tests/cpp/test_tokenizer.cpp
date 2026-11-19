/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/modernbert/runtime/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {
std::string utf8(char32_t cp) {
    std::string s;
    if (cp < 0x80)
        s += static_cast<char>(cp);
    else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

nlohmann::json fixture() {
    nlohmann::json vocab = nlohmann::json::object();
    int extra = 256;
    for (int byte = 0; byte < 256; ++byte) {
        const bool printable =
            (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || byte >= 174;
        vocab[utf8(printable ? byte : extra++)] = byte;
    }
    vocab["[CLS]"] = 256;
    vocab["[SEP]"] = 257;
    return {{"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", nlohmann::json::array()}}},
            {"normalizer", {{"type", "NFC"}}},
            {"added_tokens",
             nlohmann::json::array({{{"id", 256}, {"content", "[CLS]"}, {"special", true}},
                                    {{"id", 257}, {"content", "[SEP]"}, {"special", true}}})},
            {"pre_tokenizer",
             {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"use_regex", true}}},
            {"post_processor",
             {{"type", "TemplateProcessing"},
              {"single",
               nlohmann::json::array({{{"SpecialToken", {{"id", "[CLS]"}, {"type_id", 0}}}},
                                      {{"Sequence", {{"id", "A"}, {"type_id", 0}}}},
                                      {{"SpecialToken", {{"id", "[SEP]"}, {"type_id", 0}}}}})},
              {"special_tokens", {{"[CLS]", {{"ids", {256}}}}, {"[SEP]", {{"ids", {257}}}}}}}}};
}
int boundary_cases(bool special) {
    const std::vector<std::pair<std::string, std::vector<std::string>>> cases = {
        {u8"aאb", {u8"aאb"}},
        {u8"a،b", {"a", u8"،", "b"}},
        {u8"1٢3", {u8"1٢3"}},
        {u8"1²2", {u8"1²2"}},
        {u8"1Ⅷ2", {u8"1Ⅷ2"}},
        {u8"aिb", {"a", u8"ि", "b"}},
        {u8"a𐐀b", {u8"a𐐀b"}},
        {u8"aᾀb", {u8"aᾀb"}},
        {u8"a\u2001!", {"a", u8"\u2001", "!"}},
        {"a  b", {"a", " ", " b"}},
        {u8"aあb", {u8"aあb"}},
        {u8"a\uE000b", {"a", u8"\uE000", "b"}},
    };
    int failures = 0;
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& [input, pieces] = cases[i];
        auto config = fixture();
        config["normalizer"] = nullptr;
        auto& vocab = config["model"]["vocab"];
        auto& merges = config["model"]["merges"];
        std::vector<std::string> bytes(256);
        for (const auto& item : vocab.items()) {
            const auto id = item.value().get<std::size_t>();
            if (id < bytes.size())
                bytes[id] = item.key();
        }
        std::set<std::pair<std::string, std::string>> seen;
        const auto add_piece = [&](const std::string& text) {
            std::string prefix;
            for (const unsigned char byte : text) {
                const auto& next = bytes[byte];
                if (!prefix.empty() && seen.emplace(prefix, next).second)
                    merges.push_back(nlohmann::json::array({prefix, next}));
                prefix += next;
                if (!vocab.contains(prefix))
                    vocab[prefix] = vocab.size();
            }
            return vocab[prefix].get<int32_t>();
        };
        for (std::size_t start = 0; start < input.size(); ++start)
            add_piece(input.substr(start));
        std::vector<int32_t> expected;
        if (special)
            expected.push_back(256);
        for (const auto& piece : pieces)
            expected.push_back(add_piece(piece));
        if (special)
            expected.push_back(257);
        const auto data = config.dump();
        const auto tokenizer = trtmc::CreateBpeTokenizer(data.data(), data.size(), special);
        if (tokenizer->encode(input) != expected || tokenizer->encode(input) != expected) {
            std::cerr << "Boundary case " << i << ", special=" << special << " failed\n";
            ++failures;
        }
    }
    return failures;
}
} // namespace

int main() {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"cafe\xCC\x81", "caf\xC3\xA9"},
        {"A\xCC\x8Angstro\xCC\x88m", "\xC3\x85ngstr\xC3\xB6m"},
        {"\xE2\x84\xAB", "\xC3\x85"},
        {"a\xCC\x81\xCC\xA3", "\xE1\xBA\xA1\xCC\x81"},
        {"a\xCC\xA3\xCC\x81", "\xE1\xBA\xA1\xCC\x81"},
        {"\xE1\x84\x92\xE1\x85\xA1\xE1\x86\xAB", "\xED\x95\x9C"},
        {"\xEA\xB0\x80\xE1\x86\xA8", "\xEA\xB0\x81"},
        {"\xCC\x95\xCC\x80", "\xCC\x80\xCC\x95"},
        {"\xCD\x84", "\xCC\x88\xCC\x81"},
        {"a\xCC\x85\xCC\x81", "a\xCC\x85\xCC\x81"},
        {"hello", "hello"},
        {"\xEF\xAC\x81", "\xEF\xAC\x81"},
        {"", ""},
    };
    int failures = 0;
    for (bool special : {false, true}) {
        failures += boundary_cases(special);
        auto config = fixture();
        auto data = config.dump();
        auto tokenizer = trtmc::CreateBpeTokenizer(data.data(), data.size(), special);
        for (std::size_t i = 0; i < cases.size(); ++i) {
            const auto& [input, normalized] = cases[i];
            std::vector<int32_t> expected;
            if (special && !input.empty())
                expected.push_back(256);
            for (unsigned char byte : normalized)
                expected.push_back(byte);
            if (special && !input.empty())
                expected.push_back(257);
            if (tokenizer->encode(input) != expected) {
                std::cerr << "NFC case " << i << ", special=" << special << " failed\n";
                ++failures;
            }
        }
        config["normalizer"] = nullptr;
        data = config.dump();
        auto raw = trtmc::CreateBpeTokenizer(data.data(), data.size(), special);
        if (raw->encode(cases[0].first) == raw->encode(cases[0].second))
            ++failures;
        if (tokenizer->encode("fi") == tokenizer->encode("\xEF\xAC\x81"))
            ++failures;
    }
    std::cout << failures << " failures\n";
    return failures ? 1 : 0;
}

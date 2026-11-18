/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/modernbert/runtime/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-only checks of the MiniCPM-OCR tokenizer's NFC normalizer and Split
// pattern. The fixture is a byte-level BPE with one token per byte (id = byte
// value) plus three merges, with the checkpoint's normalizer and pre-tokenizer.
// A merge shows whether two bytes ended up in one pre-token. Expected ids were
// produced by Hugging Face `tokenizers` from the same fixture.

#include "families/minicpm_ocr/runtime/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// The checkpoint's Split pattern (tokenizer.json pre_tokenizer.pretokenizers[0]).
constexpr const char* kSplitPattern =
    R"re((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)re";

std::string utf8(uint32_t code_point) {
    std::string out;
    if (code_point < 0x80) {
        out += static_cast<char>(code_point);
    } else if (code_point < 0x800) {
        out += static_cast<char>(0xc0 | (code_point >> 6));
        out += static_cast<char>(0x80 | (code_point & 0x3f));
    } else {
        out += static_cast<char>(0xe0 | (code_point >> 12));
        out += static_cast<char>(0x80 | ((code_point >> 6) & 0x3f));
        out += static_cast<char>(0x80 | (code_point & 0x3f));
    }
    return out;
}

// GPT-2 byte-to-character table of byte-level BPE vocabularies.
std::vector<std::string> byte_symbols() {
    std::vector<std::string> symbols(256);
    uint32_t extra = 0;
    for (uint32_t byte = 0; byte < 256; ++byte) {
        const bool printable =
            (byte >= 0x21 && byte <= 0x7e) || (byte >= 0xa1 && byte <= 0xac) || byte >= 0xae;
        symbols[byte] = utf8(printable ? byte : 256 + extra++);
    }
    return symbols;
}

std::string fixture() {
    const auto symbols = byte_symbols();
    auto vocab = nlohmann::json::object();
    for (int32_t byte = 0; byte < 256; ++byte)
        vocab[symbols[static_cast<std::size_t>(byte)]] = byte;
    // " " + 0xC2 (lead of U+00BD), 0xB9 + 0xE0 (across two Devanagari characters),
    // " " + 0xC3 (lead of U+00E9).
    const std::vector<std::pair<int, int>> pairs{{0x20, 0xc2}, {0xb9, 0xe0}, {0x20, 0xc3}};
    auto merges = nlohmann::json::array();
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        const auto& left = symbols[static_cast<std::size_t>(pairs[i].first)];
        const auto& right = symbols[static_cast<std::size_t>(pairs[i].second)];
        vocab[left + right] = 256 + static_cast<int32_t>(i);
        merges.push_back(nlohmann::json::array({left, right}));
    }
    nlohmann::json split = {{"type", "Split"},
                            {"pattern", {{"Regex", kSplitPattern}}},
                            {"behavior", "Isolated"},
                            {"invert", false}};
    nlohmann::json byte_level = {{"type", "ByteLevel"},
                                 {"add_prefix_space", false},
                                 {"trim_offsets", true},
                                 {"use_regex", false}};
    nlohmann::json tokenizer = {
        {"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", merges}}},
        {"normalizer", {{"type", "NFC"}}},
        {"pre_tokenizer",
         {{"type", "Sequence"}, {"pretokenizers", nlohmann::json::array({split, byte_level})}}},
        {"decoder", {{"type", "ByteLevel"}}},
        {"added_tokens", nlohmann::json::array()}};
    return tokenizer.dump();
}

struct Case {
    const char* what;
    std::string text;
    std::vector<int32_t> ids;
};

} // namespace

int main() {
    const std::vector<Case> cases{
        {"NFC composes e + combining acute", "Cafe\xcc\x81", {67, 97, 102, 195, 169}},
        {"NFC maps a CJK compatibility ideograph", "\xef\xa4\x80", {232, 177, 136}},
        {"a vulgar fraction is a number, split from the space", "x \xc2\xbd", {120, 32, 194, 189}},
        {"a spacing vowel sign is not a letter",
         "\xe0\xa4\xb9\xe0\xa4\xbf",
         {224, 164, 185, 224, 164, 191}},
        {"a merge inside one pre-token applies", "x \xc3\xa9", {120, 258, 169}},
    };
    int failures = 0;
    try {
        const auto json = fixture();
        const auto tokenizer = trtmc::minicpm_ocr::CreateBpeTokenizer(json.data(), json.size());
        for (const auto& c : cases) {
            if (tokenizer->encode(c.text) != c.ids) {
                ++failures;
                std::cerr << "FAIL: " << c.what << '\n';
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "minicpm_ocr tokenizer checks passed\n";
    return 0;
}

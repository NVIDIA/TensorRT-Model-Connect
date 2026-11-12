/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The o200k split rule and the byte-level BPE on a miniature tokenizer.json.
// Expected pieces and ids were produced by Hugging Face tokenizers 0.23.3 from
// the same JSON (o200k split rule, 256 byte symbols, eight merges, one entry
// reachable only as a whole piece, two added tokens).

#include "families/openai_privacy_filter/runtime/o200k_tokenizer.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using trtmc::openai_privacy_filter::o200k_split;
using trtmc::openai_privacy_filter::O200kTokenizer;
using Ranges = std::vector<std::pair<std::size_t, std::size_t>>;

void check(bool condition, const std::string& what) {
    if (!condition)
        throw std::runtime_error("openai_privacy_filter tokenizer: " + what);
}

std::string byte_symbol(int byte) {
    const bool printable = (byte >= '!' && byte <= '~') || (byte >= 0xA1 && byte <= 0xAC) ||
                           (byte >= 0xAE && byte <= 0xFF);
    int extra = 0;
    for (int b = 0; b < byte; ++b) {
        const bool p =
            (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        extra += p ? 0 : 1;
    }
    const unsigned cp =
        printable ? static_cast<unsigned>(byte) : 256U + static_cast<unsigned>(extra);
    std::string utf8;
    if (cp < 0x80) {
        utf8.push_back(static_cast<char>(cp));
    } else {
        utf8.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        utf8.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return utf8;
}

constexpr const char* kPattern =
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?)"
    R"(|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?)"
    R"(|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

nlohmann::json miniature() {
    nlohmann::json vocab = nlohmann::json::object();
    for (int byte = 0; byte < 256; ++byte)
        vocab[byte_symbol(byte)] = byte;
    const std::vector<std::pair<std::string, std::string>> merges = {
        {"a", "a"}, {"aa", "aa"}, {"a", "b"}, {"\xC4\xA0", "a"},
        {"H", "e"}, {"He", "l"},  {"l", "o"}, {"Hel", "lo"}};
    nlohmann::json merge_list = nlohmann::json::array();
    for (const auto& [left, right] : merges) {
        if (!vocab.contains(left + right))
            vocab[left + right] = static_cast<int>(vocab.size());
        merge_list.push_back({left, right});
    }
    vocab["\xC4\xA0"
          "aaab"] = static_cast<int>(vocab.size());
    const auto added = [](int id, const char* content) {
        return nlohmann::json{{"id", id},        {"content", content}, {"single_word", false},
                              {"lstrip", false}, {"rstrip", false},    {"normalized", false},
                              {"special", true}};
    };
    return {
        {"version", "1.0"},
        {"added_tokens", {added(265, "<|endoftext|>"), added(266, "<|end|>")}},
        {"normalizer", nullptr},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers",
           {{{"type", "Split"},
             {"pattern", {{"Regex", kPattern}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", true},
             {"use_regex", false}}}}}},
        {"post_processor",
         {{"type", "ByteLevel"},
          {"add_prefix_space", true},
          {"trim_offsets", false},
          {"use_regex", true}}},
        {"model",
         {{"type", "BPE"},
          {"dropout", nullptr},
          {"unk_token", nullptr},
          {"continuing_subword_prefix", nullptr},
          {"end_of_word_suffix", nullptr},
          {"fuse_unk", false},
          {"byte_fallback", false},
          {"ignore_merges", true},
          {"vocab", vocab},
          {"merges", merge_list}}},
    };
}

struct Case {
    std::string text;
    Ranges splits; // empty when the text holds added tokens
    std::vector<std::int32_t> ids;
};

const std::vector<Case>& cases() {
    static const std::vector<Case> all = {
        {"HelloWorld helloWorld HELLOworld helloWORLD",
         {{0, 5}, {5, 10}, {10, 16}, {16, 21}, {21, 32}, {32, 38}, {38, 43}},
         {263, 87, 111, 114, 108, 100, 32,  104, 101, 108, 262, 87,  111, 114, 108, 100, 32, 72, 69,
          76,  76, 79,  119, 111, 114, 108, 100, 32,  104, 101, 108, 262, 87,  79,  82,  76, 68}},
        {"don't I'M they'RE we'Ve she'LL he'd it'\xC5\xBF",
         {{0, 5}, {5, 9}, {9, 17}, {17, 23}, {23, 30}, {30, 35}, {35, 41}},
         {100, 111, 110, 39,  116, 32,  73,  39, 77,  32,  116, 104, 101, 121,
          39,  82,  69,  32,  119, 101, 39,  86, 101, 32,  115, 104, 101, 39,
          76,  76,  32,  104, 101, 39,  100, 32, 105, 116, 39,  197, 191}},
        {"1234567 +49 30 1234567",
         {{0, 3},
          {3, 6},
          {6, 7},
          {7, 9},
          {9, 11},
          {11, 12},
          {12, 14},
          {14, 15},
          {15, 18},
          {18, 21},
          {21, 22}},
         {49, 50, 51, 52, 53, 54, 55, 32, 43, 52, 57, 32, 51, 48, 32, 49, 50, 51, 52, 53, 54, 55}},
        {"...\n\n/// //\n !!! ?!/\r\n",
         {{0, 8}, {8, 12}, {12, 16}, {16, 22}},
         {46, 46, 46, 10, 10, 47, 47, 47, 32, 47, 47, 10, 32, 33, 33, 33, 32, 63, 33, 47, 13, 10}},
        {"a  b   c\t\td \xC2\xA0"
         "e\xE3\x80\x80"
         "f",
         {{0, 1}, {1, 2}, {2, 4}, {4, 6}, {6, 8}, {8, 9}, {9, 11}, {11, 12}, {12, 15}, {15, 19}},
         {97, 32, 32, 98, 32, 32, 32, 99, 9, 9, 100, 32, 194, 160, 101, 227, 128, 128, 102}},
        {"  \n  \n  x  ",
         {{0, 6}, {6, 7}, {7, 9}, {9, 11}},
         {32, 32, 10, 32, 32, 10, 32, 32, 120, 32, 32}},
        {"line1\r\nline2\n\n\n",
         {{0, 4}, {4, 5}, {5, 7}, {7, 11}, {11, 12}, {12, 15}},
         {108, 105, 110, 101, 49, 13, 10, 108, 105, 110, 101, 50, 10, 10, 10}},
        {"e\xCC\x81 cafe\xCC\x81 \xCC\x81x",
         {{0, 3}, {3, 10}, {10, 14}},
         {101, 204, 129, 32, 99, 97, 102, 101, 204, 129, 32, 204, 129, 120}},
        {"\xCE\x95\xCE\xBB\xCE\xBB\xCE\xB7\xCE\xBD\xCE\xB9\xCE\xBA\xCE\xAC "
         "\xCE\x91\xCE\x92\xCE\x93 "
         "\xCE\xB1\xCE\xB2\xCE\xB3 \xD0\x98\xD0\xB2\xD0\xB0\xD0\xBD "
         "\xD0\x98\xD0\x92\xD0\x90\xD0\x9D",
         {{0, 16}, {16, 23}, {23, 30}, {30, 39}, {39, 48}},
         {206, 149, 206, 187, 206, 187, 206, 183, 206, 189, 206, 185, 206, 186, 206, 172,
          32,  206, 145, 206, 146, 206, 147, 32,  206, 177, 206, 178, 206, 179, 32,  208,
          152, 208, 178, 208, 176, 208, 189, 32,  208, 152, 208, 146, 208, 144, 208, 157}},
        {"aaaa aaab ab<|endoftext|>ba<|end|>", {}, {257, 264, 32, 258, 265, 98, 97, 266}},
        {"<|endoftext|><|endoftext|>", {}, {265, 265}},
        {"", {}, {}},
    };
    return all;
}

void splits_and_ids() {
    const auto tokenizer = O200kTokenizer::from_json(miniature().dump());
    check(tokenizer.vocabulary_size() == 267, "vocabulary size");
    for (const auto& item : cases()) {
        if (!item.splits.empty())
            check(o200k_split(item.text) == item.splits, "splits of \"" + item.text + "\"");
        if (!item.ids.empty())
            check(tokenizer.encode(item.text) == item.ids, "ids of \"" + item.text + "\"");
    }
    // A string literal cannot hold the leading NUL of this case; build it explicitly.
    const std::string controls("\x00\x1C\xE2\x80\x8B\xEF\xBB\xBF", 8);
    check(o200k_split(controls) == Ranges{{0, 8}}, "control and format characters form one piece");
    check(tokenizer.encode(controls) ==
              std::vector<std::int32_t>({0, 28, 226, 128, 139, 239, 187, 191}),
          "control and format characters");
    check(tokenizer.encode("").empty() && o200k_split("").empty(), "empty text");
}

template <class Function>
void rejects(Function&& function, const std::string& what) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("openai_privacy_filter tokenizer accepted " + what);
}

void rejects_malformed_text() {
    const auto tokenizer = O200kTokenizer::from_json(miniature().dump());
    const std::vector<std::string> malformed = {"\xFF", "a\xC3", "\xC0\xAF", "\xED\xA0\x80",
                                                "\xF4\x90\x80\x80"};
    for (const auto& text : malformed)
        rejects([&] { tokenizer.encode(text); }, "malformed UTF-8");
}

void rejects_other_configurations() {
    const std::vector<std::pair<std::string, std::function<void(nlohmann::json&)>>> edits = {
        {"a normalizer", [](nlohmann::json& j) { j["normalizer"] = {{"type", "NFC"}}; }},
        {"another split rule",
         [](nlohmann::json& j) {
             j["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = "\\s+";
         }},
        {"a prefix space",
         [](nlohmann::json& j) {
             j["pre_tokenizer"]["pretokenizers"][1]["add_prefix_space"] = true;
         }},
        {"a framing post-processor",
         [](nlohmann::json& j) { j["post_processor"] = {{"type", "TemplateProcessing"}}; }},
        {"merge-only BPE", [](nlohmann::json& j) { j["model"]["ignore_merges"] = false; }},
        {"byte fallback", [](nlohmann::json& j) { j["model"]["byte_fallback"] = true; }},
        {"a stripping added token",
         [](nlohmann::json& j) { j["added_tokens"][0]["lstrip"] = true; }},
        {"renumbered added tokens", [](nlohmann::json& j) { j["added_tokens"][0]["id"] = 300; }},
        {"a missing byte symbol", [](nlohmann::json& j) { j["model"]["vocab"].erase("A"); }},
    };
    for (const auto& [what, edit] : edits) {
        auto json = miniature();
        edit(json);
        rejects([&] { O200kTokenizer::from_json(json.dump()); }, what);
    }
}

} // namespace

int main() {
    try {
        splits_and_ids();
        rejects_malformed_text();
        rejects_other_configurations();
        std::cout << "openai_privacy_filter tokenizer contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

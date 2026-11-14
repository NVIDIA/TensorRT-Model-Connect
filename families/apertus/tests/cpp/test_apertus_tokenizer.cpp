/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// CPU checks of the Apertus byte-level BPE on a small tokenizer.json that uses the
// checkpoint's split expression, added-token flags, ignore_merges and decoder.

#include "families/apertus/runtime/tokenizer.h"

#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Split expression serialized in swiss-ai/Apertus-8B-Instruct-2509 tokenizer.json.
const char* kSplitRegex =
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|"
    "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|"
    "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

std::string encode_code_point(std::uint32_t cp) {
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
std::vector<std::string> byte_units() {
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

std::string tokenizer_json(Fixture& fixture) {
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
        {"normalizer", nullptr},
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
          {"continuing_subword_prefix", nullptr},
          {"end_of_word_suffix", nullptr},
          {"fuse_unk", false},
          {"byte_fallback", false},
          {"ignore_merges", true},
          {"vocab", fixture.vocab},
          {"merges", merge_list}}},
    };
    return tokenizer.dump();
}

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "FAILED: " << label << '\n';
        ++failures;
    }
}

std::vector<std::int32_t> ids(Fixture& fixture, const std::vector<std::string>& tokens) {
    std::vector<std::int32_t> result;
    for (const auto& token : tokens)
        result.push_back(fixture.vocab.at(token).get<std::int32_t>());
    return result;
}

} // namespace

int main() {
    Fixture fixture;
    const trtmc::apertus::Tokenizer tokenizer(tokenizer_json(fixture));
    const auto space = byte_units()[' '];

    expect(tokenizer.encode("hello") == ids(fixture, {"hello"}), "merges build one word");
    expect(tokenizer.encode("World") == ids(fixture, {"World"}),
           "ignore_merges keeps a vocab word");
    expect(tokenizer.encode("helloWorld") == ids(fixture, {"hello", "World"}),
           "the split expression separates a lower-case run from a capitalized word");
    expect(tokenizer.encode("123") == ids(fixture, {"1", "2", "3"}), "digits are split singly");
    expect(tokenizer.encode(" world") == ids(fixture, {space + "w", "o", "r", "l", "d"}),
           "a leading space joins the word and merges by rank");
    expect(tokenizer.encode("<s>hello</s>") ==
               std::vector<std::int32_t>{1, fixture.vocab.at("hello").get<std::int32_t>(), 2},
           "special tokens inside the text are matched before splitting");
    expect(tokenizer.encode("a\\begin{b") ==
               std::vector<std::int32_t>{fixture.vocab.at("a").get<std::int32_t>(), 3,
                                         fixture.vocab.at("b").get<std::int32_t>()},
           "non-special added tokens are matched");
    expect(tokenizer.decode({1, fixture.vocab.at("hello").get<std::int32_t>(), 2}) == "hello",
           "decode skips special tokens");
    expect(tokenizer.decode({3}) == "\\begin{", "decode keeps non-special added tokens");
    const std::string multilingual = "Gr\xC3\xBC"
                                     "ezi mitenand,\n\tZ\xC3\xBCrich 2025! "
                                     "\xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E camelCaseHTTPServer";
    expect(tokenizer.decode(tokenizer.encode(multilingual)) == multilingual,
           "UTF-8 text survives an encode/decode round trip");
    expect(tokenizer.vocab_size() == fixture.next_id, "vocabulary size counts every ID");
    try {
        (void)tokenizer.decode({fixture.next_id});
        expect(false, "an unknown token ID is rejected");
    } catch (const std::invalid_argument&) {
    }
    if (failures == 0)
        std::cout << "test_apertus_tokenizer: all checks passed\n";
    return failures == 0 ? 0 : 1;
}

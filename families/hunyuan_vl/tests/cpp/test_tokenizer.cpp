/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/hunyuan_vl/runtime/tokenizer.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
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
    // Ids stay dense, as in the released vocabulary, so added tokens start at the entry count.
    vocab["xy"] = 13;
    vocab["yx"] = int('q');
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
             {"id", 260}, {"content", "<|x|>"}, {"special", true}, {"normalized", false}}})},
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
// Loads the fixture after `change`; true when the tokenizer rejects it.
bool rejected(const std::function<void(nlohmann::json&)>& change) {
    auto document = nlohmann::json::parse(serialized());
    change(document);
    try {
        trtmc::hunyuan_vl::Tokenizer tokenizer(document.dump());
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}
std::string decoded(const trtmc::hunyuan_vl::Tokenizer& tokenizer,
                    const std::vector<int32_t>& ids) {
    try {
        return tokenizer.decode(ids);
    } catch (const std::invalid_argument&) {
        return "<rejected>";
    }
}
bool budget_exceeded(const trtmc::hunyuan_vl::Tokenizer& tokenizer, const std::string& text,
                     size_t budget) {
    try {
        tokenizer.encode(text, budget);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}
// True when the tokenizer rejects the serialized text as given.
bool text_rejected(const std::string& text) {
    try {
        trtmc::hunyuan_vl::Tokenizer tokenizer(text);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}
nlohmann::json added_token(int id, const std::string& content) {
    return {{"id", id}, {"content", content}, {"special", true}, {"normalized", false}};
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
    expect_ids(plain.encode("a<|x|>b"), {97, 260, 98}, "added token");
    // Characters outside the vocabulary are dropped before merges, so the merge across them
    // applies.
    expect_ids(plain.encode("aqb"), {259}, "unknown bytes are dropped");
    expect_ids(plain.encode("a\r\n"), {97, 10}, "carriage return has no token");
    expect(plain.decode({97, 260, 98}) == "ab", "special tokens are skipped");
    expect(decoded(plain, {97, 1000, -5, 98}) == "ab",
           "ids without a token decode to nothing, as in the reference");
    // A token with a character that is no byte symbol keeps its own UTF-8, as in the reference.
    auto mixed = nlohmann::json::parse(serialized());
    auto emoji = added_token(261, "\xC3\xA9\xF0\x9F\x99\x82");
    emoji["special"] = false;
    mixed["added_tokens"].push_back(emoji);
    const trtmc::hunyuan_vl::Tokenizer with_emoji(mixed.dump());
    expect(with_emoji.decode({97, 261, 98}) == "a\xC3\xA9\xF0\x9F\x99\x82"
                                               "b",
           "a token that is not all byte symbols decodes whole");
    expect(plain.decode(plain.encode("Hi .")) == "Hi .", "spaces before punctuation are kept");
    // Token ids are integers inside int32, each base id names one token, and added tokens never
    // remap an id or a content.
    expect(rejected([](auto& j) { j["model"]["vocab"]["zz"] = 97; }), "duplicate base id");
    expect(rejected([](auto& j) { j["model"]["vocab"]["zz"] = 400.5; }), "fractional id");
    expect(rejected([](auto& j) { j["model"]["vocab"]["zz"] = true; }), "boolean id");
    expect(rejected([](auto& j) { j["model"]["vocab"]["zz"] = -1; }), "negative id");
    // Truncation, padding and a post-processor that adds tokens are not implemented.
    expect(!rejected([](auto& j) {
        j["truncation"] = nullptr;
        j["padding"] = nullptr;
        j["post_processor"] = {{"type", "ByteLevel"}, {"trim_offsets", false}};
    }),
           "no truncation or padding and an offsets-only post-processor");
    expect(rejected([](auto& j) { j["truncation"] = {{"max_length", 8}}; }), "truncation");
    expect(rejected([](auto& j) { j["padding"] = {{"pad_id", 0}}; }), "padding");
    expect(rejected([](auto& j) { j["post_processor"] = {{"type", "TemplateProcessing"}}; }),
           "a post-processor that adds tokens");
    // Characters the vocabulary lacks are dropped, so an unknown token is not supported.
    expect(rejected([](auto& j) { j["model"]["unk_token"] = "a"; }), "unknown token");
    expect(rejected([](auto& j) { j["added_tokens"].push_back(added_token(97, "<|y|>")); }),
           "added token reusing a base id");
    expect(rejected([](auto& j) { j["added_tokens"].push_back(added_token(261, "<|x|>")); }),
           "added content with a second id");
    // A new added token takes the next id after the entries and the earlier added tokens.
    expect(!rejected([](auto& j) { j["added_tokens"].push_back(added_token(261, "<|z|>")); }),
           "added token at the reference's next id");
    expect(rejected([](auto& j) { j["added_tokens"].push_back(added_token(262, "<|z|>")); }),
           "added token past the reference's next id");
    expect(!rejected([](auto& j) { j["added_tokens"].push_back(added_token(97, "a")); }),
           "added token repeating a base token");
    // A base token registered again keeps its id and takes no number, even a high one.
    const auto high_base = [](nlohmann::json& j, int next) {
        j["model"]["vocab"]["zz"] = 400;
        j["added_tokens"][0]["id"] = 261;
        j["added_tokens"].push_back(added_token(400, "zz"));
        j["added_tokens"].push_back(added_token(next, "<|z|>"));
    };
    expect(!rejected([&](auto& j) { high_base(j, 262); }),
           "a high base id registered again takes no number");
    expect(rejected([&](auto& j) { high_base(j, 401); }),
           "an added id counted from a high base id");
    // A merge has two operands; operands and result are in the vocabulary.
    expect(rejected([](auto& j) {
               j["model"]["merges"].push_back(nlohmann::json::array({"1", "2", "3"}));
           }),
           "merge with three operands");
    expect(rejected([](auto& j) { j["model"]["merges"].push_back("1 2 3"); }),
           "merge string with three operands");
    expect(rejected(
               [](auto& j) { j["model"]["merges"].push_back(nlohmann::json::array({"q", "1"})); }),
           "merge operand outside the vocabulary");
    expect(rejected(
               [](auto& j) { j["model"]["merges"].push_back(nlohmann::json::array({"b", "c"})); }),
           "merge result outside the vocabulary");
    // A budget bounds the tokens: exceeded while encoding, or, for a text whose length alone
    // needs more, before it is tokenized.
    expect_ids(plain.encode("a 12345", 5), {97, 32, 257, 52, 53}, "text within its budget");
    expect(budget_exceeded(plain, "a 12345", 4), "text over its budget");
    expect(budget_exceeded(plain, std::string(size_t(1) << 26, 'a'), 1000),
           "text whose length alone exceeds its budget");
    // Gathered text parts fail before a part that would exceed the budget is copied: here a token
    // spans at most 5 bytes (the added token).
    trtmc::hunyuan_vl::TextRun run(plain, 2);
    run.append("ab");
    run.append("cdefg");
    expect(run.text() == "abcdefg", "parts within the budget are gathered");
    bool over = false;
    try {
        run.append("hijk");
    } catch (const std::invalid_argument&) {
        over = true;
    }
    expect(over && run.text() == "abcdefg", "a part over the budget fails before it is copied");
    trtmc::hunyuan_vl::TextRun many(plain, 1000);
    size_t parts = 0;
    try {
        for (; parts < 100000; ++parts)
            many.append("aaaaa");
    } catch (const std::invalid_argument&) {
    }
    expect(parts == 1000 && many.text().size() == 5000, "many small parts stop at the budget");
    run.restart(1);
    run.append("a");
    expect(run.text() == "a", "a restarted run is empty");

    // Bytes the vocabulary drops count too: carriage returns are bounded like any other text.
    expect(budget_exceeded(plain, std::string(100, '\r'), 10), "carriage returns past the bound");
    trtmc::hunyuan_vl::TextRun returns(plain, 1);
    returns.append(std::string(5, '\r'));
    bool refused = false;
    try {
        returns.append("\r");
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    expect(refused && returns.text().size() == 5, "carriage-return parts stop at the bound");

    // Merges take O(n log n): a million-byte piece that merges pairwise.
    auto document = nlohmann::json::parse(serialized());
    document["model"]["vocab"]["aa"] = 260;
    document["model"]["merges"].push_back(nlohmann::json::array({"a", "a"}));
    document["added_tokens"][0]["id"] = 261;
    const trtmc::hunyuan_vl::Tokenizer pairs(document.dump());
    const auto merged = pairs.encode(std::string(1000000, 'a'));
    expect(merged.size() == 500000 &&
               std::all_of(merged.begin(), merged.end(), [](int32_t id) { return id == 260; }),
           "a long piece merges pairwise");
    expect_ids(pairs.encode("aaa"), {260, 97}, "equal ranks merge leftmost first");
    expect_ids(pairs.encode("aab"), {97, 259}, "the lowest rank merges first");

    // A repeated key would silently replace the earlier value, so such text is rejected.
    const auto text = serialized();
    expect(!text_rejected(text), "the fixture text is accepted");
    expect(text_rejected("{\"model\":{}," + text.substr(1)), "repeated top-level key");
    auto vocab = text;
    vocab.replace(vocab.find("\"vocab\":{"), 9, "\"vocab\":{\"a\":98,");
    expect(text_rejected(vocab), "repeated vocabulary entry");
    return failures == 0 ? 0 : 1;
}

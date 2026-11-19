/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// What the native tokenizer does beyond splitting text: the order in which it applies merges,
// the encoding its configuration selects, the tokenizer.json files it refuses to load, the bytes
// it keeps of text that is not UTF-8, and the prompt frame it hands to the text encoder.
//
// tests/data/merge_tokenizer.json has a merge table on which the reference order (one merge at a
// time, lowest rank first, leftmost pair among equal ranks) and merging every occurrence of the
// best pair per pass give different ids. tests/data/merge_cases.json holds the ids the Hugging
// Face library produces with it, and with the same tokenizer after U+2581 became an added token;
// test_merge_reference.py checks them against the library.
//
//   test_qwen_image2_1_tokenizer MERGE_TOKENIZER_JSON MERGE_CASES_JSON BOUNDARY_TOKENIZER_JSON
#include "families/qwen_image2_1/runtime/prompt_tokens.h"
#include "families/qwen_image2_1/runtime/tokenizer.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using nlohmann::json;
using trtmc::qwen_image2_1::RuntimeConfig;
using trtmc::qwen_image2_1::tokenize_prompt;

std::size_t failures = 0;

void expect(bool condition, const std::string& message) {
    if (condition)
        return;
    ++failures;
    std::cerr << "FAILED: " << message << '\n';
}

std::string read_file(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error(std::string("cannot open ") + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// More ids than any fixture defines; the checks of the limit itself pass their own.
constexpr std::int32_t kTokenIdLimit = 1024;

std::unique_ptr<trtmc::ITokenizer> load(const json& definition,
                                        std::int32_t token_id_limit = kTokenIdLimit) {
    const std::string document = definition.dump();
    auto tokenizer =
        trtmc::CreateBpeTokenizer(document.data(), document.size(), false, token_id_limit);
    if (!tokenizer)
        throw std::runtime_error("the definition did not give a tokenizer");
    return tokenizer;
}

// Every case of the merge fixture must encode to the ids of the reference.
void check_merge_order(const json& definition, const json& document) {
    const auto tokenizer = load(definition);
    std::size_t checked = 0;
    for (const auto& item : document.at("cases")) {
        const auto text = item.at("text").get<std::string>();
        const auto expected = item.at("ids").get<std::vector<std::int32_t>>();
        const auto actual = tokenizer->encode(text);
        ++checked;
        expect(actual == expected, "merge order of " + json(text).dump() + ": reference " +
                                       json(expected).dump() + ", native " + json(actual).dump());
    }
    expect(checked >= 40, "the merge case file holds too few cases");
    std::cout << checked << " texts merged\n";
}

// The merge tokenizer with U+2581, which SentencePiece vocabularies write for a space, as one
// more added token. Its pre-tokenizer is byte-level, so text without that character must encode
// as before, and the character itself as the added token.
void check_space_marker(const json& valid, const json& document) {
    auto definition = valid;
    auto marker = definition.at("added_tokens").back();
    marker["id"] = marker.at("id").get<std::int32_t>() + 1;
    marker["content"] = "\xe2\x96\x81";
    marker["special"] = false;
    definition["added_tokens"].push_back(marker);
    const auto tokenizer = load(definition);
    std::size_t checked = 0;
    for (const auto& item : document.at("space_marker_cases")) {
        const auto text = item.at("text").get<std::string>();
        const auto expected = item.at("ids").get<std::vector<std::int32_t>>();
        const auto actual = tokenizer->encode(text);
        ++checked;
        expect(actual == expected, "with U+2581 as an added token, " + json(text).dump() +
                                       ": reference " + json(expected).dump() + ", native " +
                                       json(actual).dump());
    }
    expect(checked >= 8, "the case file holds too few space-marker cases");
}

// A definition the tokenizer does not reproduce must fail to load, with a message naming it.
void expect_refused(const json& definition, const std::string& fragment,
                    std::int32_t token_id_limit = kTokenIdLimit) {
    try {
        load(definition, token_id_limit);
    } catch (const std::exception& error) {
        expect(std::string(error.what()).find(fragment) != std::string::npos,
               "the load error \"" + std::string(error.what()) + "\" does not mention \"" +
                   fragment + "\"");
        return;
    }
    expect(false, "a tokenizer.json with " + fragment + " loaded");
}

// Every id the tokenizer can return is below the limit the model gives it: the largest id of
// the merge tokenizer is 266.
void check_token_id_limit(const json& valid) {
    expect(load(valid, 267)->encode("<|im_end|>") == std::vector<std::int32_t>({266}),
           "a limit just above the largest id loads");
    expect_refused(valid, "added_tokens[1] has the id 266; the model takes ids below 266", 266);
    expect_refused(valid, "has the id 265; the model takes ids below 265", 265);
    expect_refused(valid, "model.vocab entry", 100);

    // An id far outside is refused before it sizes anything.
    for (const std::uint64_t id : {std::uint64_t{2000000000}, std::uint64_t{1} << 40,
                                   std::numeric_limits<std::uint64_t>::max()}) {
        auto definition = valid;
        definition["added_tokens"][1]["id"] = id;
        expect_refused(definition, "added_tokens[1] has the id " + std::to_string(id));
        definition = valid;
        definition["model"]["vocab"]["g"] = id;
        expect_refused(definition, "model.vocab entry \"g\" has the id " + std::to_string(id));
    }

    // Ids a post-processor would add are held to the same limit.
    auto definition = valid;
    definition["post_processor"] = {{"type", "RobertaProcessing"},
                                    {"cls", json::array({"a", 97})},
                                    {"sep", json::array({"b", 5000})}};
    expect_refused(definition, "post_processor.sep has the id 5000");
    definition["post_processor"]["sep"][1] = "98";
    expect_refused(definition, "post_processor.sep needs a token id");

    for (const std::int32_t limit : {0, -1}) {
        bool rejected = false;
        try {
            load(valid, limit);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected, "a tokenizer without a positive id limit must be rejected");
    }
}

void check_refused_definitions(const json& valid) {
    // Complete added-token records only.
    for (const char* field : {"id", "content"}) {
        auto definition = valid;
        definition["added_tokens"][0].erase(field);
        expect_refused(definition, "added_tokens[0]");
    }
    for (const auto& [field, value] : std::vector<std::pair<std::string, json>>{
             {"id", "265"}, {"id", -1}, {"id", 1.5}, {"content", 7}, {"content", ""}}) {
        auto definition = valid;
        definition["added_tokens"][1][field] = value;
        expect_refused(definition, "added_tokens[1]");
    }
    {
        auto definition = valid;
        definition["added_tokens"][0] = "<|im_start|>";
        expect_refused(definition, "added_tokens[0]");
        definition["added_tokens"] = json::object({{"<|im_start|>", 265}});
        expect_refused(definition, "added_tokens must be an array");
    }

    // Added tokens match literally; the options that restrict or widen a match are refused.
    for (const char* option : {"single_word", "lstrip", "rstrip"}) {
        auto definition = valid;
        definition["added_tokens"][1][option] = true;
        expect_refused(definition, std::string("sets ") + option);
    }

    // Model options that change ids and are not implemented.
    for (const auto& [option, value] :
         std::vector<std::pair<std::string, json>>{{"ignore_merges", true},
                                                   {"fuse_unk", true},
                                                   {"byte_fallback", true},
                                                   {"dropout", 0.1},
                                                   {"unk_token", "<unk>"},
                                                   {"continuing_subword_prefix", "##"}}) {
        auto definition = valid;
        definition["model"][option] = value;
        expect_refused(definition, "model." + option);
    }

    // Merge rules are pairs of vocabulary tokens with a result in the vocabulary.
    for (const auto& rule : std::vector<json>{json::array({"a"}), json(7), json("ab")}) {
        auto definition = valid;
        definition["model"]["merges"].push_back(rule);
        expect_refused(definition, "is not a pair of tokens");
    }
    for (const auto& rule : std::vector<json>{json::array({"a", "zz"}), json::array({"g", "g"})}) {
        auto definition = valid;
        definition["model"]["merges"].push_back(rule);
        expect_refused(definition, "outside the vocabulary");
    }
    {
        auto definition = valid;
        definition["model"]["vocab"]["a"] = "97";
        expect_refused(definition, "model.vocab");
    }

    check_token_id_limit(valid);

    // The same options left out, null or false are the supported configuration.
    auto definition = valid;
    for (auto& token : definition["added_tokens"]) {
        token.erase("single_word");
        token["lstrip"] = nullptr;
    }
    definition["model"].erase("ignore_merges");
    definition["model"]["continuing_subword_prefix"] = nullptr;
    const auto tokenizer = load(definition);
    expect(tokenizer->encode("ababa") == std::vector<std::int32_t>({256, 98, 97}),
           "a definition without the optional fields encodes like the complete one");
}

// The prompt frame of the boundary tokenizer: one id per byte, 256 more at the end of a piece,
// <|im_start|> = 512 and <|im_end|> = 513.
void check_prompt_frame(const json& boundary) {
    const auto tokenizer = load(boundary);
    RuntimeConfig config;
    config.prompt_prefix = "<|im_start|>s<|im_end|><|im_start|>u\n";
    config.prompt_suffix = "<|im_end|>";
    config.prefix_tokens = 3; // the system turn, dropped from the encoder output
    config.text_tokens = 8;
    config.pad_token_id = 999;

    const auto framed = tokenize_prompt(*tokenizer, config, "hi");
    expect(framed.ids == std::vector<std::int32_t>({512, 's' + 256, 513, 512, 'u' + 256, '\n' + 256,
                                                    'h', 'i' + 256, 513, 999, 999}),
           "the frame is tokenized and padded to prefix + text tokens: " + json(framed.ids).dump());
    expect(framed.length == 6, "text tokens after the system turn");

    const auto empty = tokenize_prompt(*tokenizer, config, "");
    expect(empty.length == 5 && empty.ids[6] == ' ' + 256, "an empty prompt is read as a space");

    bool rejected = false;
    try {
        tokenize_prompt(*tokenizer, config, "hi there you");
    } catch (const std::invalid_argument& error) {
        rejected = std::string(error.what()).find("needs 16 text tokens") != std::string::npos &&
                   std::string(error.what()).find("built for 8") != std::string::npos;
        expect(rejected, std::string("over-budget message: ") + error.what());
    }
    expect(rejected, "a prompt over the text budget must be rejected as std::invalid_argument");

    config.prefix_tokens = 50;
    bool lost = false;
    try {
        tokenize_prompt(*tokenizer, config, "hi");
    } catch (const std::runtime_error&) {
        lost = true;
    }
    expect(lost, "a frame shorter than the system turn must be rejected");
}

// Text that is not UTF-8 has no reference ids. The tokenizer keeps every byte of it: a sequence
// that the end of the text cuts off is split like a character that is neither a letter, a number
// nor whitespace. Complete sequences in the same place decode as usual.
void check_cut_off_sequences(const json& boundary) {
    const auto tokenizer = load(boundary);
    const auto last = [](unsigned char byte) { return std::int32_t{256} + byte; };
    const std::vector<std::pair<std::string, std::vector<std::int32_t>>> cases{
        // A letter of two bytes joins the word; cut off, its first byte is a piece of its own.
        {"a\xC3\xA9", {'a', 0xC3, last(0xA9)}},
        {"a\xC3", {last('a'), last(0xC3)}},
        // U+20AC and U+1F600 are symbols.
        {"a\xE2\x82\xAC", {last('a'), 0xE2, 0x82, last(0xAC)}},
        {"a\xE2\x82", {last('a'), 0xE2, last(0x82)}},
        {"a\xE2", {last('a'), last(0xE2)}},
        {"a\xF0\x9F\x98\x80", {last('a'), 0xF0, 0x9F, 0x98, last(0x80)}},
        {"a\xF0\x9F\x98", {last('a'), 0xF0, 0x9F, last(0x98)}},
        {"a\xF0\x9F", {last('a'), 0xF0, last(0x9F)}},
        {"a\xF0", {last('a'), last(0xF0)}},
        // U+20000 is a letter of four bytes.
        {"a\xF0\xA0\x80\x80", {'a', 0xF0, 0xA0, 0x80, last(0x80)}},
        {"\xF0", {last(0xF0)}},
    };
    for (const auto& [text, expected] : cases) {
        const auto actual = tokenizer->encode(text);
        expect(actual == expected,
               "bytes " + json(std::vector<std::uint8_t>(text.begin(), text.end())).dump() +
                   ": expected " + json(expected).dump() + ", native " + json(actual).dump());
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " MERGE_TOKENIZER_JSON MERGE_CASES_JSON BOUNDARY_TOKENIZER_JSON\n";
        return 2;
    }
    try {
        const auto merge_tokenizer = json::parse(read_file(argv[1]));
        const auto merge_cases = json::parse(read_file(argv[2]));
        const auto boundary_tokenizer = json::parse(read_file(argv[3]));
        check_merge_order(merge_tokenizer, merge_cases);
        check_space_marker(merge_tokenizer, merge_cases);
        check_refused_definitions(merge_tokenizer);
        check_cut_off_sequences(boundary_tokenizer);
        check_prompt_frame(boundary_tokenizer);
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "tokenizer contract checks passed\n";
    return 0;
}

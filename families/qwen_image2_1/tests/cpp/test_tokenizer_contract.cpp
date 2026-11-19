/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// What the native tokenizer does beyond splitting text: the order in which it applies merges,
// the tokenizer.json files it refuses to load, and the prompt frame it hands to the text encoder.
//
// tests/data/merge_tokenizer.json has a merge table on which the reference order (one merge at a
// time, lowest rank first, leftmost pair among equal ranks) and merging every occurrence of the
// best pair per pass give different ids. tests/data/merge_cases.json holds the ids the Hugging
// Face library produces with it; test_merge_reference.py checks them against the library.
//
//   test_qwen_image2_1_tokenizer MERGE_TOKENIZER_JSON MERGE_CASES_JSON BOUNDARY_TOKENIZER_JSON
#include "families/qwen_image2_1/runtime/prompt_tokens.h"
#include "families/qwen_image2_1/runtime/tokenizer.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
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

std::unique_ptr<trtmc::ITokenizer> load(const json& definition) {
    const std::string document = definition.dump();
    auto tokenizer = trtmc::CreateBpeTokenizer(document.data(), document.size(), false);
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

// A definition the tokenizer does not reproduce must fail to load, with a message naming it.
void expect_refused(const json& definition, const std::string& fragment) {
    try {
        load(definition);
    } catch (const std::exception& error) {
        expect(std::string(error.what()).find(fragment) != std::string::npos,
               "the load error \"" + std::string(error.what()) + "\" does not mention \"" +
                   fragment + "\"");
        return;
    }
    expect(false, "a tokenizer.json with " + fragment + " loaded");
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

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " MERGE_TOKENIZER_JSON MERGE_CASES_JSON BOUNDARY_TOKENIZER_JSON\n";
        return 2;
    }
    try {
        const auto merge_tokenizer = json::parse(read_file(argv[1]));
        check_merge_order(merge_tokenizer, json::parse(read_file(argv[2])));
        check_refused_definitions(merge_tokenizer);
        check_prompt_frame(json::parse(read_file(argv[3])));
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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Native tokenizer checks against fixtures recorded from the Tokenizers 0.22.2
// oracle: Split-regex spans on the checkpoint's own pre-tokenizer, and token IDs
// of a small synthetic tokenizer.json with the same contract (special-token
// split, ignore_merges, merge ranks). Contract violations must fail to load.

#include "families/ministral3/runtime/tokenizer.h"

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using trtmc::ministral3::TekkenTokenizer;

int failures = 0;

void check(bool condition, const std::string& name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

nlohmann::json fixture(const char* name) {
    const auto path =
        std::filesystem::path(__FILE__).parent_path().parent_path() / "fixtures" / name;
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("missing fixture " + path.string());
    return nlohmann::json::parse(input);
}

std::shared_ptr<const TekkenTokenizer> load(const nlohmann::json& definition) {
    const auto data = definition.dump();
    return TekkenTokenizer::load(data.data(), data.size());
}

void check_pre_tokenizer() {
    for (const auto& sample : fixture("tekken_pre_tokenizer.json").at("cases")) {
        const auto text = sample.at("input").get<std::string>();
        std::vector<std::vector<std::size_t>> spans;
        for (const auto& [begin, end] : trtmc::ministral3::tekken_pre_tokenize(text))
            spans.push_back({begin, end});
        check(spans == sample.at("spans").get<std::vector<std::vector<std::size_t>>>(),
              "pre-tokenizer spans of " + nlohmann::json(text).dump());
    }
}

void check_token_ids(const nlohmann::json& data) {
    const auto tokenizer = load(data.at("tokenizer"));
    for (const auto& sample : data.at("cases")) {
        const auto text = sample.at("input").get<std::string>();
        const auto tokens = tokenizer->encode(text);
        std::vector<std::int32_t> ids;
        std::string covered;
        std::uint64_t position = 0;
        bool contiguous = true;
        for (const auto& token : tokens) {
            ids.push_back(token.id);
            contiguous = contiguous && token.byte_begin == position && token.byte_end > position;
            covered += text.substr(token.byte_begin, token.byte_end - token.byte_begin);
            position = token.byte_end;
        }
        const auto name = nlohmann::json(text).dump();
        check(ids == sample.at("ids").get<std::vector<std::int32_t>>(), "token ids of " + name);
        check(contiguous && covered == text, "byte offsets of " + name);
    }
}

void rejects(const nlohmann::json& base, const std::function<void(nlohmann::json&)>& edit,
             const std::string& name) {
    auto definition = base;
    edit(definition);
    try {
        load(definition);
    } catch (const std::exception&) {
        return;
    }
    check(false, "contract violation was accepted: " + name);
}

void check_contract(const nlohmann::json& base) {
    rejects(base, [](auto& j) { j["normalizer"] = {{"type", "NFC"}}; }, "normalizer");
    rejects(
        base, [](auto& j) { j["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = "\\s+"; },
        "regex");
    rejects(
        base, [](auto& j) { j["pre_tokenizer"]["pretokenizers"][1]["use_regex"] = true; },
        "byte-level regex");
    rejects(base, [](auto& j) { j["model"]["ignore_merges"] = false; }, "ignore_merges");
    rejects(base, [](auto& j) { j["model"]["byte_fallback"] = true; }, "byte_fallback");
    rejects(base, [](auto& j) { j["model"]["unk_token"] = "<unk>"; }, "unk_token");
    rejects(
        base,
        [](auto& j) {
            j["post_processor"]["single"].insert(
                j["post_processor"]["single"].begin(),
                nlohmann::json{{"SpecialToken", {{"id", "<s>"}, {"type_id", 0}}}});
        },
        "template tokens");
    rejects(base, [](auto& j) { j["added_tokens"][0]["lstrip"] = true; }, "lstrip");
    rejects(base, [](auto& j) { j["added_tokens"][1]["content"] = "<other>"; }, "added content");
    rejects(
        base, [](auto& j) { j["model"]["merges"].push_back(j["model"]["merges"][0]); },
        "duplicate merge");
    rejects(base, [](auto& j) { j["model"]["merges"].push_back({"x", "q"}); }, "merge result");
    rejects(base, [](auto& j) { j["truncation"] = {{"max_length", 8}}; }, "truncation");
    rejects(
        base, [](auto& j) { j["model"]["vocab"]["he"] = j["model"]["vocab"]["\xC4\xA0t"]; },
        "duplicate vocab id");
}

void check_max_token_bytes(const nlohmann::json& data) {
    const auto tokenizer = load(data.at("tokenizer"));
    // The fixture's longest vocabulary entry and longest added token span 7 bytes.
    check(tokenizer->max_token_bytes() == 7, "max_token_bytes of the fixture");
    for (const auto& sample : data.at("cases"))
        for (const auto& token : tokenizer->encode(sample.at("input").get<std::string>()))
            check(token.byte_end - token.byte_begin <= tokenizer->max_token_bytes(),
                  "every token spans at most max_token_bytes");
}

void check_invalid_utf8(const nlohmann::json& base) {
    const auto tokenizer = load(base);
    for (const std::string& bad :
         {std::string("\xC3"), std::string("a\x80"), std::string("\xED\xA0\x80"),
          std::string("\xF4\x90\x80\x80"), std::string("\xC0\xAF")}) {
        try {
            tokenizer->encode(bad);
            check(false, "invalid UTF-8 was accepted");
        } catch (const std::invalid_argument&) {
        }
    }
}

} // namespace

int main() {
    try {
        check_pre_tokenizer();
        const auto data = fixture("tekken_tokenizer.json");
        check_token_ids(data);
        check_contract(data.at("tokenizer"));
        check_max_token_bytes(data);
        check_invalid_utf8(data.at("tokenizer"));
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " ministral3 tokenizer checks failed\n";
        return 1;
    }
    std::cout << "ministral3 tokenizer checks passed\n";
    return 0;
}

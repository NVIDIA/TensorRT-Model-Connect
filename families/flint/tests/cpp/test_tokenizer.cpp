/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/flint/runtime/tokenizer.h"

#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

using nlohmann::json;

namespace {
std::unique_ptr<trtmc::ITokenizer> load(const json& config) {
    const auto document = config.dump();
    return trtmc::CreateBpeTokenizer(document.data(), document.size(), false);
}

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void rejects(const json& config) {
    try {
        (void)load(config);
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("unsupported tokenizer configuration was accepted");
}

void test_tokenizer() {
    // A minimal vocabulary makes merge ranks and Unicode byte IDs explicit.
    // The pre-tokenizer and decoder describe the supported Flint pipeline.
    const auto config = json::parse(
        R"flint({"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"bc":4,"abc":5,"1":6,"2":7,"3":8,"4":9,"12":10,"123":11,"1234":12,"Ġ":13,"Ġa":14,"Ã":15,"©":16,"Ã©":17,"ð":18,"Ł":19,"ĺ":20,"Ģ":21,"Ċ":22,"ĉ":23,"<x>":24,"<x>suffix":25,"aa":26,"aaa":27},"merges":[["b","c"],["a","b"],["a","bc"],["ab","c"],["a","a"],["aa","a"],["1","2"],["12","3"],["123","4"],["Ġ","a"],["Ã","©"]]},"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"\\p{N}{1,3}"},"behavior":"Isolated","invert":false},{"type":"Split","pattern":{"Regex":"[一-龥぀-ゟ゠-ヿ]+"},"behavior":"Isolated","invert":false},{"type":"Split","pattern":{"Regex":"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},"normalizer":null,"post_processor":null,"added_tokens":[{"id":24,"content":"<x>","special":true},{"id":25,"content":"<x>suffix","special":true}]})flint");
    const auto tokenizer = load(config);
    const std::vector<std::pair<std::string, std::vector<std::int32_t>>> cases = {
        {"", {}},
        {"abc", {5}},
        {"abab", {3, 3}},
        {"aaa", {27}},
        {"aaaaaa", {26, 26, 26}},
        {"1234", {11, 9}},
        {" a", {14}},
        {"a\n\tb", {0, 22, 23, 1}},
        {"é", {17}},
        {"😀", {18, 19, 20, 21}},
    };
    for (const auto& [text, ids] : cases) {
        require(tokenizer->encode(text) == ids, "BPE token IDs changed");
        require(tokenizer->decode(ids) == text, "byte-level decoding changed");
    }
    require(tokenizer->encode("a<x>suffixb<x>") == std::vector<std::int32_t>({0, 25, 1, 24}),
            "added tokens must use the longest match");
    require(tokenizer->decode({0, 25, 1, 24}) == "ab", "special tokens must be filtered");
    require(tokenizer->decode({15}) == "�", "incomplete UTF-8 must decode with replacement");
    require(tokenizer->id_for_token("abc") == 5, "vocabulary lookup failed");
    require(tokenizer->id_for_token("missing") == -1, "unknown token lookup changed");
    require(tokenizer->token_for_id(14) == "Ġa", "reverse vocabulary lookup failed");
    require(tokenizer->token_for_id(-1).empty(), "negative token ID must be rejected");
    require(tokenizer->token_for_id(999).empty(), "out-of-range token ID must be rejected");
    for (const auto* field : {"normalizer", "post_processor"}) {
        auto unsupported = config;
        unsupported[field] = {{"type", "Lowercase"}};
        rejects(unsupported);
    }
    for (const auto* flag : {"byte_fallback", "ignore_merges", "fuse_unk"}) {
        auto unsupported = config;
        unsupported["model"][flag] = true;
        rejects(unsupported);
    }
    for (const auto* flag : {"add_prefix_space", "use_regex"}) {
        auto unsupported = config;
        unsupported["pre_tokenizer"]["pretokenizers"][3][flag] = true;
        rejects(unsupported);
    }
    auto unsupported = config;
    unsupported["model"]["dropout"] = 0.1;
    rejects(unsupported);
}
} // namespace

int main() {
    try {
        test_tokenizer();
        std::cout << "Flint tokenizer tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

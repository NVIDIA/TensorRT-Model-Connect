/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/blip/runtime/tokenizer.h"

#include <algorithm>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {

nlohmann::json tokenizer_config() {
    const std::vector<std::string> vocabulary{
        "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[PAD]", "pho", "ᄀ",      "##ᅡ", "ａ", "οσ",
        "i",     "a",     "𐐨",     "[M",     "ab",    "bc",  "i\u0307", "I",   "α"};
    nlohmann::json vocab = nlohmann::json::object();
    for (std::size_t i = 0; i < vocabulary.size(); ++i)
        vocab[vocabulary[i]] = i;
    nlohmann::json added = nlohmann::json::array();
    for (int id : {0, 1, 2, 3, 4, 13, 14, 15})
        added.push_back({{"id", id},
                         {"content", vocabulary[id]},
                         {"special", true},
                         {"normalized", false},
                         {"single_word", false}});
    return {{"model", {{"type", "WordPiece"}, {"unk_token", "[UNK]"}, {"vocab", vocab}}},
            {"normalizer", {{"type", "BertNormalizer"}, {"lowercase", true}}},
            {"added_tokens", added},
            {"post_processor",
             {{"type", "BertProcessing"}, {"cls", {"[CLS]", 1}}, {"sep", {"[SEP]", 2}}}}};
}

auto create(const nlohmann::json& config) {
    const auto data = config.dump();
    return trtmc::blip::CreateWordPieceTokenizer(data.data(), data.size());
}

void expect(const trtmc::blip::ITokenizer& tokenizer, const std::string& text,
            const std::vector<std::int32_t>& expected) {
    if (tokenizer.encode(text) != expected)
        throw std::runtime_error("BLIP tokenizer differs from the upstream fixture: " + text);
}

void expect_decoded(const trtmc::blip::ITokenizer& tokenizer, const std::vector<std::int32_t>& ids,
                    const std::string& expected) {
    if (tokenizer.decode(ids) != expected)
        throw std::runtime_error("BLIP WordPiece decoding differs from the upstream fixture");
}

} // namespace

int main() {
    try {
        auto config = tokenizer_config();
        auto tokenizer = create(config);
        // Expected IDs follow tokenizers' BertNormalizer and WordPiece behavior.
        expect(*tokenizer, "phở", {1, 5, 2});
        expect(*tokenizer, "가", {1, 6, 7, 2});
        expect(*tokenizer, "Ａ", {1, 8, 2});
        expect(*tokenizer, "ΟΣ", {1, 9, 2});
        expect(*tokenizer, "İ", {1, 10, 2});
        expect(*tokenizer, "a\u05b0", {1, 11, 2});
        expect(*tokenizer, "𐐀", {1, 12, 2});
        expect(*tokenizer, "ᾈ", {1, 18, 2});
        // Preserve earliest matches, longest ties, and overlapping-token invalidation.
        expect(*tokenizer, "[MASK][M[MASK] a", {1, 3, 13, 3, 11, 2});
        expect(*tokenizer, "abc[MASK]", {1, 14, 0, 3, 2});

        // A continuation can start with a subword after special tokens are skipped.
        expect_decoded(*tokenizer, {7, 2}, "##ᅡ");
        expect_decoded(*tokenizer, {1, 4, 7, 2}, "##ᅡ");
        expect_decoded(*tokenizer, {7, 7, 2}, "##ᅡᅡ");
        expect_decoded(*tokenizer, {6, 7, 2}, "가");
        expect_decoded(*tokenizer, {11, 5, 2}, "a pho");
        expect_decoded(*tokenizer, {1, 2}, "");

        config["normalizer"]["strip_accents"] = false;
        expect(*create(config), "İ", {1, 16, 2});
        config["normalizer"]["strip_accents"] = true;
        config["normalizer"]["lowercase"] = false;
        expect(*create(config), "İ", {1, 17, 2});

        constexpr std::size_t repetitions = 170000;
        std::string long_prefix;
        long_prefix.reserve(repetitions * 6);
        for (std::size_t i = 0; i < repetitions; ++i)
            long_prefix += "[MASK]";
        const auto ids = tokenizer->encode(long_prefix);
        if (ids.size() != repetitions + 2 || ids.front() != 1 || ids.back() != 2 ||
            !std::all_of(ids.begin() + 1, ids.end() - 1, [](auto id) { return id == 3; }))
            throw std::runtime_error("Repeated special tokens changed encoding");
        std::cout << "BLIP Unicode and special-token regressions passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

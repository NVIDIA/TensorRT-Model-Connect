/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/roberta/runtime/tokenizer.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    std::ifstream input(argv[1]);
    const auto data = nlohmann::json::parse(input);
    const auto& fixture = data.at("tokenizer");
    int failures = 0;
    const auto create = [](const nlohmann::json& config) {
        const auto data = config.dump();
        return trtmc::CreateUnigramTokenizer(data.data(), data.size(), false);
    };
    const auto check = [&](const trtmc::ITokenizer& tokenizer, const std::string& text,
                           const std::vector<int32_t>& expected) {
        if (tokenizer.encode(text) != expected) {
            std::cerr << "Unexpected token IDs for: " << text << '\n';
            ++failures;
        }
    };
    auto tokenizer = create(fixture);
    check(*tokenizer, "AB", {3, 4, 5});
    check(*tokenizer, u8"ＡＢ", {3, 4, 5});
    check(*tokenizer, u8"① ﬁ", {3, 6, 3, 7});
    check(*tokenizer, u8"e\u0301", {3, 8});
    check(*tokenizer, u8"x\u0301", {3, 9});
    check(*tokenizer, u8"x\u0301\u0327", {3, 10});
    check(*tokenizer, u8"A\u200bB", {3, 4, 3, 5});
    check(*tokenizer, u8"A\u00adB", {3, 4, 5});
    check(*tokenizer, u8"éz", {3, 8, 11});
    check(*tokenizer, std::string("A\0B", 3), {3, 4, 0, 5});
    check(*tokenizer, "", {});

    auto sequence = fixture;
    sequence["normalizer"] = {{"type", "Sequence"}, {"normalizers", {fixture["normalizer"]}}};
    check(*create(sequence), u8"ＡＢ", {3, 4, 5});
    auto identity = fixture;
    identity["normalizer"] = nullptr;
    check(*create(identity), u8"ＡＢ", {3, 0, 0});
    check(*create(identity), "AB", {3, 4, 5});

    const std::vector<std::string> corrupt_maps = {
        "", "!!!!", "AAAAA", "AAAAAA==", "AAQAAA==", "AAQAAAE=", "AAQAAAA===",
    };
    for (const auto& map : corrupt_maps) {
        auto invalid = fixture;
        invalid["normalizer"]["precompiled_charsmap"] = map;
        try {
            create(invalid);
            std::cerr << "Accepted invalid charsmap: " << map << '\n';
            ++failures;
        } catch (const std::runtime_error&) {
        }
    }
    for (const auto& entry : data.at("invalid_charsmaps").items()) {
        auto invalid = fixture;
        invalid["normalizer"]["precompiled_charsmap"] = entry.value();
        try {
            create(invalid);
            std::cerr << "Accepted corrupt charsmap: " << entry.key() << '\n';
            ++failures;
        } catch (const std::runtime_error&) {
        }
    }
    auto missing = fixture;
    missing["normalizer"].erase("precompiled_charsmap");
    try {
        create(missing);
        std::cerr << "Accepted missing charsmap\n";
        ++failures;
    } catch (const std::runtime_error&) {
    }
    return failures == 0 ? 0 : 1;
}

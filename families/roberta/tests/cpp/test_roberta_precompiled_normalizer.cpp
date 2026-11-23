/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/roberta/runtime/tokenizer.h"
#include "families/roberta/runtime/unicode_grapheme.h"

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
    check(*tokenizer, u8"x\u0301\u0327", {3, 9});
    check(*tokenizer, u8"e\u0301\u0327", {3, 8});
    check(*tokenizer, u8"A\u200bB", {3, 4, 3, 5});
    check(*tokenizer, u8"A\u00adB", {3, 4, 5});
    check(*tokenizer, u8"éz", {3, 8, 11});
    check(*tokenizer, std::string("A\0B", 3), {3, 4, 0, 5});
    check(*tokenizer, "", {});

    auto map_b = fixture["normalizer"];
    map_b["precompiled_charsmap"] = data.at("normalizer_maps").at("B_to_A");
    auto maps = fixture;
    maps["normalizer"] = {{"type", "Sequence"}, {"normalizers", {fixture["normalizer"], map_b}}};
    check(*create(maps), u8"ＡＢ", {3, 4, 4});
    maps["normalizer"]["normalizers"] = {map_b, fixture["normalizer"]};
    check(*create(maps), u8"ＡＢ", {3, 4, 5});
    auto map_a = fixture["normalizer"];
    map_a["precompiled_charsmap"] = data.at("normalizer_maps").at("a_to_A");
    maps["normalizer"]["normalizers"] = {{{"type", "Lowercase"}}, map_a};
    check(*create(maps), "A", {3, 4});
    maps["normalizer"]["normalizers"] = {map_a, {{"type", "Lowercase"}}};
    check(*create(maps), "A", {3, 0});
    maps["normalizer"]["normalizers"] = {{{"type", "Lowercase"}},
                                         {{"type", "Sequence"}, {"normalizers", {map_a}}}};
    check(*create(maps), "A", {3, 4});
    auto whitespace = fixture;
    whitespace["pre_tokenizer"] = {
        {"type", "Sequence"},
        {"pretokenizers", {{{"type", "WhitespaceSplit"}}, fixture["pre_tokenizer"]}}};
    check(*create(whitespace), u8"A\u1680B", {3, 4, 3, 5});
    check(*create(whitespace), u8"A\u0085B", {3, 4, 3, 5});
    check(*create(whitespace), "A\vB\fA", {3, 4, 3, 5, 3, 4});
    auto metaspace_only = fixture;
    metaspace_only["normalizer"] = nullptr;
    check(*create(metaspace_only), u8"A\u1680B", {3, 4, 0, 5});
    check(*create(metaspace_only), u8"A\u0085B", {3, 4, 0, 5});
    for (const auto* control : {"A\tB", "A\nB", "A\rB"}) {
        check(*create(metaspace_only), control, {3, 4, 0, 5});
        check(*create(whitespace), control, {3, 4, 3, 5});
    }
    auto suffix = fixture;
    suffix["normalizer"]["precompiled_charsmap"] = data.at("valid_charsmaps").at("ascii_suffix");
    check(*create(suffix), u8"ﬁ", {3, 0});
    suffix["normalizer"]["precompiled_charsmap"] = data.at("valid_charsmaps").at("empty_suffix");
    check(*create(suffix), u8"ﬁ", {});
    suffix["normalizer"]["precompiled_charsmap"] = data.at("valid_charsmaps").at("unicode_pool");
    check(*create(suffix), "AB", {3, 4, 5});

    auto sequence = fixture;
    sequence["normalizer"] = {{"type", "Sequence"}, {"normalizers", {fixture["normalizer"]}}};
    check(*create(sequence), u8"ＡＢ", {3, 4, 5});
    auto identity = fixture;
    identity["normalizer"] = nullptr;
    check(*create(identity), u8"ＡＢ", {3, 0});
    check(*create(identity), "YY", {3, 0});
    check(*create(identity), u8"🫨🫩", {3, 0});
    check(*create(identity), u8"AＡＢB", {3, 4, 0, 5});
    check(*create(identity), "AA", {3, 4, 4});
    check(*create(identity), u8"ＡＢ ＡＢ", {3, 0, 3, 0});
    check(*create(identity), u8"ＡＡAＢＢ", {3, 0, 4, 0});
    const auto framed_data = identity.dump();
    auto framed = trtmc::CreateUnigramTokenizer(framed_data.data(), framed_data.size(), true);
    check(*framed, u8"ＡＢ", {1, 3, 0, 2});
    check(*framed, "", {1, 2});
    check(*create(identity), "AB", {3, 4, 5});
    auto split_identity = identity;
    split_identity["pre_tokenizer"] = whitespace["pre_tokenizer"];
    check(*create(split_identity), u8"A\u0085B", {3, 4, 3, 5});
    check(*create(split_identity), "A\vB\fA", {3, 4, 3, 5, 3, 4});

    auto added = fixture;
    added["pre_tokenizer"] = {
        {"type", "Sequence"},
        {"pretokenizers", {{{"type", "WhitespaceSplit"}}, fixture["pre_tokenizer"]}}};
    added["added_tokens"] = {
        {{"id", 0}, {"content", "<unk>"}, {"normalized", false}, {"special", true}},
        {{"id", 1}, {"content", "<s>"}, {"normalized", false}, {"special", true}},
        {{"id", 2}, {"content", "</s>"}, {"normalized", false}, {"special", true}},
        {{"id", 12},
         {"content", "<mask>"},
         {"normalized", false},
         {"lstrip", true},
         {"rstrip", true},
         {"special", true}},
        {{"id", 13}, {"content", u8"[Ａ]"}, {"normalized", false}},
        {{"id", 14}, {"content", u8"[Ｂ]"}, {"normalized", true}},
        {{"id", 15}, {"content", "<mask>x"}, {"normalized", false}},
    };
    auto with_added = create(added);
    check(*with_added, "<mask>", {12});
    check(*with_added, "A<mask>B", {3, 4, 12, 3, 5});
    check(*with_added, "<s>A</s>", {1, 3, 4, 2});
    check(*with_added, "<unk><unk>", {0, 0});
    check(*with_added, "<mask>x", {15});
    check(*with_added, u8"[Ａ]", {13});
    check(*with_added, "[A]", {3, 0, 4, 0});
    check(*with_added, u8"[Ｂ]", {14});
    check(*with_added, "[B]", {14});
    check(*with_added, u8"＜mask＞", {3, 0});
    auto added_identity = added;
    added_identity["normalizer"] = nullptr;
    auto stripped = create(added_identity);
    check(*stripped, u8"A<mask>▁B", {3, 4, 12, 3, 5});
    check(*stripped, u8"▁AB", {3, 4, 5});
    auto meta_identity = added_identity;
    meta_identity["pre_tokenizer"] = fixture["pre_tokenizer"];
    auto meta = create(meta_identity);
    check(*meta, " ", {3});
    check(*meta, "A ", {3, 4, 3});
    check(*meta, " A", {3, 4});
    check(*meta, "A  B", {3, 4, 3, 3, 5});
    check(*meta, u8"▁ ", {3, 3});
    check(*meta, u8"<mask>▁ ", {12, 3, 3});
    auto no_crossing = meta_identity;
    no_crossing["model"]["vocab"].push_back({u8"▁A▁B", 100.0});
    check(*create(no_crossing), u8"A▁B", {3, 4, 3, 5});
    check(*stripped, u8"A\u0085<mask>\u00a0B", {3, 4, 12, 3, 5});
    check(*stripped, u8"<mask>\u3000<mask>", {12, 12});
    check(*stripped, "A <mask> B", {3, 4, 12, 3, 5});
    auto whitespace_token = added_identity;
    whitespace_token["added_tokens"].push_back(
        {{"id", 16}, {"content", " "}, {"normalized", false}});
    auto with_whitespace = create(whitespace_token);
    check(*with_whitespace, "<mask> ", {12, 16});
    check(*with_whitespace, "<mask> <mask>", {12, 16, 12});
    whitespace_token["added_tokens"].back()["lstrip"] = true;
    auto stripped_whitespace = create(whitespace_token);
    check(*stripped_whitespace, "<mask> ", {12});
    check(*stripped_whitespace, "<mask> B", {12, 3, 5});
    const auto added_data = added.dump();
    auto added_frame = trtmc::CreateUnigramTokenizer(added_data.data(), added_data.size(), true);
    check(*added_frame, "<s><unk><unk></s>", {1, 1, 0, 0, 2, 2});
    const auto identity_data = added_identity.dump();
    auto identity_frame =
        trtmc::CreateUnigramTokenizer(identity_data.data(), identity_data.size(), true);
    check(*identity_frame, u8"A<mask>▁B", {1, 3, 4, 12, 3, 5, 2});
    auto reversed = added_identity;
    reversed["pre_tokenizer"] = {
        {"type", "Sequence"},
        {"pretokenizers", {fixture["pre_tokenizer"], {{"type", "WhitespaceSplit"}}}}};
    try {
        create(reversed);
        std::cerr << "Accepted unsupported Metaspace then WhitespaceSplit\n";
        ++failures;
    } catch (const std::runtime_error&) {
    }
    for (const auto& stages :
         nlohmann::json::array({{{{"type", "Metaspace"}}, {{"type", "Metaspace"}}},
                                {{{"type", "WhitespaceSplit"}}},
                                {{{"type", "Digits"}}, fixture["pre_tokenizer"]}})) {
        auto unsupported = added_identity;
        unsupported["pre_tokenizer"] = {{"type", "Sequence"}, {"pretokenizers", stages}};
        try {
            create(unsupported);
            std::cerr << "Accepted unsupported pre-tokenizer composition\n";
            ++failures;
        } catch (const std::runtime_error&) {
        }
    }
    auto single_word = added;
    single_word["added_tokens"][3]["single_word"] = true;
    try {
        create(single_word);
        std::cerr << "Accepted unsupported single_word added token\n";
        ++failures;
    } catch (const std::runtime_error&) {
    }
    auto empty_pattern = added;
    empty_pattern["added_tokens"][5]["content"] = u8"\u00ad";
    try {
        create(empty_pattern);
        std::cerr << "Accepted added token that normalizes to empty\n";
        ++failures;
    } catch (const std::runtime_error&) {
    }

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
    for (const std::string& grapheme :
         {std::string("\r\n"), std::string(u8"x\u0301\u0327"), std::string(u8"\u1100\u1161\u11a8"),
          std::string(u8"\U0001f1fa\U0001f1f8"), std::string(u8"\U0001f469\u200d\U0001f4bb")}) {
        if (trtmc::roberta_unicode::grapheme_end(grapheme + "A", 0) != grapheme.size()) {
            std::cerr << "Incorrect grapheme boundary\n";
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}

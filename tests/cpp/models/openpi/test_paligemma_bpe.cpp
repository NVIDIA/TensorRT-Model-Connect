/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/openpi/paligemma_bpe.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

template <typename Function>
void check_throws(Function&& function, const char* name) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, name);
}

std::string byte_piece(int value) {
    char piece[7]{};
    std::snprintf(piece, sizeof(piece), "<0x%02X>", value);
    return piece;
}

struct SyntheticAsset {
    trtmc::openpi::PaligemmaBpeAsset asset;
    int32_t space{-1};
    int32_t a{-1};
    int32_t b{-1};
    int32_t c{-1};
    int32_t aa{-1};
    int32_t hello{-1};
    int32_t space_hello{-1};
    int32_t space_world{-1};
    int32_t abc{-1};
    int32_t robot{-1};
    int32_t capital_a{-1};
};

SyntheticAsset make_synthetic_asset() {
    using trtmc::openpi::BpePiece;
    using trtmc::openpi::SentencePieceType;

    SyntheticAsset result;
    auto& asset = result.asset;
    asset.unknown_id = 1;
    asset.bos_id = 2;
    asset.eos_id = 3;
    asset.pad_id = 0;
    asset.pieces = {
        BpePiece{"<pad>", 0.0F, SentencePieceType::kControl},
        BpePiece{"<unk>", 0.0F, SentencePieceType::kUnknown},
        BpePiece{"<s>", 0.0F, SentencePieceType::kControl},
        BpePiece{"</s>", 0.0F, SentencePieceType::kControl},
    };
    for (int value = 0; value < 256; ++value) {
        asset.pieces.push_back(BpePiece{byte_piece(value), 0.0F, SentencePieceType::kByte});
    }
    auto add = [&](std::string text, float score,
                   SentencePieceType type = SentencePieceType::kNormal) {
        const auto id = static_cast<int32_t>(asset.pieces.size());
        asset.pieces.push_back(BpePiece{std::move(text), score, type});
        return id;
    };

    result.space = add("\xE2\x96\x81", 0.0F);
    result.a = add("a", 0.0F);
    result.b = add("b", 0.0F);
    result.c = add("c", 0.0F);
    (void)add("h", 0.0F);
    (void)add("e", 0.0F);
    (void)add("l", 0.0F);
    (void)add("o", 0.0F);
    (void)add("w", 0.0F);
    (void)add("r", 0.0F);
    (void)add("d", 0.0F);
    (void)add("he", 1.0F);
    (void)add("ll", 3.0F);
    (void)add("llo", 4.0F);
    result.hello = add("hello", 10.0F);
    result.space_hello = add("\xE2\x96\x81hello", 11.0F);
    (void)add("wo", 1.0F);
    (void)add("wor", 2.0F);
    (void)add("worl", 3.0F);
    (void)add("world", 4.0F);
    result.space_world = add("\xE2\x96\x81world", 5.0F);
    result.aa = add("aa", 1.0F);
    (void)add("ab", 5.0F, SentencePieceType::kUnused);
    result.abc = add("abc", 6.0F);
    result.robot = add("<robot>", 100.0F, SentencePieceType::kUserDefined);
    result.capital_a = add("A", 0.0F);
    asset.normalization_rules = {
        {"\xEF\xBC\xA1", "A"}, // full-width A
        {"\t", " "},
        {"\n", " "},
    };
    return result;
}

void test_flat_asset_round_trip_and_validation() {
    const auto synthetic = make_synthetic_asset();
    const auto bytes = trtmc::openpi::serialize_paligemma_bpe_asset(synthetic.asset);
    const auto parsed = trtmc::openpi::parse_paligemma_bpe_asset(
        std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
    check(parsed.version == 1, "flat BPE parser preserves version");
    check(parsed.pieces.size() == synthetic.asset.pieces.size(),
          "flat BPE parser preserves every piece");
    check(parsed.normalization_rules.size() == synthetic.asset.normalization_rules.size(),
          "flat BPE parser preserves normalization rules");
    check(parsed.byte_fallback && parsed.add_dummy_prefix && parsed.escape_whitespaces,
          "flat BPE parser preserves SentencePiece flags");

    auto corrupted = bytes;
    corrupted[0] = 'X';
    check_throws(
        [&] {
            (void)trtmc::openpi::parse_paligemma_bpe_asset(std::string_view(
                reinterpret_cast<const char*>(corrupted.data()), corrupted.size()));
        },
        "flat BPE parser rejects bad magic");

    auto incomplete_bytes = synthetic.asset;
    incomplete_bytes.pieces.erase(incomplete_bytes.pieces.begin() + 4);
    check_throws([&] { (void)trtmc::openpi::serialize_paligemma_bpe_asset(incomplete_bytes); },
                 "flat BPE validation rejects incomplete byte fallback table");
}

void test_sentencepiece_normalization_and_score_ordered_bpe() {
    const auto synthetic = make_synthetic_asset();
    trtmc::openpi::PaligemmaBpeTokenizer tokenizer(synthetic.asset);

    check(tokenizer.normalize("  hello   world ") == "\xE2\x96\x81hello\xE2\x96\x81world",
          "SentencePiece normalization collapses spaces and adds dummy prefix");
    check(tokenizer.normalize("\t\xEF\xBC\xA1 ") == "\xE2\x96\x81"
                                                    "A",
          "SentencePiece normalization uses flattened longest-prefix rules");

    const auto hello_world = tokenizer.encode("hello world", true);
    check(hello_world == std::vector<int32_t>({synthetic.asset.bos_id, synthetic.space_hello,
                                               synthetic.space_world}),
          "SentencePiece BPE performs highest-score merges");

    const auto tied = tokenizer.encode("aaa", false);
    check(tied == std::vector<int32_t>({synthetic.space, synthetic.aa, synthetic.a}),
          "SentencePiece BPE resolves equal-score merges from the left");
}

void test_unused_resegmentation_user_symbols_and_byte_fallback() {
    const auto synthetic = make_synthetic_asset();
    trtmc::openpi::PaligemmaBpeTokenizer tokenizer(synthetic.asset);

    const auto unused_only = tokenizer.encode("ab", false);
    check(unused_only == std::vector<int32_t>({synthetic.space, synthetic.a, synthetic.b}),
          "unused merged piece is recursively resegmented");
    const auto useful_unused = tokenizer.encode("abc", false);
    check(useful_unused == std::vector<int32_t>({synthetic.space, synthetic.abc}),
          "unused merge can participate in a later normal merge");

    const auto user_defined = tokenizer.encode("<robot>", false);
    check(user_defined == std::vector<int32_t>({synthetic.space, synthetic.robot}),
          "user-defined symbol is frozen as one piece");

    const auto unknown = tokenizer.encode("\xF0\x9F\xA4\x96", false); // robot emoji
    check(unknown ==
              std::vector<int32_t>({synthetic.space, 4 + 0xF0, 4 + 0x9F, 4 + 0xA4, 4 + 0x96}),
          "unknown Unicode is decomposed into SentencePiece byte fallback ids");
}

void test_openpi_fixed_length_prompt_contract() {
    const auto synthetic = make_synthetic_asset();
    trtmc::openpi::PaligemmaBpeTokenizer tokenizer(synthetic.asset);

    const auto pi0 = tokenizer.tokenize_pi0(" hello_", 5);
    check(pi0.token_ids[0] == synthetic.asset.bos_id && pi0.token_ids[1] == synthetic.space_hello,
          "pi0 tokenizer cleans prompt and inserts BOS");
    check(pi0.token_ids[2] == 0 && pi0.token_ids[4] == 0,
          "OpenPI prompt padding uses integer zero");
    check(pi0.token_mask == std::vector<uint8_t>({1, 1, 0, 0, 0}),
          "OpenPI prompt mask tracks only real tokens");
    check(!pi0.truncated, "short OpenPI prompt is not marked truncated");

    const auto truncated = tokenizer.tokenize_pi0("hello world", 2);
    check(truncated.truncated && truncated.token_ids.size() == 2 &&
              truncated.token_mask == std::vector<uint8_t>({1, 1}),
          "OpenPI prompt truncation matches max length");

    const auto pi05 = tokenizer.tokenize_pi05("A", {-1.0F, 0.0F, 1.0F}, 200);
    check(pi05.token_ids.size() == 200 && pi05.token_mask.size() == 200,
          "pi05 tokenization emits fixed 200-token contract");
    check(pi05.token_ids.front() == synthetic.asset.bos_id,
          "pi05 tokenization inserts BOS before formatted task/state text");
    check_throws([&] { (void)tokenizer.tokenize_pi0("hello", 0); },
                 "OpenPI tokenizer rejects zero maximum length");
}

void test_official_paligemma_asset(std::string_view path) {
    std::ifstream input(std::string(path), std::ios::binary);
    if (!input) {
        std::cerr << "FAIL: cannot open official PaliGemma flat asset: " << path << '\n';
        ++g_failures;
        return;
    }
    const std::string bytes((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    auto asset = trtmc::openpi::parse_paligemma_bpe_asset(bytes);
    check(asset.pieces.size() == 257152, "official PaliGemma vocabulary size");
    check(asset.unknown_id == 3 && asset.bos_id == 2 && asset.eos_id == 1 && asset.pad_id == 0,
          "official PaliGemma special token ids");
    check(!asset.add_dummy_prefix && !asset.remove_extra_whitespaces && asset.escape_whitespaces &&
              asset.byte_fallback,
          "official PaliGemma identity-normalizer flags");

    trtmc::openpi::PaligemmaBpeTokenizer tokenizer(std::move(asset));
    check(tokenizer.encode("hello world", true) == std::vector<int32_t>({2, 17534, 2134}),
          "native BPE matches official SentencePiece hello-world ids");
    check(tokenizer.encode("\xF0\x9F\xA4\x96", true) == std::vector<int32_t>({2, 243349}),
          "native BPE matches official SentencePiece Unicode ids");

    const auto pi0 = tokenizer.tokenize_pi0(" hello_", 48);
    check(std::vector<int32_t>(pi0.token_ids.begin(), pi0.token_ids.begin() + 4) ==
              std::vector<int32_t>({2, 17534, 235248, 108}),
          "native pi0 prompt path matches official SentencePiece ids");
    check(std::vector<uint8_t>(pi0.token_mask.begin(), pi0.token_mask.begin() + 5) ==
              std::vector<uint8_t>({1, 1, 1, 1, 0}),
          "native pi0 prompt padding mask matches OpenPI");

    const auto pi05 = tokenizer.tokenize_pi05("pick_up\nblock", {-1.0F, 0.0F, 1.0F}, 200);
    const std::vector<int32_t> expected_pi05{
        2,      7071,   235292, 4788,   908,    3963,   235269, 3040,
        235292, 235248, 235276, 235248, 235274, 235284, 235321, 235248,
        235284, 235308, 235308, 235289, 108,    4022,   235292, 235248,
    };
    check(std::vector<int32_t>(pi05.token_ids.begin(),
                               pi05.token_ids.begin() + expected_pi05.size()) == expected_pi05,
          "native pi05 formatted prompt matches official SentencePiece ids");
    check(pi05.token_mask[expected_pi05.size() - 1] == 1U &&
              pi05.token_mask[expected_pi05.size()] == 0U,
          "native pi05 prompt mask begins padding after official token sequence");
}

} // namespace

int main(int argc, char** argv) {
    test_flat_asset_round_trip_and_validation();
    test_sentencepiece_normalization_and_score_ordered_bpe();
    test_unused_resegmentation_user_symbols_and_byte_fallback();
    test_openpi_fixed_length_prompt_contract();
    if (argc == 2) {
        test_official_paligemma_asset(argv[1]);
    } else if (argc > 2) {
        std::cerr << "FAIL: expected zero arguments or one official flat BPE asset path\n";
        ++g_failures;
    }

    if (g_failures != 0) {
        std::cerr << g_failures << " OpenPI PaliGemma BPE test(s) failed\n";
        return 1;
    }
    return 0;
}

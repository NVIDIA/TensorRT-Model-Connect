/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU checks of the PaliGemma runtime pieces that need no engine: tokenizer
// encode/decode, prompt assembly and the Pillow-exact image resize.
//
//   test_paligemma_runtime PILLOW_GOLDEN.json

#include "families/paligemma/runtime/image_preprocess.h"
#include "families/paligemma/runtime/paligemma_bpe.h"
#include "families/paligemma/runtime/prompt.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace trtmc::paligemma;
using trtmc::internal::ImagesTextMessage;
using trtmc::internal::ImagesTextPart;
using trtmc::internal::ImageView;
using trtmc::internal::MessageRole;
using trtmc::internal::TextPartView;

int failures = 0;

void check(bool condition, const std::string& name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

template <typename Function>
void check_throws(Function&& function, const std::string& name) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    check(false, name + " must be rejected");
}

// <pad> <eos> <bos> <unk>, 256 byte pieces, then a few merges and one user-defined symbol.
PaligemmaBpeTokenizer toy_tokenizer() {
    PaligemmaBpeAsset asset;
    asset.add_dummy_prefix = false;
    asset.remove_extra_whitespaces = false;
    asset.pad_id = 0;
    asset.eos_id = 1;
    asset.bos_id = 2;
    asset.unknown_id = 3;
    asset.pieces = {{"<pad>", 0.0F, SentencePieceType::kControl},
                    {"<eos>", 0.0F, SentencePieceType::kControl},
                    {"<bos>", 0.0F, SentencePieceType::kControl},
                    {"<unk>", 0.0F, SentencePieceType::kUnknown}};
    for (int value = 0; value < 256; ++value) {
        char text[7];
        std::snprintf(text, sizeof(text), "<0x%02X>", value);
        asset.pieces.push_back({text, 0.0F, SentencePieceType::kByte});
    }
    for (const auto& [text, score] :
         std::vector<std::pair<std::string, float>>{{"a", -1.0F},
                                                    {"b", -2.0F},
                                                    {"\xE2\x96\x81", -3.0F},
                                                    {"\n", -4.0F},
                                                    {"ab", -5.0F},
                                                    {"\xE2\x96\x81"
                                                     "ab",
                                                     -6.0F}})
        asset.pieces.push_back({text, score, SentencePieceType::kNormal});
    asset.pieces.push_back({"<loc0001>", 0.0F, SentencePieceType::kUserDefined});
    return PaligemmaBpeTokenizer(parse_paligemma_bpe_asset([&] {
        const auto bytes = serialize_paligemma_bpe_asset(asset);
        return std::string(bytes.begin(), bytes.end());
    }()));
}

constexpr int32_t kA = 260, kB = 261, kSpace = 262, kNewline = 263, kAb = 264, kSpaceAb = 265,
                  kLoc = 266;

void test_tokenizer() {
    const auto tokenizer = toy_tokenizer();
    check(tokenizer.encode("ab ab\n", false) == std::vector<int32_t>{kAb, kSpaceAb, kNewline},
          "spaces become U+2581 and merge by score, without a dummy prefix");
    check(tokenizer.encode("a<loc0001>b", false) == std::vector<int32_t>{kA, kLoc, kB},
          "user-defined symbols are atomic");
    // U+00E9 is not a piece: byte fallback emits its two UTF-8 bytes.
    check(tokenizer.encode("\xC3\xA9", false) == std::vector<int32_t>{4 + 0xC3, 4 + 0xA9},
          "unknown characters fall back to bytes");
    const std::unordered_set<int32_t> special{0, 1, 2};
    check(tokenizer.decode({kAb, kSpaceAb, kNewline, 1}, special) == "ab ab\n",
          "decode restores spaces and skips special ids");
    check(tokenizer.decode({kA, 4 + 0xC3, 4 + 0xA9, kLoc}, special) == "a\xC3\xA9<loc0001>",
          "decode fuses byte runs and keeps user-defined symbols");
    check(tokenizer.decode({4 + 0xC3, kB}, special) == "\xEF\xBF\xBD"
                                                       "b",
          "an invalid byte run becomes one replacement character per byte");
    check(tokenizer.decode({9999, kA}, special) == "a", "ids outside the vocabulary have no text");
}

void test_prompt() {
    const auto tokenizer = toy_tokenizer();
    const std::uint8_t pixel[3] = {0, 0, 0};
    const ImageView image{pixel, 3, 1, 1, 3, trtmc::internal::ImageFormat::UInt8};
    std::vector<ImagesTextPart> parts{image, image, TextPartView{"ab"}, TextPartView{" ab"}};
    std::vector<ImagesTextMessage> messages{{MessageRole::User, {parts.data(), parts.size()}}};
    const auto prompt = collect_prompt({messages.data(), messages.size()}, {});
    check(prompt.images.size() == 2 && prompt.prefix == "ab ab", "images then concatenated text");

    PromptTokens tokens{500, 3, 2, {"<bos>", "<image>"}};
    const std::vector<int32_t> expected{500, 500, 500, 500, 500, 500, 2, kAb, kSpaceAb, kNewline};
    check(prompt_ids(prompt, tokenizer, tokens) == expected,
          "image tokens, BOS, prefix and the newline separator");

    PromptParts reserved{prompt.images, "say <bos>"};
    check_throws([&] { prompt_ids(reserved, tokenizer, tokens); }, "a special token in the prefix");
    check(reserved_text(tokenizer, {2, 500}, 500) == std::vector<std::string>{"<bos>", "<image>"},
          "reserved ids map to piece text and the image token");
    check_throws([&] { reserved_text(tokenizer, {9999}, 500); },
                 "a reserved id outside the vocabulary");

    std::vector<ImagesTextPart> late{TextPartView{"ab"}, image};
    std::vector<ImagesTextMessage> late_message{{MessageRole::User, {late.data(), late.size()}}};
    check_throws([&] { collect_prompt({late_message.data(), 1}, {}); }, "an image after text");
    std::vector<ImagesTextPart> text_only{TextPartView{"ab"}};
    std::vector<ImagesTextMessage> text_message{
        {MessageRole::User, {text_only.data(), text_only.size()}}};
    check_throws([&] { collect_prompt({text_message.data(), 1}, {}); }, "a prompt without images");
    std::vector<ImagesTextMessage> two{messages[0], messages[0]};
    check_throws([&] { collect_prompt({two.data(), two.size()}, {}); }, "two messages");
    std::vector<ImagesTextMessage> system{{MessageRole::System, {parts.data(), parts.size()}}};
    check_throws([&] { collect_prompt({system.data(), 1}, {}); }, "a non-user message");
}

void test_resize(const std::string& path) {
    std::ifstream file(path);
    const auto golden = nlohmann::json::parse(file);
    for (const auto& item : golden.at("cases")) {
        const int height = item.at("height"), width = item.at("width");
        std::vector<std::uint8_t> source(size_t(height) * width * 3);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                for (int c = 0; c < 3; ++c)
                    source[(size_t(y) * width + x) * 3 + c] =
                        std::uint8_t((37 * y + 11 * x + 89 * c + (x * y) % 7 * 13) % 256);
        const auto actual =
            resize_pillow_bicubic(source.data(), height, width, item.at("out_h"), item.at("out_w"));
        check(actual == item.at("pixels").get<std::vector<std::uint8_t>>(),
              "Pillow bicubic " + std::to_string(height) + "x" + std::to_string(width) + " -> " +
                  item.at("out_h").dump() + "x" + item.at("out_w").dump());
    }
    // Normalization: (v / 255 - 0.5) / 0.5 per channel, planar output.
    const std::vector<std::uint8_t> flat(2 * 2 * 3, 255);
    ImageNormalization config;
    config.size = 2;
    const auto pixels = preprocess_image(flat.data(), 2, 2, config);
    check(pixels.size() == 12 && pixels[0] == 1.0F && pixels[11] == 1.0F,
          "rescale and normalize to [-1, 1]");
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_paligemma_runtime PILLOW_GOLDEN.json\n";
        return 2;
    }
    try {
        test_tokenizer();
        test_prompt();
        test_resize(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0)
        std::cout << "paligemma runtime checks passed\n";
    return failures == 0 ? 0 : 1;
}

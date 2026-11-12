/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-only checks of the MiniCPM-OCR image processor and prompt rendering.
// Expected values were produced by the official Transformers MiniCPM-OCR
// processor (torchvision CPU resize) on the same synthetic inputs.

#include "families/minicpm_ocr/runtime/image_processor.h"
#include "families/minicpm_ocr/runtime/prompt.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace trtmc::minicpm_ocr;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << what << '\n';
    }
}

std::vector<uint8_t> pattern(int32_t height, int32_t width) {
    std::vector<uint8_t> pixels(static_cast<std::size_t>(height) * width * 3);
    for (int32_t y = 0; y < height; ++y)
        for (int32_t x = 0; x < width; ++x)
            for (int32_t c = 0; c < 3; ++c)
                pixels[(static_cast<std::size_t>(y) * width + x) * 3 + c] =
                    static_cast<uint8_t>((x * 7 + y * 13 + c * 31 + (x * y) % 17) % 256);
    return pixels;
}

uint64_t checksum(const std::vector<uint8_t>& values) {
    uint64_t sum = 0;
    for (std::size_t i = 0; i < values.size(); ++i)
        sum += static_cast<uint64_t>(values[i]) * (i % 65521 + 1);
    return sum % (uint64_t{1} << 61);
}

ImageProcessorConfig processor_config() {
    ImageProcessorConfig config;
    config.max_piece_patches = 2048;
    return config;
}

struct GridCase {
    int32_t height, width, rows, cols;
    std::vector<int32_t> pieces; // patch rows, patch cols per piece
};

void test_slicing_and_piece_sizes() {
    const std::vector<GridCase> cases{
        {468, 931, 1, 2, {22, 44, 32, 32, 32, 32}},
        {36, 870, 0, 0, {6, 158}},
        {240, 931, 1, 3, {16, 62, 28, 36, 28, 36, 28, 36}},
        {1507, 604, 4, 1, {50, 20, 26, 40, 26, 40, 26, 40, 26, 40}},
        {13, 2137, 0, 0, {2, 400}},
        {1430, 1340, 3, 3, {32, 30, 32, 30, 32, 30, 32, 30, 32, 30,
                            32, 30, 32, 30, 32, 30, 32, 30, 32, 30}},
        {448, 448, 0, 0, {32, 32}},
        {100, 100, 0, 0, {32, 32}},
        {2000,
         1000,
         4,
         2,
         {46, 22, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32}},
        {50, 30, 0, 0, {42, 24}},
    };
    const auto config = processor_config();
    for (const auto& c : cases) {
        auto pixels = pattern(c.height, c.width);
        const trtmc::internal::ImageView view{pixels.data(),
                                              pixels.size(),
                                              static_cast<uint32_t>(c.height),
                                              static_cast<uint32_t>(c.width),
                                              3,
                                              trtmc::internal::ImageFormat::UInt8};
        const auto image = preprocess_image(config, view);
        const std::string name = std::to_string(c.height) + "x" + std::to_string(c.width);
        check(image.grid.rows == c.rows && image.grid.cols == c.cols, name + " slice grid");
        std::vector<int32_t> pieces;
        for (const auto& piece : image.pieces) {
            pieces.push_back(piece.height);
            pieces.push_back(piece.width);
            check(piece.patches.size() ==
                          static_cast<std::size_t>(piece.height) * piece.width * 588 &&
                      piece.position_ids.size() ==
                          static_cast<std::size_t>(piece.height) * piece.width,
                  name + " piece storage");
        }
        check(pieces == c.pieces, name + " piece patch grids");
    }
}

void test_resize_matches_reference() {
    struct ResizeCase {
        int32_t height, width, out_height, out_width;
        uint64_t sum;
    };
    const std::vector<ResizeCase> cases{
        {37, 53, 120, 77, 49179038647ULL},     {468, 931, 308, 616, 2318819396484ULL},
        {36, 870, 84, 2492, 2556719067551ULL}, {64, 64, 64, 30, 2116735503ULL},
        {31, 40, 90, 40, 7466351863ULL},       {500, 300, 31, 29, 463797455ULL},
    };
    for (const auto& c : cases) {
        const auto out =
            resize_rgb8(pattern(c.height, c.width), c.height, c.width, c.out_height, c.out_width);
        check(checksum(out) == c.sum,
              "resize " + std::to_string(c.height) + "x" + std::to_string(c.width) + " -> " +
                  std::to_string(c.out_height) + "x" + std::to_string(c.out_width));
    }
}

void test_normalization_and_window_order() {
    auto config = processor_config();
    // 28x28 RGB image: one 2x2 window, no resize (it is upscaled to 448x448).
    std::vector<uint8_t> pixels(28 * 28 * 3, 255);
    const trtmc::internal::ImageView view{
        pixels.data(), pixels.size(), 28, 28, 3, trtmc::internal::ImageFormat::UInt8};
    const auto image = preprocess_image(config, view);
    check(image.pieces.size() == 1 && image.pieces[0].height == 32 && image.pieces[0].width == 32,
          "small image upscaled to the slice resolution");
    bool all_one = true;
    for (const float value : image.pieces[0].patches)
        all_one = all_one && value == 1.0F;
    check(all_one, "white pixels normalize to 1");
    const auto ids = image.pieces[0].position_ids;
    // First merge window = patches (0,0), (0,1), (1,0), (1,1) of a 32x32 grid.
    const auto rows = nearest_position_ids(32, 32, 70);
    check(ids[0] == rows[0] && ids[1] == rows[1] && ids[2] == rows[32] && ids[3] == rows[33],
          "patches are grouped by 2x2 merge window");
    check(rows[1] == 2 && rows[32] == 140, "nearest position buckets of a 32x32 grid");
    // Extents whose fractions k / n hit bucket boundaries exactly; expected rows
    // are the reference's buckets evaluated on the GPU.
    const auto buckets = [](int32_t extent) {
        std::vector<int32_t> result;
        for (const auto id : nearest_position_ids(extent, 1, 70))
            result.push_back(id / 70);
        return result;
    };
    check(buckets(14) == std::vector<int32_t>{0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65},
          "position buckets of a 14-patch extent");
    check(buckets(28) == std::vector<int32_t>{0,  2,  5,  7,  10, 12, 15, 17, 20, 22,
                                              25, 27, 30, 32, 35, 37, 40, 42, 45, 47,
                                              50, 52, 55, 57, 60, 62, 65, 67},
          "position buckets of a 28-patch extent");
    std::vector<int32_t> identity(70);
    for (int32_t i = 0; i < 70; ++i)
        identity[i] = i;
    check(buckets(70) == identity, "position buckets of a 70-patch extent");
}

void test_chat_template() {
    const PromptTokens tokens;
    const std::vector<PromptMessage> single{
        {PromptRole::User, {{true, {}}, {false, "Text Recognition:"}}}};
    check(render_chat(single, false, tokens) ==
              "<|im_start|>user\n<|image_pad|>\nText Recognition:<|im_end|>\n<|im_start|>"
              "assistant\n<think>\n\n</think>\n\n",
          "single image prompt");
    check(render_chat(single, true, tokens) ==
              "<|im_start|>user\n<|image_pad|>\nText Recognition:<|im_end|>\n<|im_start|>"
              "assistant\n<think>\n",
          "thinking generation prompt");
    const std::vector<PromptMessage> system_two{
        {PromptRole::System, {{false, "  Be exact.\n"}}},
        {PromptRole::User, {{false, "Compare"}, {true, {}}, {true, {}}, {false, "  "}}}};
    check(render_chat(system_two, false, tokens) ==
              "<|im_start|>system\nBe exact.<|im_end|>\n<|im_start|>user\nCompare\n<|image_pad|>"
              "\n<|image_pad|><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
          "system message and two images");
    const std::vector<PromptMessage> history{
        {PromptRole::User, {{true, {}}, {false, "Read it."}}},
        {PromptRole::Assistant, {{false, "<think>\nplan\n</think>\n\nHello"}}},
        {PromptRole::User, {{false, "Again"}}}};
    check(render_chat(history, false, tokens) ==
              "<|im_start|>user\n<|image_pad|>\nRead it.<|im_end|>\n<|im_start|>assistant\n<think>"
              "\n\n</think>\n\nHello<|im_end|>\n<|im_start|>user\nAgain<|im_end|>\n<|im_start|>"
              "assistant\n<think>\n\n</think>\n\n",
          "assistant history drops earlier reasoning");
    bool rejected = false;
    try {
        (void)render_chat({{PromptRole::System, {{true, {}}}}, {PromptRole::User, {{false, "x"}}}},
                          false, tokens);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "images in a system message are rejected");
}

void test_image_expansion() {
    const PromptTokens tokens;
    const std::string rendered = "a<|image_pad|>b<|image_pad|>c";
    std::vector<ImageTokenLayout> images(2);
    images[0].overview_tokens = 2;
    images[1].overview_tokens = 1;
    images[1].grid_rows = 2;
    images[1].grid_cols = 2;
    images[1].slice_tokens = {1, 2, 1, 1};
    const std::string expected =
        "a<image_id>0</image_id><image><|image_pad|><|image_pad|></image>b"
        "<image_id>1</image_id><image><|image_pad|></image>"
        "<slice><|image_pad|></slice><slice><|image_pad|><|image_pad|></slice>\n"
        "<slice><|image_pad|></slice><slice><|image_pad|></slice>c";
    check(expand_images(rendered, images, true, tokens) == expected, "placeholder expansion");
    bool rejected = false;
    try {
        (void)expand_images(rendered, {images[0]}, true, tokens);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "placeholder and image counts must agree");
    check(python_strip("\xe3\x80\x80 x\t\n") == "x", "Python strip of Unicode whitespace");
}

// Expected strings are Python's bytes.decode("utf-8", errors="replace").
void test_invalid_utf8_replacement() {
    const std::string fffd = "\xef\xbf\xbd";
    const std::vector<std::pair<std::string, std::string>> cases{
        {"abc\xe4\xb8", "abc" + fffd},                   // generation cut inside a CJK character
        {"\xf0\x9f\x99", fffd},                          // cut inside an emoji
        {"\xe4\xb8x", fffd + "x"},                       // truncated sequence followed by ASCII
        {"\xe0\x80", fffd + fffd},                       // overlong lead: each byte replaced
        {"\xc0\xaf", fffd + fffd},                       // invalid lead byte
        {"\xed\xa0\x80", fffd + fffd + fffd},            // UTF-16 surrogate
        {"\xf4\x90\x80\x80", fffd + fffd + fffd + fffd}, // above U+10FFFF
        {"\x80\xff", fffd + fffd},                       // stray continuation and 0xFF
        {"\xc3\xa9\xe4\xb8\xad\xf0\x9f\x99\x82", "\xc3\xa9\xe4\xb8\xad\xf0\x9f\x99\x82"},
    };
    for (const auto& [input, expected] : cases)
        check(replace_invalid_utf8(input) == expected,
              "UTF-8 replacement of " + std::to_string(input.size()) + "-byte input");
}

} // namespace

int main() {
    try {
        test_slicing_and_piece_sizes();
        test_resize_matches_reference();
        test_normalization_and_window_order();
        test_chat_template();
        test_image_expansion();
        test_invalid_utf8_replacement();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "minicpm_ocr host checks passed\n";
    return 0;
}

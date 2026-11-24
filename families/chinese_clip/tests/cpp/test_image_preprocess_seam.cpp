/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/chinese_clip/runtime/image_preprocess_seam.h"
#include "families/chinese_clip/runtime/tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
std::size_t allocation_limit = std::numeric_limits<std::size_t>::max();
struct AllocationBound {
    explicit AllocationBound(std::size_t limit) { allocation_limit = limit; }
    ~AllocationBound() { allocation_limit = std::numeric_limits<std::size_t>::max(); }
};
} // namespace

void* operator new(std::size_t bytes) {
    if (bytes > allocation_limit)
        throw std::bad_alloc();
    if (void* pointer = std::malloc(bytes ? bytes : 1))
        return pointer;
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept {
    std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void check_close(float actual, float expected, float tolerance, const char* name) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        !(std::fabs(actual - expected) <= tolerance)) {
        std::cerr << "FAIL: " << name << " actual=" << actual << " expected=" << expected << '\n';
        ++g_failures;
    }
}

trtmc::ChineseClipPreprocessConfig base_config() {
    trtmc::ChineseClipPreprocessConfig config;
    config.input_image_h = 224;
    config.input_image_w = 224;
    return config;
}

void test_resize_preserves_the_full_rectangular_source() {
    const auto config = base_config();
    for (const auto shape : {trtmc::compute_chinese_clip_resize_shape(300, 400, config),
                             trtmc::compute_chinese_clip_resize_shape(400, 300, config),
                             trtmc::compute_chinese_clip_resize_shape(224, 224, config)}) {
        check(shape.height == 224 && shape.width == 224, "direct resize uses both target axes");
    }
}

void test_resize_matches_pillow_bicubic_pixels() {
    auto config = base_config();
    config.input_image_h = config.input_image_w = 3;
    config.image_mean = {0, 0, 0};
    config.image_std = {1, 1, 1};
    std::vector<float> source(24);
    for (std::size_t index = 0; index < source.size(); ++index)
        source[index] = static_cast<float>(index * 10) / 255.0F;
    // Pillow RGB resize of the 2x4 source to 3x3, BICUBIC; planar channel order.
    const std::vector<int> expected = {0,  37, 76, 66, 105, 144, 134, 173, 212,
                                       8,  47, 86, 76, 115, 154, 144, 183, 222,
                                       18, 57, 96, 86, 125, 164, 154, 193, 232};
    const auto result = trtmc::preprocess_chinese_clip_image(source.data(), 2, 4, config);
    check(result.size() == expected.size(), "Pillow golden output shape");
    if (result.size() != expected.size())
        return;
    for (std::size_t index = 0; index < expected.size(); ++index)
        check_close(result[index], static_cast<float>(expected[index]) / 255.0F, 1e-7F,
                    "direct bicubic resize matches Pillow");
}

void test_normalisation_is_per_channel() {
    // A flat image cannot tell interleaved from planar, so each channel gets
    // its own value and its own statistics.
    auto config = base_config();
    config.input_image_h = 224;
    config.input_image_w = 224;
    config.image_mean = {0.1F, 0.2F, 0.3F};
    config.image_std = {0.5F, 0.25F, 0.125F};
    std::vector<float> pixels(static_cast<std::size_t>(224) * 224 * 3);
    for (std::size_t pixel = 0; pixel < 224U * 224U; ++pixel) {
        pixels[pixel * 3] = 0.6F;
        pixels[pixel * 3 + 1] = 0.4F;
        pixels[pixel * 3 + 2] = 0.8F;
    }
    const auto values = trtmc::preprocess_chinese_clip_image(pixels.data(), 224, 224, config);
    const std::size_t plane = 224U * 224U;
    check(values.size() == 3 * plane, "normalization output shape");
    if (values.size() != 3 * plane)
        return;
    check_close(values[0], (0.6F - 0.1F) / 0.5F, 2e-2F, "red channel normalised");
    check_close(values[plane], (0.4F - 0.2F) / 0.25F, 2e-2F, "green channel normalised");
    check_close(values[2 * plane], (0.8F - 0.3F) / 0.125F, 2e-2F, "blue channel normalised");
}

void test_invalid_input_is_refused() {
    auto config = base_config();
    bool threw = false;
    try {
        trtmc::preprocess_chinese_clip_image(nullptr, 224, 224, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a null image is refused");

    threw = false;
    config.image_std = {1.0F, 0.0F, 1.0F};
    std::vector<float> pixels(3U * 4U * 4U, 0.5F);
    try {
        trtmc::preprocess_chinese_clip_image(pixels.data(), 4, 4, config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a zero standard deviation is refused");
}

void test_resize_pass_order_matches_pillow13() {
    struct Case {
        int height, width, output;
        std::vector<int> expected;
    };
    // Independent Pillow 13 RGB goldens for ((index * 29 + 7) % 256) bytes.
    const std::vector<Case> cases{
        {513, 257, 2, {128, 128, 127, 127, 128, 128, 127, 127, 128, 128, 127, 127}},
        {17, 5, 3, {114, 123, 119, 132, 121, 138, 115, 120, 121, 139, 140, 145, 118, 115,
                    125, 145, 133, 152, 132, 128, 112, 132, 133, 131, 130, 117, 114}},
        {5, 17, 3, {106, 119, 124, 139, 143, 112, 110, 121, 131, 135, 149, 134, 118, 124,
                    124, 139, 151, 133, 122, 140, 117, 127, 135, 125, 127, 130, 112}},
    };
    for (const auto& item : cases) {
        auto config = base_config();
        config.input_image_h = config.input_image_w = item.output;
        config.image_mean = {0, 0, 0};
        config.image_std = {1, 1, 1};
        std::vector<float> pixels(static_cast<std::size_t>(item.height) * item.width * 3);
        for (std::size_t index = 0; index < pixels.size(); ++index)
            pixels[index] = static_cast<float>((index * 29 + 7) % 256) / 255.0F;
        const auto result =
            trtmc::preprocess_chinese_clip_image(pixels.data(), item.height, item.width, config);
        check(result.size() == item.expected.size(), "Pillow 13 resize shape");
        if (result.size() != item.expected.size())
            continue;
        for (std::size_t index = 0; index < result.size(); ++index)
            check_close(result[index], static_cast<float>(item.expected[index]) / 255.0F, 1e-7F,
                        "Pillow 13 resize pass order and rounding");
    }
}

void test_postprocessor_special_ids_are_validated() {
    const std::string fixture =
        R"({"model":{"type":"WordPiece","unk_token":"[UNK]","vocab":{"[UNK]":0,"[CLS]":1,"[SEP]":2,"[PAD]":3,"hello":4}},"post_processor":{"type":"RobertaProcessing","cls":["[CLS]",1],"sep":["[SEP]",2]}})";
    for (const auto& type : {std::string("RobertaProcessing"), std::string("BertProcessing")}) {
        auto valid = fixture;
        const auto offset = valid.find("RobertaProcessing");
        valid.replace(offset, std::string("RobertaProcessing").size(), type);
        const auto tokenizer = trtmc::CreateWordPieceTokenizer(valid.data(), valid.size());
        check(tokenizer->encode("hello") == std::vector<int32_t>{1, 4, 2},
              "recognized postprocessor preserves vocabulary CLS/SEP framing");
        const std::string old = R"("[CLS]",1)";
        valid.replace(valid.find(old), old.size(), R"("[CLS]",0)");
        bool rejected = false;
        try {
            (void)trtmc::CreateWordPieceTokenizer(valid.data(), valid.size());
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        check(rejected, "recognized postprocessor rejects a mismatched CLS ID");
    }
}

void test_tall_image_has_bounded_resize_allocations() {
    auto config = base_config();
    config.image_mean = {0, 0, 0};
    config.image_std = {1, 1, 1};
    std::vector<float> pixels(65536U * 3U, 0.25F);
    try {
        AllocationBound bound(16U * 1024U * 1024U);
        const auto result = trtmc::preprocess_chinese_clip_image(pixels.data(), 65536, 1, config);
        check(result.size() ==
                  static_cast<std::size_t>(config.input_image_h) * config.input_image_w * 3,
              "tall image output shape");
        for (float value : result)
            check_close(value, 64.0F / 255.0F, 1e-7F, "tall constant image value");
    } catch (const std::bad_alloc&) {
        check(false, "resize must not allocate a source-height by output-width intermediate");
    }
}

} // namespace

int main() {
    test_postprocessor_special_ids_are_validated();
    test_resize_pass_order_matches_pillow13();
    test_tall_image_has_bounded_resize_allocations();
    test_resize_preserves_the_full_rectangular_source();
    test_resize_matches_pillow_bicubic_pixels();
    test_normalisation_is_per_channel();
    test_invalid_input_is_refused();
    if (g_failures == 0)
        std::cout << "chinese_clip image preprocess seam: all checks passed\n";
    return g_failures == 0 ? 0 : 1;
}

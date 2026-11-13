/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Bundle-pixel regression: runtime.json normalization must reach the pixels
// produced by the native preprocessor. The header defaults are CLIP values, so
// a consumer that ignores the bundle's ImageNet metadata fails this test.

#include "families/internvl/runtime/image_preprocessor.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

constexpr std::array<float, 3> kImageNetMean{0.485F, 0.456F, 0.406F};
constexpr std::array<float, 3> kImageNetStd{0.229F, 0.224F, 0.225F};

// Runtime metadata emitted by get_vl_config for the pinned InternVL3 checkpoints.
std::string runtime_json() {
    const nlohmann::json document = {
        {"preprocessor_type", "simple_chw"},
        {"image_token_id", 151667},
        {"fixed_image_size", 448},
        {"patch_size", 14},
        {"merge_size", 2},
        {"temporal_patch_size", 1},
        {"num_image_pad_tokens", 256},
        {"vision_output_dim", 1536},
        {"vl_prompt_template", "{image_pads}{prompt}"},
        {"image_token_str", "<IMG_CONTEXT>"},
        {"interpolation", "bicubic"},
        {"image_mean", kImageNetMean},
        {"image_std", kImageNetStd},
    };
    return document.dump();
}

void check_uniform_image(const trtmc::InternVlPreprocessConfig& config, std::uint8_t value) {
    trtmc::runtime::adapters::io::DecodedImage image;
    image.width = 64;
    image.height = 64;
    image.channels = 3;
    image.pixels.assign(static_cast<std::size_t>(image.width) * image.height * image.channels,
                        value);
    const auto result = trtmc::internvl_preprocess_decoded_image(image, config);
    const std::string label = value ? "white" : "black";
    check(result.ok && !result.pixel_values.empty(), label + " preprocesses");

    std::array<float, 3> expected{};
    for (std::size_t channel = 0; channel < expected.size(); ++channel)
        expected[channel] =
            (static_cast<float>(value) / 255.0F - kImageNetMean[channel]) / kImageNetStd[channel];

    std::array<bool, 3> seen{};
    bool all_expected = true;
    for (const float pixel : result.pixel_values) {
        bool matched = false;
        for (std::size_t channel = 0; channel < expected.size(); ++channel) {
            if (std::fabs(pixel - expected[channel]) <= 1.0e-5F) {
                seen[channel] = true;
                matched = true;
            }
        }
        all_expected = all_expected && matched;
    }
    check(all_expected, label + " pixels use the runtime.json ImageNet normalization");
    check(seen[0] && seen[1] && seen[2], label + " pixels cover every channel");
}

void check_dynamic_tiles(bool portrait) {
    auto document = nlohmann::json::parse(runtime_json());
    document["preprocessor_type"] = "dynamic_tiles_chw";
    document["max_image_tiles"] = 2;
    // Match get_vl_config, which declares the thumbnail explicitly.
    document["use_thumbnail"] = true;
    document["fixed_image_size"] = 4;
    document["image_mean"] = std::array<float, 3>{0, 0, 0};
    document["image_std"] = std::array<float, 3>{1, 1, 1};
    const auto config = trtmc::internvl_parse_preprocess_config(document.dump());
    trtmc::runtime::adapters::io::DecodedImage image;
    image.width = portrait ? 4 : 8;
    image.height = portrait ? 8 : 4;
    image.channels = 3;
    image.pixels.assign(image.width * image.height * 3, 0);
    for (int y = 0; y < image.height; ++y)
        for (int x = 0; x < image.width; ++x) {
            const bool first = portrait ? y < 4 : x < 4;
            image.pixels[(y * image.width + x) * 3 + (first ? 0 : 2)] = 255;
        }
    const auto result = trtmc::internvl_preprocess_decoded_image(image, config);
    check(result.ok && result.num_tiles == 3 && result.pixel_values.size() == 144,
          "two local tiles plus one thumbnail");
    if (result.pixel_values.size() != 144)
        return;
    for (int pixel = 0; pixel < 16; ++pixel) {
        check(result.pixel_values[pixel] == 1 && result.pixel_values[32 + pixel] == 0,
              "first tile is red in row-major order");
        check(result.pixel_values[48 + pixel] == 0 && result.pixel_values[80 + pixel] == 1,
              "second tile is blue in row-major order");
    }
    check(result.pixel_values[96] > result.pixel_values[111],
          "thumbnail follows local tiles and preserves the complete image");
    auto prompt_config = config;
    prompt_config.num_image_pad_tokens *= result.num_tiles;
    const auto prompt = trtmc::internvl_format_prompt("Question?", prompt_config);
    check(prompt.size() == 3 * 256 * config.image_token_str.size() + 9,
          "prompt reserves every tile's image tokens");
}

} // namespace

int main() {
    const auto config = trtmc::internvl_parse_preprocess_config(runtime_json());
    check_uniform_image(config, 0);
    check_uniform_image(config, 255);
    check_dynamic_tiles(false);
    check_dynamic_tiles(true);
    if (g_failures != 0) {
        std::cerr << g_failures << " InternVL image normalization check(s) failed\n";
        return 1;
    }
    std::cout << "InternVL image normalization tests passed\n";
    return 0;
}

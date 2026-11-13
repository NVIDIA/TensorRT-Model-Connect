/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Bundle-pixel regression: runtime.json normalization must reach the pixels
// produced by the native preprocessor for both Qwen-VL generations.

#include "families/qwen_vl/runtime/image_preprocessor.h"

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

struct Generation {
    const char* name;
    int patch_size;
    std::array<float, 3> mean;
    std::array<float, 3> standard_deviation;
};

// Runtime metadata emitted by get_vl_config for the pinned manifest checkpoints.
std::string runtime_json(const Generation& generation) {
    const int fixed = 448;
    const int merged = (fixed / generation.patch_size) * (fixed / generation.patch_size) / 4;
    const nlohmann::json document = {
        {"preprocessor_type", "merge_group_chw"},
        {"image_token_id", 151655},
        {"fixed_image_size", fixed},
        {"fixed_image_height", fixed},
        {"fixed_image_width", fixed},
        {"patch_size", generation.patch_size},
        {"merge_size", 2},
        {"temporal_patch_size", 2},
        {"num_image_pad_tokens", merged},
        {"vision_output_dim", 2048},
        {"vl_prompt_template", "{image_pads}{prompt}"},
        {"image_token_str", "<|image_pad|>"},
        {"interpolation", "bicubic"},
        {"image_mean", generation.mean},
        {"image_std", generation.standard_deviation},
        {"dynamic_image_resolution", false},
    };
    return document.dump();
}

void check_uniform_image(const Generation& generation, const trtmc::QwenVlPreprocessConfig& config,
                         std::uint8_t value) {
    trtmc::runtime::adapters::io::DecodedImage image;
    image.width = 64;
    image.height = 64;
    image.channels = 3;
    image.pixels.assign(static_cast<std::size_t>(image.width) * image.height * image.channels,
                        value);
    const auto result = trtmc::qwen_vl_preprocess_decoded_image(image, config);
    const std::string label = std::string(generation.name) + (value ? " white" : " black");
    check(result.ok && !result.pixel_values.empty(), label + " preprocesses");

    std::array<float, 3> expected{};
    for (std::size_t channel = 0; channel < expected.size(); ++channel)
        expected[channel] = (static_cast<float>(value) / 255.0F - generation.mean[channel]) /
                            generation.standard_deviation[channel];

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
    check(all_expected, label + " pixels use the runtime.json normalization");
    check(seen[0] && seen[1] && seen[2], label + " pixels cover every channel");
}

void check_qwen3_smart_geometry(int height, int width) {
    trtmc::QwenVlPreprocessConfig config;
    config.preprocessor_type = "qwen3_smart_resize_patchify";
    config.dynamic_image_resolution = true;
    config.patch_size = 16;
    config.min_pixels = 1024;
    config.max_pixels = 448 * 448;
    config.vision_embed_dim = 1024;
    config.vision_num_heads = 16;
    config.vision_position_grid_size = 48;
    for (int channel = 0; channel < 3; ++channel) {
        config.image_mean[channel] = 0.5F;
        config.image_std[channel] = 0.5F;
    }
    trtmc::runtime::adapters::io::DecodedImage image;
    image.width = width;
    image.height = height;
    image.channels = 3;
    image.pixels.resize(height * width * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image.pixels[(y * width + x) * 3] = x;
            image.pixels[(y * width + x) * 3 + 1] = y;
            image.pixels[(y * width + x) * 3 + 2] = 128;
        }
    }
    const auto result = trtmc::qwen_vl_preprocess_decoded_image(image, config);
    check(result.ok, "Qwen3 smart geometry preprocesses");
    check(result.height == height && result.width == width,
          "aligned portrait/landscape/square images keep their original geometry");
    const int gh = height / 16, gw = width / 16;
    check(result.image_grid_hws == std::vector<int32_t>({gh, gw}), "actual image grid");
    check(result.pixel_values.size() == static_cast<std::size_t>(height * width * 6),
          "no extra square image patches");
    for (int i = 0; i < gh * gw; ++i) {
        const int group = i / 4;
        const int row = group / (gw / 2) * 2 + (i % 4) / 2;
        const int col = group % (gw / 2) * 2 + i % 2;
        const int patch_base = i * 6 * 16 * 16;
        const float red = (col * 16 / 255.0F - 0.5F) / 0.5F;
        const float green = (row * 16 / 255.0F - 0.5F) / 0.5F;
        check(std::fabs(result.pixel_values[patch_base] - red) < 1e-5F,
              "merge-group patch x coordinate");
        check(std::fabs(result.pixel_values[patch_base + 2 * 16 * 16] - green) < 1e-5F,
              "merge-group patch y coordinate");
        float h = 0, w = 0, weight = 0;
        for (int corner = 0; corner < 4; ++corner) {
            const int index = result.vision_position_indices[corner][i];
            const float blend = result.vision_position_weights[corner][i];
            check(index >= 0 && index < 48 * 48, "learned position index in bounds");
            h += blend * (index / 48);
            w += blend * (index % 48);
            weight += blend;
        }
        check(std::fabs(weight - 1.0F) < 1e-5F, "position weights form a partition of unity");
        check(std::fabs(h - row * 47.0F / (gh - 1)) < 1e-5F,
              "interpolation preserves linear height coordinates");
        check(std::fabs(w - col * 47.0F / (gw - 1)) < 1e-5F,
              "interpolation preserves linear width coordinates");
        check(std::fabs(result.vision_cos_half[i * 32] - std::cos(float(row))) < 1e-6F,
              "spatial RoPE follows merge-group rows without window permutation");
    }
}

} // namespace

int main() {
    const Generation qwen25{"Qwen2.5-VL",
                            14,
                            {0.48145466F, 0.4578275F, 0.40821073F},
                            {0.26862954F, 0.26130258F, 0.27577711F}};
    const Generation qwen3{"Qwen3-VL", 16, {0.5F, 0.5F, 0.5F}, {0.5F, 0.5F, 0.5F}};
    for (const Generation* generation : {&qwen25, &qwen3}) {
        const auto config = trtmc::qwen_vl_parse_preprocess_config(runtime_json(*generation));
        check_uniform_image(*generation, config, 0);
        check_uniform_image(*generation, config, 255);
    }
    check_qwen3_smart_geometry(64, 96);
    check_qwen3_smart_geometry(96, 64);
    check_qwen3_smart_geometry(64, 64);
    if (g_failures != 0) {
        std::cerr << g_failures << " Qwen-VL image normalization check(s) failed\n";
        return 1;
    }
    std::cout << "Qwen-VL image normalization tests passed\n";
    return 0;
}

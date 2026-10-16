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
    if (g_failures != 0) {
        std::cerr << g_failures << " Qwen-VL image normalization check(s) failed\n";
        return 1;
    }
    std::cout << "Qwen-VL image normalization tests passed\n";
    return 0;
}

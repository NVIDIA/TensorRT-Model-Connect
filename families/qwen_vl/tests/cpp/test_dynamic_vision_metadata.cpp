/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen_vl/runtime/image_preprocessor.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    nlohmann::json cases;
    std::ifstream(argv[1]) >> cases;
    int failures = 0;
    for (const auto& sample : cases) {
        trtmc::QwenVlPreprocessConfig config;
        config.preprocessor_type = "qwen3_smart_resize_patchify";
        config.dynamic_image_resolution = true;
        config.patch_size = 16;
        config.vision_embed_dim = 1024;
        config.vision_num_heads = 16;
        config.min_pixels = 1024;
        config.max_pixels = 16777216;
        config.vision_position_grid_size = sample.at("side").get<int>();
        trtmc::runtime::adapters::io::DecodedImage image;
        image.height = sample.at("height").get<int>() * config.patch_size;
        image.width = sample.at("width").get<int>() * config.patch_size;
        image.channels = 3;
        image.pixels.assign(static_cast<std::size_t>(image.height) * image.width * 3, 128);
        const auto actual = trtmc::qwen_vl_preprocess_decoded_image(image, config);
        const auto indices = sample.at("indices").get<std::vector<int32_t>>();
        const auto weights = sample.at("weights").get<std::vector<float>>();
        const auto patches = static_cast<std::size_t>(image.height / 16) * (image.width / 16);
        bool matches = actual.ok && indices.size() == 4 * patches && weights.size() == 4 * patches;
        for (std::size_t corner = 0; matches && corner < 4; ++corner) {
            matches = actual.vision_position_indices[corner].size() == patches &&
                      actual.vision_position_weights[corner].size() == patches;
            for (std::size_t patch = 0; matches && patch < patches; ++patch) {
                const auto offset = corner * patches + patch;
                matches = actual.vision_position_indices[corner][patch] == indices[offset] &&
                          std::fabs(actual.vision_position_weights[corner][patch] -
                                    weights[offset]) <= 5.0e-6F;
            }
        }
        matches = matches && actual.vision_cos_half.size() == patches * 32 &&
                  actual.vision_sin_half.size() == patches * 32 &&
                  actual.vision_window_mask.empty() && actual.vision_padded_window_indices.empty();
        if (!matches) {
            std::cerr << "Qwen3-VL metadata differs from HF: " << image.height << 'x' << image.width
                      << '\n';
            ++failures;
        }
        // The old Qwen2.5 path still generates window padding and no learned
        // position interpolation when the optional grid metadata is absent.
        config.preprocessor_type = "qwen_smart_resize_patchify";
        config.vision_position_grid_size = 0;
        const auto windowed = trtmc::qwen_vl_preprocess_decoded_image(image, config);
        const bool no_indices = std::all_of(windowed.vision_position_indices.begin(),
                                            windowed.vision_position_indices.end(),
                                            [](const auto& values) { return values.empty(); });
        const bool no_weights = std::all_of(windowed.vision_position_weights.begin(),
                                            windowed.vision_position_weights.end(),
                                            [](const auto& values) { return values.empty(); });
        if (!windowed.ok || !no_indices || !no_weights || windowed.vision_window_mask.empty())
            ++failures;
    }
    return failures == 0 ? 0 : 1;
}

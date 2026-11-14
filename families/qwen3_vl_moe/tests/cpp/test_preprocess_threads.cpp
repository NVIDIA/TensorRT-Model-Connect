/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_vl_moe/runtime/image_preprocessor.cpp"

#include <atomic>

int main(int argc, char**) {
    if (argc > 1) {
        nlohmann::json requests;
        std::cin >> requests;
        nlohmann::json results = nlohmann::json::array();
        for (const auto& request : requests) {
            trtmc::Qwen3VlMoePreprocessConfig config;
            config.preprocessor_type = "qwen3_smart_resize_patchify";
            config.dynamic_image_resolution = true;
            config.patch_size = 16;
            config.vision_embed_dim = 72;
            config.vision_num_heads = 1;
            config.vision_position_grid_size = 2;
            config.min_pixels = 1024;
            config.max_pixels = 16777216;
            trtmc::runtime::adapters::io::DecodedImage image;
            image.height = request.at("height").get<int>() * 16;
            image.width = request.at("width").get<int>() * 16;
            image.channels = 3;
            image.pixels.assign(static_cast<std::size_t>(image.height) * image.width * 3, 128);
            const auto actual = trtmc::qwen3_vl_moe_preprocess_decoded_image(image, config);
            if (!actual.ok)
                return 1;
            results.push_back({{"cos", actual.vision_cos_half}, {"sin", actual.vision_sin_half}});
        }
        std::cout << results.dump() << '\n';
        return 0;
    }
    std::atomic<int> completed{0};
    const auto workers = trtmc::preprocess_worker_count();
    bool caught = false;
    try {
        trtmc::parallel_for_ranges(64, 1, [&](int, int end) {
            if (end == 64)
                throw std::runtime_error("injected exception after workers started");
            ++completed;
        });
    } catch (const std::runtime_error&) {
        caught = true;
    }
    return caught && completed == workers - 1 ? 0 : 1;
}

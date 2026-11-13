/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Dynamic tiling must follow InternVLProcessor (GotOcr2 crop_to_patches). The
// expected grids come from transformers' get_optimal_tiled_canvas with 448 px
// tiles and 1..12 patches.

#include "families/internvl/runtime/image_preprocessor.h"
#include "families/internvl/runtime/internvl_generation_budget.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

void test_tile_grid_matches_processor() {
    struct Case {
        int32_t height;
        int32_t width;
        std::pair<int32_t, int32_t> grid; // (columns, rows)
    };
    const std::vector<Case> cases = {
        {332, 500, {3, 2}},   {720, 1280, {4, 2}}, {682, 1024, {3, 2}}, {27, 104, {4, 1}},
        {60, 164, {5, 2}},    {448, 448, {1, 1}},  {500, 500, {1, 1}},  {1000, 100, {1, 10}},
        {100, 1000, {10, 1}}, {448, 896, {2, 1}},  {13, 1700, {12, 1}}, {896, 448, {1, 2}},
    };
    for (const auto& item : cases) {
        const auto grid = trtmc::internvl_tile_grid(item.height, item.width, 448, 1, 12);
        check(grid == item.grid,
              "tile grid for " + std::to_string(item.height) + "x" + std::to_string(item.width));
    }
}

trtmc::InternVlPreprocessConfig tiling_config() {
    trtmc::InternVlPreprocessConfig config;
    config.preprocessor_type = "dynamic_tiles_chw";
    config.fixed_image_size = 448;
    config.in_channels = 3;
    config.min_image_tiles = 1;
    config.max_image_tiles = 12;
    config.use_thumbnail = true;
    config.num_image_pad_tokens = 256;
    config.image_token_str = "<IMG_CONTEXT>";
    config.vl_prompt_template = "<img>{image_pads}</img>\n{prompt}";
    return config;
}

void test_tiles_and_thumbnail() {
    trtmc::runtime::adapters::io::DecodedImage image;
    image.width = 164;
    image.height = 60;
    image.channels = 3;
    image.pixels.assign(static_cast<std::size_t>(image.width) * image.height * 3, 128);
    const auto result = trtmc::internvl_preprocess_decoded_image(image, tiling_config());
    check(result.ok && result.num_tiles == 11, "5x2 tiles plus a thumbnail");
    check(result.pixel_values.size() == 11UL * 3UL * 448UL * 448UL, "one CHW image per tile");

    image.width = 448;
    image.height = 448;
    image.pixels.assign(448UL * 448UL * 3UL, 64);
    const auto single = trtmc::internvl_preprocess_decoded_image(image, tiling_config());
    check(single.ok && single.num_tiles == 1, "a single tile has no thumbnail");
}

void test_prompt_has_256_pads_per_tile() {
    const auto prompt = trtmc::internvl_format_prompt("hi", tiling_config(), 3);
    std::size_t count = 0;
    for (std::size_t at = prompt.find("<IMG_CONTEXT>"); at != std::string::npos;
         at = prompt.find("<IMG_CONTEXT>", at + 1))
        ++count;
    check(count == 768, "256 image tokens per tile");
    check(prompt.rfind("<img>", 0) == 0 && prompt.find("</img>\nhi") != std::string::npos,
          "image tokens framed by <img></img>");
}

} // namespace

// A 13-tile prompt fits the 4096-row cache, but its generation budget must
// stop at the cache rather than slide over image tokens.
void test_generation_budget_respects_the_cache() {
    using trtmc::internvl::generation_budget;
    check(generation_budget(3400, 128, 4096) == 128, "short generation keeps its request");
    check(generation_budget(3400, 4096, 4096) == 696, "generation stops at the KV cache");
    check(generation_budget(100, 64, 0) == 64, "unknown cache length keeps the request");
    bool rejected = false;
    try {
        generation_budget(4096, 1, 4096);
    } catch (const std::invalid_argument& error) {
        rejected = std::string(error.what()).find("KV cache") != std::string::npos;
    }
    check(rejected, "a prompt the KV cache cannot hold is rejected");
}

int main() {
    test_tile_grid_matches_processor();
    test_tiles_and_thumbnail();
    test_prompt_has_256_pads_per_tile();
    test_generation_budget_respects_the_cache();
    if (g_failures != 0) {
        std::cerr << g_failures << " internvl tiling test(s) failed\n";
        return 1;
    }
    return 0;
}

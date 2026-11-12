/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-only contract of the MiniCPM-V 4.7 runtime against values recorded from the provider
// (transformers 5.22.2): image slicing, vision position ids, canvas M-RoPE and the chat format.

#include "families/minicpmv4_7/runtime/image_preprocess.h"
#include "families/minicpmv4_7/runtime/prompt.h"
#include "families/minicpmv4_7/runtime/sampler.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace trtmc::minicpmv4_7;

namespace {

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << "\n";
        ++failures;
    }
}

int64_t digest(const std::vector<int32_t>& values) {
    int64_t sum = 0;
    for (std::size_t i = 0; i < values.size(); ++i)
        sum = (sum + static_cast<int64_t>(i + 1) * (values[i] + 7)) % 1000000007;
    return sum;
}

void slicing() {
    struct Case {
        int32_t height, width, rows, cols, thumb_h, thumb_w, slice_h, slice_w;
    };
    // MiniCPMV4_6ImageProcessor target_sizes (patches) for black images of these sizes.
    const std::vector<Case> cases{
        {382, 640, 1, 2, 24, 40, 36, 28},   {200, 300, 0, 0, 28, 40, 0, 0},
        {1600, 1600, 3, 3, 32, 32, 32, 32}, {10, 2000, 0, 0, 4, 444, 0, 0},
        {2000, 10, 0, 0, 452, 4, 0, 0},     {448, 448, 0, 0, 32, 32, 0, 0},
        {449, 449, 2, 1, 32, 32, 24, 44},   {1000, 3000, 2, 4, 20, 56, 28, 40},
        {3000, 1000, 4, 2, 56, 20, 40, 28}, {700, 701, 2, 2, 32, 32, 32, 32},
        {2500, 1300, 4, 2, 44, 24, 32, 32}, {90, 3500, 1, 3, 4, 196, 8, 116},
    };
    for (const auto& c : cases) {
        const std::string name = std::to_string(c.height) + "x" + std::to_string(c.width);
        const auto grid = sliced_grid(c.height, c.width, 9, 448);
        expect(grid[0] == c.rows && grid[1] == c.cols, "slice grid " + name);
        const auto thumb = best_resize(c.height, c.width, 448, 14, 4, grid[0] == 0);
        expect(thumb[0] / 14 == c.thumb_h && thumb[1] / 14 == c.thumb_w, "thumbnail " + name);
        if (grid[0] > 0) {
            const auto refine = refine_size(c.height, c.width, grid, 448, 14, 4);
            expect(refine[0] / grid[0] / 14 == c.slice_h && refine[1] / grid[1] / 14 == c.slice_w,
                   "slice size " + name);
        }
    }
}

void positions() {
    struct Case {
        int32_t h, w;
        int64_t digest;
    };
    // get_vision_nearest_position_ids(target_sizes, 70)
    const std::vector<Case> cases{{4, 4, 349796},      {24, 40, 464372553}, {36, 28, 631848645},
                                  {28, 28, 984666172}, {32, 32, 676626553}, {140, 4, 512571220},
                                  {68, 52, 207599368}};
    for (const auto& c : cases)
        expect(digest(nearest_position_ids(c.h, c.w, 70)) == c.digest,
               "position ids " + std::to_string(c.h) + "x" + std::to_string(c.w));
    expect(block_order(4, 4) ==
               std::vector<int32_t>{0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15},
           "2x2 block order");
    expect(canvas_coordinates(9, 6, 0) == std::vector<int32_t>{0, 2, 3, 5, 6, 8}, "linspace 9/6");
    expect(canvas_coordinates(6, 3, 10) == std::vector<int32_t>{10, 12, 15},
           "linspace tie to even");
    expect(canvas_coordinates(7, 1, 0) == std::vector<int32_t>{0}, "linspace single");
}

std::vector<int32_t> image_tokens(const TokenIds& t, std::array<int32_t, 2> thumb,
                                  const std::vector<std::array<int32_t, 2>>& slices, int32_t cols) {
    std::vector<int32_t> ids{t.image_start};
    ids.insert(ids.end(), static_cast<std::size_t>(thumb[0] * thumb[1] / 16), t.image_pad);
    ids.push_back(t.image_end);
    for (std::size_t k = 0; k < slices.size(); ++k) {
        if (k && k % static_cast<std::size_t>(cols) == 0)
            ids.push_back(t.newline);
        ids.push_back(t.slice_start);
        ids.insert(ids.end(), static_cast<std::size_t>(slices[k][0] * slices[k][1] / 16),
                   t.image_pad);
        ids.push_back(t.slice_end);
    }
    return ids;
}

void canvas() {
    TokenIds t;
    t.image_pad = 248056;
    t.image_start = 248078;
    t.image_end = 248079;
    t.slice_start = 248088;
    t.slice_end = 248089;
    t.image_id_start = 248090;
    t.image_id_end = 248091;
    t.newline = 198;
    auto join = [](std::vector<std::vector<int32_t>> pieces) {
        std::vector<int32_t> out;
        for (const auto& piece : pieces)
            out.insert(out.end(), piece.begin(), piece.end());
        return out;
    };
    auto repeat = [&](int32_t count) {
        return std::vector<int32_t>(static_cast<std::size_t>(count), 1000);
    };
    struct Case {
        std::vector<int32_t> ids;
        std::vector<std::array<int32_t, 2>> grids;
        int64_t digest;
        int32_t decode;
    };
    const std::vector<std::array<int32_t, 2>> four(4, {28, 28});
    std::vector<std::array<int32_t, 2>> pair{{28, 44}};
    pair.insert(pair.end(), four.begin(), four.end());
    auto both = pair;
    both.insert(both.end(), pair.begin(), pair.end());
    // MiniCPMV4_7Model.get_rope_index on the same sequences.
    const std::vector<Case> cases{
        {join({repeat(5), image_tokens(t, {24, 40}, {{36, 28}, {36, 28}}, 2), repeat(3)}),
         {{24, 40}, {36, 28}, {36, 28}},
         2941071,
         23},
        {join({repeat(2), image_tokens(t, {32, 32}, {}, 1), repeat(4)}), {{32, 32}}, 297264, 15},
        {join({repeat(1),
               image_tokens(t, {28, 44}, four, 2),
               {198, 1000},
               image_tokens(t, {28, 44}, four, 2)}),
         both, 36113584, 32},
    };
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto result = canvas_positions(cases[i].ids, t, cases[i].grids, 4);
        std::vector<int32_t> flat;
        for (const auto& position : result.positions)
            flat.insert(flat.end(), position.begin(), position.end());
        expect(digest(flat) == cases[i].digest && result.decode_position == cases[i].decode,
               "canvas M-RoPE case " + std::to_string(i));
    }
}

void conversation() {
    // Recipe extracted from the checkpoint template at build time (enable_thinking=false).
    const auto recipe = nlohmann::json::parse(R"({
      "default_prefix": "", "user": {"prefix": "<|im_start|>user\n", "suffix": "<|im_end|>\n"},
      "system": {"prefix": "<|im_start|>system\n", "suffix": "<|im_end|>\n"},
      "empty_system_prefix": "<|im_start|>system\n<|im_end|>\n",
      "generation": "<|im_start|>assistant\n<think>\n\n</think>\n\n",
      "assistant_recent": {"prefix": "<|im_start|>assistant\n<think>\n", "include_reasoning": true,
        "reasoning_prefix": "", "reasoning_suffix": "\n</think>\n\n", "empty_reasoning": "\n</think>\n\n",
        "suffix": "<|im_end|>\n"},
      "assistant_history": {"prefix": "<|im_start|>assistant\n<think>\n\n</think>\n\n", "include_reasoning": false,
        "reasoning_prefix": "", "reasoning_suffix": "", "empty_reasoning": "", "suffix": "<|im_end|>\n"}})");
    const std::string image = kImagePlaceholder;
    expect(format_conversation(recipe, {{Role::User, join_parts({image, "What is shown?"})}}) ==
               "<|im_start|>user\n<|image_pad|>\nWhat is shown?<|im_end|>\n<|im_start|>assistant\n"
               "<think>\n\n</think>\n\n",
           "single image turn");
    expect(format_conversation(recipe, {{Role::System, " Be brief. "},
                                        {Role::User, join_parts({image, "Describe."})},
                                        {Role::Assistant, "<think>\nlooking\n</think>\n\nA car."},
                                        {Role::User, join_parts({"Color?", image})}}) ==
               "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\n<|image_pad|>\nDescribe."
               "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nA car.<|im_end|>\n"
               "<|im_start|>user\nColor?\n<|image_pad|><|im_end|>\n<|im_start|>assistant\n"
               "<think>\n\n</think>\n\n",
           "multi-turn conversation");

    ImageCrops crops;
    crops.rows = 1;
    crops.cols = 2;
    crops.crops.resize(3);
    crops.crops[0].grid_h = 8;
    crops.crops[0].grid_w = 4;
    crops.crops[1].grid_h = crops.crops[2].grid_h = 4;
    crops.crops[1].grid_w = crops.crops[2].grid_w = 4;
    expect(expand_images("A" + image + "B", {crops}, 4) ==
               "A<image_id>0</image_id><image>" + image + image + "</image><slice>" + image +
                   "</slice><slice>" + image + "</slice>B",
           "image expansion");
}

void generation() {
    expect(generation_budget(std::nullopt, 4096, 219) == 128, "default budget, large capacity");
    expect(generation_budget(std::nullopt, 64, 30) == 35, "default budget limited by the capacity");
    expect(generation_budget(8, 64, 30) == 8, "requested budget");
    const auto rejected = [](std::optional<int64_t> requested, int64_t capacity, int64_t prompt) {
        try {
            (void)generation_budget(requested, capacity, prompt);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    expect(rejected(36, 64, 30), "requested budget beyond the capacity");
    expect(rejected(std::nullopt, 64, 65), "prompt filling the capacity");
    expect(rejected(std::nullopt, 64, 0), "empty prompt");

    // Top-2 of three equal maxima keeps the lower ids 1 and 2, never id 4 or a smaller logit.
    Sampler top_k(SamplingConfig{1.0, 2, 1.0, 7});
    const std::vector<float> logits{1.0F, 3.0F, 3.0F, 2.0F, 3.0F, 0.0F};
    std::array<int, 6> counts{};
    for (int draw = 0; draw < 300; ++draw)
        ++counts[static_cast<std::size_t>(top_k.sample(logits))];
    expect(counts[1] > 0 && counts[2] > 0 && counts[1] + counts[2] == 300, "top-k ties by id");
    // Nucleus over the full order: the dominant candidate alone covers top_p = 0.5.
    Sampler nucleus(SamplingConfig{1.0, 0, 0.5, 7});
    bool dominant = true;
    for (int draw = 0; draw < 50; ++draw)
        dominant = dominant && nucleus.sample({0.0F, 5.0F, 1.0F}) == 1;
    expect(dominant, "nucleus keeps the dominant candidate");
}

} // namespace

int main() {
    slicing();
    positions();
    canvas();
    conversation();
    generation();
    if (failures == 0)
        std::cout << "minicpmv4_7 host contract: all checks passed\n";
    return failures == 0 ? 0 : 1;
}

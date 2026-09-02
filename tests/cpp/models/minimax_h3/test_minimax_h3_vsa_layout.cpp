/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/vsa_attention.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using trtmc::minimax_h3::vsa::TileLayout;

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void check_layout_is_a_padded_permutation(const TileLayout& layout) {
    const auto& geometry = layout.geometry;
    require(layout.valid_sizes.size() == static_cast<std::size_t>(geometry.total_tiles),
            "VSA valid-size count does not match tile count");
    require(layout.tiled_to_packed.size() == static_cast<std::size_t>(geometry.padded_rows),
            "VSA row-map count does not match padded rows");
    require(std::accumulate(layout.valid_sizes.begin(), layout.valid_sizes.end(), 0) ==
                geometry.logical_rows,
            "VSA valid-size sum does not match logical rows");

    std::vector<int32_t> visits(static_cast<std::size_t>(geometry.logical_rows), 0);
    for (int32_t tile = 0; tile < geometry.total_tiles; ++tile) {
        const int32_t valid = layout.valid_sizes[static_cast<std::size_t>(tile)];
        require(valid > 0 && valid <= trtmc::minimax_h3::vsa::kTileTokens,
                "VSA tile has an invalid row count");
        for (int32_t row = 0; row < trtmc::minimax_h3::vsa::kTileTokens; ++row) {
            const int32_t packed = layout.tiled_to_packed[static_cast<std::size_t>(tile) *
                                                              trtmc::minimax_h3::vsa::kTileTokens +
                                                          row];
            if (row < valid) {
                require(packed >= 0 && packed < geometry.logical_rows,
                        "VSA valid tile row has an invalid packed index");
                ++visits[static_cast<std::size_t>(packed)];
            } else {
                require(packed == -1, "VSA padding row is not marked -1");
            }
        }
    }
    require(std::all_of(visits.begin(), visits.end(), [](int32_t count) { return count == 1; }),
            "VSA tile map is not a permutation of packed rows");
}

void test_qualified_geometries() {
    const auto five = trtmc::minimax_h3::vsa::make_tile_layout(124, 537);
    require(five.geometry.audio_tokens == 414, "124f VSA audio geometry differs");
    require(five.geometry.video_latent_frames == 37, "124f VSA latent-frame geometry differs");
    require(five.geometry.video_tokens == 37296, "124f VSA video-row geometry differs");
    require(five.geometry.prefix_tiles == 16, "124f VSA prefix tile count differs");
    require(five.geometry.video_tiles == 660, "124f VSA video tile count differs");
    require(five.geometry.total_tiles == 676, "124f VSA total tile count differs");
    require(five.geometry.top_video_tiles == 66, "124f VSA top-10% count differs");
    require(five.geometry.logical_rows == 38247, "124f VSA logical rows differ");
    require(five.geometry.padded_rows == 43264, "124f VSA padded rows differ");
    check_layout_is_a_padded_permutation(five);

    // Prefix segments never share a tile, and video edge tiles preserve the
    // exact clipped (time, height, width) raster order.
    require(five.valid_sizes[8] == 25, "text tail must stay in its own tile");
    require(five.valid_sizes[15] == 30, "audio tail must stay in its own tile");
    require(five.valid_sizes[16] == 64, "first video tile must be full");
    require(five.valid_sizes[26] == 32, "right-edge video tile must have 32 rows");
    require(five.valid_sizes.back() == 8,
            "124f final temporal/right-edge video tile must have 8 rows");

    const auto video_begin = five.geometry.text_tokens + five.geometry.audio_tokens;
    require(five.tiled_to_packed[16 * 64] == video_begin,
            "first tiled video row must map to raster origin");
    require(five.tiled_to_packed[16 * 64 + 4] == video_begin + 42,
            "video tile rows must use H/W raster order");
    require(five.tiled_to_packed[16 * 64 + 16] == video_begin + 24 * 42,
            "video tile time must be the outer raster dimension");

    const auto fifteen = trtmc::minimax_h3::vsa::make_tile_layout(345, 537);
    require(fifteen.geometry.audio_tokens == 1150, "345f VSA audio geometry differs");
    require(fifteen.geometry.video_latent_frames == 102, "345f VSA latent-frame geometry differs");
    require(fifteen.geometry.video_tokens == 102816, "345f VSA video-row geometry differs");
    require(fifteen.geometry.prefix_tiles == 27, "345f VSA prefix tile count differs");
    require(fifteen.geometry.video_tiles == 1716, "345f VSA video tile count differs");
    require(fifteen.geometry.total_tiles == 1743, "345f VSA total tile count differs");
    require(fifteen.geometry.top_video_tiles == 172, "345f VSA top-10% count differs");
    require(fifteen.geometry.logical_rows == 104503, "345f VSA logical rows differ");
    require(fifteen.geometry.padded_rows == 111552, "345f VSA padded rows differ");
    require(fifteen.valid_sizes.back() == 16,
            "345f final temporal/right-edge video tile must have 16 rows");
    check_layout_is_a_padded_permutation(fifteen);
}

void test_selector_and_attention_map_semantics() {
    constexpr int32_t heads = 2;
    constexpr int32_t prefix = 2;
    constexpr int32_t video = 3;
    constexpr int32_t total = prefix + video;
    constexpr int32_t top = 2;
    std::vector<float> scores(static_cast<std::size_t>(heads) * total * total, -1000.0F);
    for (int32_t head = 0; head < heads; ++head) {
        for (int32_t query = 0; query < total; ++query) {
            float* row = scores.data() + (static_cast<std::size_t>(head) * total + query) * total;
            row[2] = static_cast<float>(query + head * 10);
            row[3] = static_cast<float>(20 - query - head);
            row[4] = static_cast<float>(5 + 2 * query + head);
        }
    }
    const auto selected = trtmc::minimax_h3::vsa::select_video_topk_reference(
        scores.data(), heads, total, prefix, video, top);
    for (int32_t head = 0; head < heads; ++head) {
        for (int32_t query = 0; query < total; ++query) {
            const int32_t* row =
                selected.data() + (static_cast<std::size_t>(head) * total + query) * top;
            require(row[0] < row[1], "selected video indices must be compacted in map order");
            require(row[0] >= prefix && row[1] < total, "selector emitted a non-video tile");
        }
    }

    const auto dense = trtmc::minimax_h3::vsa::attended_key_tiles_reference(selected.data(), 1,
                                                                            prefix, video, top);
    require(dense == std::vector<int32_t>({0, 1, 2, 3, 4}), "prefix queries must remain dense");
    const int32_t* video_selection = selected.data() + 4 * top;
    const auto sparse = trtmc::minimax_h3::vsa::attended_key_tiles_reference(video_selection, 4,
                                                                             prefix, video, top);
    require(sparse.size() == 4 && sparse[0] == 0 && sparse[1] == 1 &&
                sparse[2] == video_selection[0] && sparse[3] == video_selection[1],
            "video queries must attend dense prefix plus selected video keys");
}

void test_rejected_geometry() {
    for (const int32_t frames : {123, 125, 344, 346}) {
        bool rejected = false;
        try {
            (void)trtmc::minimax_h3::vsa::make_geometry(frames, 537);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "VSA accepted an unqualified frame bucket");
    }
    for (const int32_t text : {0, 1025}) {
        bool rejected = false;
        try {
            (void)trtmc::minimax_h3::vsa::make_geometry(124, text);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "VSA accepted an out-of-profile text length");
    }
}

} // namespace

int main() {
    try {
        test_qualified_geometries();
        test_selector_and_attention_map_semantics();
        test_rejected_geometry();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

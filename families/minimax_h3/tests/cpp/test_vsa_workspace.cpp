/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/minimax_h3/kernels/vsa_workspace.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

void require(bool condition) {
    if (!condition)
        throw std::runtime_error("VSA workspace lifetime invariant failed");
}

void check_query_batch_layout(int heads, int tiles, int64_t capacity) {
    const auto a = trtmc::minimax_h3::make_vsa_query_batch_layout(heads, tiles, capacity);
    const int64_t b = a.query_tile_batch;
    const int64_t pool = int64_t(heads) * tiles * 128 * 4;
    const int64_t route = int64_t(heads) * tiles * tiles * 4;
    const int64_t counts = int64_t(heads) * tiles * 4;
    const int64_t rows = int64_t(tiles) * 128;
    const int64_t kv = rows * 128 * 4;
    const int64_t tile = 128 * 128 * 4;
    require(b >= 1 && b <= 32 && b <= tiles);
    require(a.fine_offset == pool && a.scaled_key_offset == pool);
    require(a.scaled_key_offset + kv <= a.value_offset);
    require(a.value_offset + kv <= a.scaled_query_offset);
    require(a.scaled_query_offset + b * tile <= a.score_offset);
    require(a.score_offset + b * 128 * rows * 4 <= a.fine_output_offset);
    require(a.fine_output_offset + b * tile <= a.key_mask_offset);
    require(a.key_mask_offset + b * rows <= a.index_offset);
    require(a.index_offset + route <= a.count_offset);
    require(a.count_offset + counts <= capacity);
    require(a.fine_offset + a.fine_bytes <= a.index_offset);
    for (auto offset : {a.scaled_key_offset, a.value_offset, a.scaled_query_offset,
                        a.score_offset, a.fine_output_offset, a.key_mask_offset, a.index_offset})
        require(offset % 256 == 0);
    // QK reads K/Q and writes scores; its pointer table occupies dead PV output.
    require(3 * b * int64_t(sizeof(void*)) <= b * tile);
    // PV reads V/probability and writes independent tile outputs. Its pointer
    // table occupies the completed softmax mask, including the B=1 duplicate.
    const int64_t products = std::max(int64_t(2), b);
    require(3 * products * int64_t(sizeof(void*)) <= b * rows);
    if (b == 1)
        require(a.scaled_query_offset + tile <= a.score_offset);
}

int main() {
    int checked = 0;
    for (int heads : {1, 2, 4, 8, 56, 128, 256}) {
        for (int tiles = 2; tiles <= 8192; tiles += 2) {
            const auto value = trtmc::minimax_h3::make_vsa_workspace_layout(heads, tiles);
            require(value.fine_offset % 256 == 0);
            require(value.index_offset >= value.fine_offset + value.fine_bytes);
            require(value.count_offset >= value.index_offset);
            require(value.bytes > value.count_offset);
            const auto legacy = trtmc::minimax_h3::make_vsa_workspace_layout(heads, tiles, false);
            require(value.bytes >= legacy.bytes);
            require(value.scaled_key_offset >= value.fine_offset);
            require(value.value_offset > value.scaled_key_offset);
            require(value.scaled_query_offset > value.value_offset);
            require(value.score_offset > value.scaled_query_offset);
            require(value.fine_output_offset > value.score_offset);
            require(value.key_mask_offset > value.fine_output_offset);
            // Batched PV reuses dead Q for one output and fine for the other;
            // neither may overlap its live score/value input or pointer table.
            require(value.scaled_query_offset + 128 * 128 * 4 <= value.score_offset);
            require(value.score_offset + int64_t(tiles) * 128 * 128 * 4 <=
                    value.fine_output_offset);
            require(value.fine_output_offset + 128 * 128 * 4 <= value.key_mask_offset);
            require(value.key_mask_offset + int64_t(6 * sizeof(void*)) <=
                    value.fine_offset + value.fine_bytes);
            if (tiles > 2)
                require(value.bytes >=
                        trtmc::minimax_h3::make_vsa_workspace_layout(heads, tiles - 2).bytes);
            ++checked;
        }
    }
    // Simulate every writable fine byte, while preserving the independently
    // produced compression result and the routing indices/counts at both ends.
    for (const auto& profile :
         {std::pair{1, 2}, std::pair{2, 38}, std::pair{4, 14}, std::pair{56, 10}}) {
        const auto value =
            trtmc::minimax_h3::make_vsa_workspace_layout(profile.first, profile.second);
        std::vector<uint8_t> arena(value.bytes, 0x5a);
        std::fill(arena.begin() + value.fine_offset,
                  arena.begin() + value.fine_offset + value.fine_bytes, 0xa5);
        require(std::all_of(arena.begin(), arena.begin() + value.fine_offset,
                            [](uint8_t item) { return item == 0x5a; }));
        require(std::all_of(arena.begin() + value.index_offset, arena.end(),
                            [](uint8_t item) { return item == 0x5a; }));
    }
    for (int tiles : {708, 1026}) {
        const auto value = trtmc::minimax_h3::make_vsa_workspace_layout(56, tiles);
        require(value.bytes ==
                trtmc::minimax_h3::make_vsa_workspace_layout(56, tiles, false).bytes);
    }
    for (const auto& profile : {std::pair{0, 2}, std::pair{257, 2}, std::pair{1, 0},
                                std::pair{1, -1}, std::pair{1, 8194}}) {
        bool rejected = false;
        try {
            trtmc::minimax_h3::make_vsa_workspace_layout(profile.first, profile.second);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected);
    }
    int batch_checked = 0;
    for (int heads : {1, 2, 4, 8, 56, 128, 256}) {
        const auto profile = trtmc::minimax_h3::make_vsa_workspace_layout(heads, 8192);
        for (int tiles = 2; tiles <= 8192; tiles += 2) {
            const auto base = trtmc::minimax_h3::make_vsa_workspace_layout(heads, tiles);
            for (int64_t capacity : {base.bytes, base.bytes + 1, base.bytes + 255,
                                      base.bytes + 256, profile.bytes}) {
                check_query_batch_layout(heads, tiles, capacity);
                ++batch_checked;
            }
            bool rejected = false;
            try { trtmc::minimax_h3::make_vsa_query_batch_layout(heads, tiles, base.bytes - 1); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected);
        }
    }
    for (int tiles : {708, 1026}) {
        const auto profile = trtmc::minimax_h3::make_vsa_workspace_layout(56, 2048);
        const auto value = trtmc::minimax_h3::make_vsa_query_batch_layout(56, tiles, profile.bytes);
        require(value.query_tile_batch == (tiles == 708 ? 32 : 24));
        require(value.bytes == 2055667712);
    }
    for (const auto& profile : {std::pair{1, 2}, std::pair{2, 38}, std::pair{56, 10}}) {
        const auto base = trtmc::minimax_h3::make_vsa_workspace_layout(profile.first, profile.second);
        const auto value = trtmc::minimax_h3::make_vsa_query_batch_layout(
            profile.first, profile.second, base.bytes * 4);
        require(value.query_tile_batch > 1);
        std::vector<uint8_t> memory(value.bytes, 0x5a);
        std::fill(memory.begin() + value.fine_offset,
                  memory.begin() + value.fine_offset + value.fine_bytes, 0xa5);
        require(std::all_of(memory.begin(), memory.begin() + value.fine_offset,
                            [](uint8_t v) { return v == 0x5a; }));
        require(std::all_of(memory.begin() + value.index_offset, memory.end(),
                            [](uint8_t v) { return v == 0x5a; }));
    }
    std::cout << "{\"geometry_profiles_checked\":" << checked
              << ",\"query_batch_capacity_profiles_checked\":" << batch_checked
              << ",\"query_batch_sentinel_checks\":3,\"retained_buffer_sentinel_checks\":4,"
                 "\"large_profiles_need_additional_workspace\":false,"
                 "\"GPU_execution_performed\":false}"
              << std::endl;
}

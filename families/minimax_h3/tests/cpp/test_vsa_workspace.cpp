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
    std::cout << "{\"geometry_profiles_checked\":" << checked
              << ",\"retained_buffer_sentinel_checks\":4,"
                 "\"large_profiles_need_additional_workspace\":false,"
                 "\"GPU_execution_performed\":false}"
              << std::endl;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <stdexcept>

namespace trtmc::minimax_h3 {

struct VsaWorkspaceLayout {
    int64_t fine_offset;
    int64_t fine_bytes;
    int64_t index_offset;
    int64_t count_offset;
    int64_t bytes;
    int64_t scaled_key_offset;
    int64_t value_offset;
    int64_t scaled_query_offset;
    int64_t score_offset;
    int64_t fine_output_offset;
    int64_t key_mask_offset;
};

inline VsaWorkspaceLayout make_vsa_workspace_layout(int heads, int tiles,
                                                    bool reserve_fp32_fine = true) {
    if (heads < 1 || heads > 256 || tiles < 1 || tiles > 8192)
        throw std::invalid_argument("Invalid VSA workspace head/tile bounds");
    const int64_t rows = static_cast<int64_t>(tiles) * 128;
    const int64_t pool = static_cast<int64_t>(heads) * tiles * 128 * 4;
    const int64_t route = static_cast<int64_t>(heads) * tiles * tiles * 4;
    const int64_t counts = static_cast<int64_t>(heads) * tiles * 4;
    const int64_t kv = rows * 128 * 4;
    const int64_t query = 128 * 128 * 4;
    const int64_t scores = 128 * rows * 4;
    VsaWorkspaceLayout result{};
    // Compression completes in the first pool before fine attention. The two
    // other pools and routing scores are then dead; indices/counts stay live.
    result.fine_offset = pool;
    result.scaled_key_offset = pool;
    result.value_offset = result.scaled_key_offset + kv;
    result.scaled_query_offset = result.value_offset + kv;
    result.score_offset = result.scaled_query_offset + query;
    result.fine_output_offset = result.score_offset + scores;
    result.key_mask_offset = result.fine_output_offset + query;
    result.fine_bytes = result.key_mask_offset + rows - result.fine_offset;
    const int64_t dead_pool_and_scores = 2 * pool + route;
    const int64_t arena = reserve_fp32_fine && result.fine_bytes > dead_pool_and_scores
                              ? result.fine_bytes
                              : dead_pool_and_scores;
    result.index_offset = pool + arena;
    result.count_offset = result.index_offset + route;
    result.bytes = result.count_offset + counts;
    return result;
}

} // namespace trtmc::minimax_h3

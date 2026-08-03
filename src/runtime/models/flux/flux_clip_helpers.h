/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::diffusion::flux_clip {

inline std::vector<int32_t> pad_and_truncate_ids(const std::vector<int32_t>& input_ids,
                                                 std::size_t sequence_length, int32_t pad_token_id,
                                                 int32_t eos_token_id) {
    std::vector<int32_t> padded(sequence_length, std::max(pad_token_id, 0));
    const auto copy_length = std::min(sequence_length, input_ids.size());
    std::copy_n(input_ids.begin(), copy_length, padded.begin());
    if (input_ids.size() > sequence_length && eos_token_id >= 0 && !padded.empty()) {
        padded.back() = eos_token_id;
    }
    return padded;
}

} // namespace trtmc::diffusion::flux_clip

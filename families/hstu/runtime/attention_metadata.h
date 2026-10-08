/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace trtmc::hstu {

// CPU-only extension of INT32[5,B+1,8]. The five original planes keep their
// offsets. In the targets plane, [B+1,2*B+1) holds the last history-page lengths.
// This does not allocate pages or change any existing offset, target or page ID.
inline void fill_attention_page_lengths(std::int32_t* metadata, std::size_t batch,
                                        std::size_t elements) {
    if (!metadata || batch == 0 || batch > std::numeric_limits<std::size_t>::max() / 40 - 1 ||
        elements != 40 * (batch + 1))
        throw std::invalid_argument("hstu attention metadata has an invalid extent");
    const auto length = batch + 1;
    const auto stride = 8 * length;
    const auto* keys = metadata + stride;
    const auto* targets = metadata + 2 * stride;
    // Validate every row before changing the metadata, including on failure.
    for (std::size_t sample = 0; sample < batch; ++sample) {
        const auto history =
            static_cast<std::int64_t>(keys[sample + 1]) - keys[sample] - targets[sample];
        if (targets[sample] < 0 || history < 0)
            throw std::invalid_argument(
                "hstu attention metadata has invalid history/target extents");
    }
    auto* page_lengths = metadata + 2 * stride + length;
    for (std::size_t sample = 0; sample < batch; ++sample) {
        const auto history =
            static_cast<std::int64_t>(keys[sample + 1]) - keys[sample] - targets[sample];
        const auto remainder = static_cast<std::int32_t>(history % 128);
        // Empty history also uses 128: the original kernel subtracts
        // (128 - length) from target columns. No history pages are implied.
        page_lengths[sample] = remainder == 0 ? 128 : remainder;
    }
}

} // namespace trtmc::hstu

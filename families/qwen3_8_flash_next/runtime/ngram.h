/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace trtmc::qwen3_8_flash_next {

// The published policy has eight bigram heads followed by eight trigram heads.
// The hash constants and each head's row range belong to the checkpoint.
struct NgramLayout {
    std::array<std::uint64_t, 3> multipliers;
    std::array<std::int64_t, 16> sizes;
    std::array<std::int64_t, 16> offsets;
    std::int32_t eos;
    std::int32_t vocabulary_size;
};

struct NgramChunk {
    std::vector<std::int64_t> rows;      // [token, head], in checkpoint table coordinates
    std::array<std::int32_t, 2> history; // oldest first
};

class NgramLookup {
  public:
    NgramLookup(NgramLayout layout, std::int64_t table_rows);
    std::array<std::int32_t, 2> initial_history() const;
    // Each request owns its history. Return the updated state so the caller need
    // not advance a request if gathering rows or executing the engine fails.
    NgramChunk lookup(const std::vector<std::int32_t>& tokens,
                      std::array<std::int32_t, 2> history) const;

  private:
    NgramLayout layout_;
};

} // namespace trtmc::qwen3_8_flash_next

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_8_flash_next/runtime/ngram.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc::qwen3_8_flash_next {
namespace {

// Torch hashes in signed int64, with wrapping multiplication and nonnegative
// remainder. Keep the bit pattern unsigned, including the INT64_MIN case.
std::uint64_t table_remainder(std::uint64_t bits, std::uint64_t size) {
    if (bits >> 63U) {
        const auto magnitude_remainder = (std::uint64_t{0} - bits) % size;
        return magnitude_remainder == 0 ? 0 : size - magnitude_remainder;
    }
    return bits % size;
}

} // namespace

NgramLookup::NgramLookup(NgramLayout layout, std::int64_t table_rows) : layout_(std::move(layout)) {
    if (layout_.eos < 0 || layout_.eos >= layout_.vocabulary_size || table_rows <= 0)
        throw std::invalid_argument("Invalid Flash-Next vocabulary or n-gram table extent");
    std::int64_t previous_end = 0;
    for (std::size_t h = 0; h < layout_.sizes.size(); ++h) {
        const auto size = layout_.sizes[h];
        const auto offset = layout_.offsets[h];
        if (size <= 0 || offset < previous_end || offset > table_rows - size)
            throw std::invalid_argument(
                "Flash-Next n-gram head ranges overlap or exceed the table");
        previous_end = offset + size;
    }
}

std::array<std::int32_t, 2> NgramLookup::initial_history() const {
    return {layout_.eos, layout_.eos};
}

NgramChunk NgramLookup::lookup(const std::vector<std::int32_t>& tokens,
                               std::array<std::int32_t, 2> history) const {
    const auto valid_id = [this](std::int32_t id) {
        return id >= 0 && id < layout_.vocabulary_size;
    };
    if (!valid_id(history[0]) || !valid_id(history[1]) || tokens.empty())
        throw std::invalid_argument(
            "Flash-Next needs a nonempty token chunk and valid token history");
    NgramChunk chunk{{}, history};
    if (tokens.size() > chunk.rows.max_size() / layout_.sizes.size())
        throw std::length_error("Flash-Next n-gram row count exceeds addressable memory");
    chunk.rows.reserve(tokens.size() * layout_.sizes.size());
    for (const auto token : tokens) {
        if (!valid_id(token))
            throw std::invalid_argument("Token outside the Flash-Next vocabulary");
        const auto prev = chunk.history[1];
        // Tokens before the most recent EOS cannot contribute to the current segment.
        const auto before_prev = prev == layout_.eos ? layout_.eos : chunk.history[0];
        const auto bigram = static_cast<std::uint64_t>(token) * layout_.multipliers[0] ^
                            static_cast<std::uint64_t>(prev) * layout_.multipliers[1];
        const auto trigram =
            bigram ^ static_cast<std::uint64_t>(before_prev) * layout_.multipliers[2];
        for (std::size_t h = 0; h < layout_.sizes.size(); ++h) {
            const auto row = table_remainder(h < 8 ? bigram : trigram,
                                             static_cast<std::uint64_t>(layout_.sizes[h]));
            chunk.rows.push_back(layout_.offsets[h] + static_cast<std::int64_t>(row));
        }
        chunk.history = {prev, token};
    }
    return chunk;
}

} // namespace trtmc::qwen3_8_flash_next

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "ple_rows.h"

#include <istream>
#include <limits>
#include <stdexcept>

namespace trtmc::qwen3_8_flash_next {

std::vector<std::uint8_t> read_ple_rows(std::istream& source, std::uint64_t section_start,
                                        std::uint64_t section_bytes,
                                        const std::vector<std::int64_t>& row_ids) {
    constexpr std::size_t width = 160; // 2,560 channels split over 16 heads
    constexpr std::uint64_t max_offset = std::numeric_limits<std::streamoff>::max();
    if (!section_bytes || section_bytes % width || section_start > max_offset ||
        section_bytes > max_offset - section_start)
        throw std::invalid_argument("Invalid Flash-Next FP8 table extent");
    if (row_ids.size() % 16 || row_ids.size() > std::vector<std::uint8_t>().max_size() / width)
        throw std::invalid_argument("Invalid Flash-Next FP8 head-row count");
    const auto table_rows = section_bytes / width;
    for (const auto id : row_ids)
        if (id < 0 || static_cast<std::uint64_t>(id) >= table_rows)
            throw std::out_of_range("Flash-Next FP8 row outside table");
    std::vector<std::uint8_t> result(row_ids.size() * width);
    for (std::size_t i = 0; i < row_ids.size(); ++i) {
        const auto pos = section_start + static_cast<std::uint64_t>(row_ids[i]) * width;
        source.seekg(static_cast<std::streamoff>(pos));
        if (!source.read(reinterpret_cast<char*>(result.data() + i * width), width))
            throw std::runtime_error("Failed to read Flash-Next FP8 table row");
    }
    return result;
}
} // namespace trtmc::qwen3_8_flash_next

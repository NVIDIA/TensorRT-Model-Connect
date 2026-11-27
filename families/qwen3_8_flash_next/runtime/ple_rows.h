/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <iosfwd>
#include <vector>

namespace trtmc::qwen3_8_flash_next {

// The table section concatenates the released FP8 shards in numerical order.
// One stream per caller; row IDs are token-major, then head-major.
std::vector<std::uint8_t> read_ple_rows(std::istream& source, std::uint64_t section_start,
                                        std::uint64_t section_bytes,
                                        const std::vector<std::int64_t>& row_ids);

} // namespace trtmc::qwen3_8_flash_next

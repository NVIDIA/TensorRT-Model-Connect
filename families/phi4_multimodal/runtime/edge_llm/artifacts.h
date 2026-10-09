/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc::phi4_multimodal::edge_llm {

/// Validate both projected GN tensors before upstream can substitute missing vectors with zeros.
inline void validate_separator_bytes(const std::vector<char>& bytes) {
    if (bytes.size() < 8)
        throw std::runtime_error("Truncated Phi4 projected separator file");
    std::uint64_t header_size = 0;
    for (unsigned int i = 0; i < 8; ++i)
        header_size |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i])) << (8 * i);
    if (header_size == 0 || header_size > 256 * 1024 || header_size > bytes.size() - 8)
        throw std::runtime_error("Invalid Phi4 projected separator header");
    const auto header = nlohmann::json::parse(bytes.begin() + 8, bytes.begin() + 8 + header_size);
    const auto data_size = bytes.size() - 8 - header_size;
    std::vector<std::pair<std::int64_t, std::int64_t>> regions;
    for (const auto* key : {"glb_GN", "sub_GN"}) {
        if (!header.contains(key))
            throw std::runtime_error("Phi4 projected separator is missing");
        const auto& value = header.at(key);
        const auto& offsets = value.at("data_offsets");
        if (value.at("dtype") != "F16" || value.at("shape") != nlohmann::json::array({3072}) ||
            !offsets.is_array() || offsets.size() != 2 || !offsets[0].is_number_integer() ||
            !offsets[1].is_number_integer())
            throw std::runtime_error("Phi4 projected separator must be FP16[3072]");
        const auto start = offsets[0].get<std::int64_t>();
        const auto end = offsets[1].get<std::int64_t>();
        if (start < 0 || end < start || end - start != 3072 * 2 ||
            static_cast<std::uint64_t>(end) > data_size)
            throw std::runtime_error("Invalid Phi4 projected separator tensor bounds");
        regions.emplace_back(start, end);
    }
    if (regions[0].first < regions[1].second && regions[1].first < regions[0].second)
        throw std::runtime_error("Overlapping Phi4 projected separator tensors");
}

} // namespace trtmc::phi4_multimodal::edge_llm

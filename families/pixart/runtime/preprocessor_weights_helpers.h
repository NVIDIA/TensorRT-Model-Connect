/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {
namespace pixart_preprocessor_weights {

inline nlohmann::json extract_preprocessor_index(const std::vector<char>& data, const char*& blob,
                                                 std::size_t& blob_size) {
    if (data.size() < sizeof(uint32_t))
        throw std::runtime_error("PixArt preprocessor.weights is too small");
    uint32_t index_size = 0;
    std::memcpy(&index_size, data.data(), sizeof(index_size));
    if (index_size > data.size() - sizeof(index_size))
        throw std::runtime_error("PixArt preprocessor.weights index exceeds the section");
    const auto index_begin = data.begin() + static_cast<std::ptrdiff_t>(sizeof(index_size));
    const auto index_end = index_begin + static_cast<std::ptrdiff_t>(index_size);
    auto index = nlohmann::json::parse(index_begin, index_end);
    if (!index.is_object())
        throw std::runtime_error("PixArt preprocessor.weights index must be a JSON object");
    blob = data.data() + sizeof(index_size) + index_size;
    blob_size = data.size() - sizeof(index_size) - index_size;
    return index;
}

inline void load_preprocessor_floats(const nlohmann::json& index, const char* blob,
                                     std::size_t blob_size, const std::string& key,
                                     std::vector<float>& output) {
    const auto& entry = index.at(key);
    if (!entry.is_object())
        throw std::runtime_error("PixArt preprocessor index entry must be an object: " + key);
    const auto& offset_value = entry.at("offset");
    if (!offset_value.is_number_integer() && !offset_value.is_number_unsigned())
        throw std::runtime_error("PixArt preprocessor offset must be an integer: " + key);
    const auto signed_offset = offset_value.get<std::int64_t>();
    if (signed_offset < 0)
        throw std::runtime_error("PixArt preprocessor offset must be nonnegative: " + key);
    const auto offset = static_cast<std::size_t>(signed_offset);
    const auto& shape = entry.at("shape");
    if (!shape.is_array())
        throw std::runtime_error("PixArt preprocessor shape must be an array: " + key);
    std::size_t count = 1;
    for (const auto& dimension : shape) {
        if (!dimension.is_number_integer() && !dimension.is_number_unsigned())
            throw std::runtime_error("PixArt preprocessor shape must contain integers: " + key);
        const auto signed_dimension = dimension.get<std::int64_t>();
        if (signed_dimension <= 0)
            throw std::runtime_error("PixArt preprocessor dimensions must be positive: " + key);
        const auto size = static_cast<std::size_t>(signed_dimension);
        if (count > std::numeric_limits<std::size_t>::max() / size)
            throw std::runtime_error("PixArt preprocessor shape overflows: " + key);
        count *= size;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::runtime_error("PixArt preprocessor byte count overflows: " + key);
    const std::size_t byte_count = count * sizeof(float);
    if (offset > blob_size || byte_count > blob_size - offset)
        throw std::runtime_error("PixArt preprocessor entry exceeds the blob: " + key);
    output.resize(count);
    std::memcpy(output.data(), blob + offset, byte_count);
}

} // namespace pixart_preprocessor_weights
} // namespace trtmc

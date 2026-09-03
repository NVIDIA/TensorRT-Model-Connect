/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/boltz2/prepared_request.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace trtmc::boltz2 {
namespace {

constexpr std::array<char, 4> kMagic{'B', '2', 'R', 'Q'};
constexpr uint32_t kVersion = 2;
constexpr std::size_t kHeaderSize = 4U + sizeof(uint32_t) + 4U * sizeof(uint64_t);

uint32_t readU32(const std::byte*& cursor) {
    uint32_t result = 0;
    for (std::size_t index = 0; index < sizeof(result); ++index)
        result |= static_cast<uint32_t>(std::to_integer<uint8_t>(cursor[index])) << (8U * index);
    cursor += sizeof(result);
    return result;
}

uint64_t readU64(const std::byte*& cursor) {
    uint64_t result = 0;
    for (std::size_t index = 0; index < sizeof(result); ++index)
        result |= static_cast<uint64_t>(std::to_integer<uint8_t>(cursor[index])) << (8U * index);
    cursor += sizeof(result);
    return result;
}

std::size_t checkedSize(uint64_t value, const char* label) {
    if (value > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument(std::string("Boltz-2 prepared-request ") + label +
                                    " size overflows this host");
    return static_cast<std::size_t>(value);
}

void requireNonEmptySections(const std::array<std::size_t, 4>& section_sizes) {
    for (const std::size_t section_size : section_sizes) {
        if (section_size == 0)
            throw std::invalid_argument("Boltz-2 prepared request contains an empty section");
    }
}

void consumeSection(std::size_t& remaining, std::size_t section_size) {
    if (section_size > remaining)
        throw std::invalid_argument("Boltz-2 prepared-request section sizes are inconsistent");
    remaining -= section_size;
}

void validateSectionSizes(std::size_t payload_size, std::size_t request_size,
                          std::size_t feature_size, std::size_t random_size,
                          std::size_t metadata_size) {
    std::size_t remaining = payload_size - kHeaderSize;
    consumeSection(remaining, request_size);
    consumeSection(remaining, feature_size);
    consumeSection(remaining, random_size);
    if (metadata_size != remaining)
        throw std::invalid_argument("Boltz-2 prepared-request section sizes are inconsistent");
}

} // namespace

bool PreparedRequest::isPrepared(const void* data, std::size_t size) {
    return data != nullptr && size >= kMagic.size() && std::memcmp(data, kMagic.data(), 4) == 0;
}

PreparedRequest PreparedRequest::parse(const void* data, std::size_t size) {
    if (data == nullptr)
        throw std::invalid_argument("truncated Boltz-2 prepared-request header");
    if (size < kHeaderSize)
        throw std::invalid_argument("truncated Boltz-2 prepared-request header");
    auto* cursor = static_cast<const std::byte*>(data);
    if (std::memcmp(cursor, kMagic.data(), kMagic.size()) != 0)
        throw std::invalid_argument("invalid Boltz-2 prepared-request magic");
    cursor += kMagic.size();
    if (readU32(cursor) != kVersion)
        throw std::invalid_argument("unsupported Boltz-2 prepared-request version");
    const auto request_size = checkedSize(readU64(cursor), "document");
    const auto feature_size = checkedSize(readU64(cursor), "feature");
    const auto random_size = checkedSize(readU64(cursor), "random");
    const auto metadata_size = checkedSize(readU64(cursor), "metadata");
    requireNonEmptySections({request_size, feature_size, random_size, metadata_size});
    validateSectionSizes(size, request_size, feature_size, random_size, metadata_size);

    PreparedRequest result;
    result.request.assign(reinterpret_cast<const char*>(cursor), request_size);
    cursor += request_size;
    result.features = FeatureBundle::parse(cursor, feature_size);
    cursor += feature_size;
    result.random_samples = RandomSamples::parse(cursor, random_size);
    cursor += random_size;
    result.structure_metadata_json.assign(reinterpret_cast<const char*>(cursor), metadata_size);
    return result;
}

} // namespace trtmc::boltz2

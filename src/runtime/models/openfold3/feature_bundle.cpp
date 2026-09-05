/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/openfold3/feature_bundle.h"

#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace trtmc::openfold3 {
namespace {

constexpr std::array<char, 4> kMagic{'O', 'F', '3', 'F'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kExpectedTensorCount = 20;

class Reader {
  public:
    Reader(const void* data, std::size_t size)
        : cursor_(static_cast<const std::byte*>(data)), remaining_(size) {
        if (data == nullptr && size != 0)
            throw std::invalid_argument("OpenFold3 feature section has null storage");
    }

    template <typename T>
    T littleEndian() {
        static_assert(std::is_unsigned_v<T>);
        requireBytes(sizeof(T));
        T value = 0;
        for (std::size_t index = 0; index < sizeof(T); ++index)
            value |= static_cast<T>(std::to_integer<uint8_t>(cursor_[index])) << (index * 8U);
        cursor_ += sizeof(T);
        remaining_ -= sizeof(T);
        return value;
    }

    std::string string(std::size_t size) {
        requireBytes(size);
        std::string value(reinterpret_cast<const char*>(cursor_), size);
        cursor_ += size;
        remaining_ -= size;
        return value;
    }

    std::vector<std::byte> bytes(std::size_t size) {
        requireBytes(size);
        std::vector<std::byte> value(cursor_, cursor_ + size);
        cursor_ += size;
        remaining_ -= size;
        return value;
    }

    std::size_t remaining() const { return remaining_; }

  private:
    void requireBytes(std::size_t size) const {
        if (size > remaining_)
            throw std::invalid_argument("truncated OpenFold3 feature section");
    }

    const std::byte* cursor_;
    std::size_t remaining_;
};

DType decodeDtype(uint8_t code) {
    switch (code) {
    case 1:
        return DType::kFloat32;
    case 2:
        return DType::kInt32;
    case 3:
        return DType::kInt8;
    default:
        throw std::invalid_argument("unsupported OpenFold3 feature dtype code");
    }
}

std::size_t checkedElementCount(const std::vector<int64_t>& shape) {
    std::size_t count = 1;
    for (const int64_t dimension : shape) {
        if (dimension <= 0)
            throw std::invalid_argument("OpenFold3 feature dimensions must be positive");
        const auto value = static_cast<std::size_t>(dimension);
        if (count > std::numeric_limits<std::size_t>::max() / value)
            throw std::invalid_argument("OpenFold3 feature shape overflows host size");
        count *= value;
    }
    return count;
}

struct ParsedFeature {
    std::string name;
    FeatureTensor tensor;
};

ParsedFeature readFeature(Reader& source) {
    auto name = source.string(source.littleEndian<uint16_t>());
    if (name.empty())
        throw std::invalid_argument("OpenFold3 feature name must not be empty");
    const auto dtype = decodeDtype(source.littleEndian<uint8_t>());
    const auto rank = source.littleEndian<uint8_t>();
    if (rank == 0 || rank > 8)
        throw std::invalid_argument("OpenFold3 feature rank must be in [1, 8]");
    std::vector<int64_t> shape;
    shape.reserve(rank);
    for (uint8_t axis = 0; axis < rank; ++axis)
        shape.push_back(source.littleEndian<uint32_t>());
    const auto payload_size_u64 = source.littleEndian<uint64_t>();
    if (payload_size_u64 > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("OpenFold3 feature payload is too large");
    const auto payload_size = static_cast<std::size_t>(payload_size_u64);
    const auto element_count = checkedElementCount(shape);
    if (element_count > std::numeric_limits<std::size_t>::max() / dtype_size(dtype))
        throw std::invalid_argument("OpenFold3 feature payload size overflows host size");
    if (payload_size != element_count * dtype_size(dtype))
        throw std::invalid_argument("OpenFold3 feature payload size does not match its shape");
    return {std::move(name), {std::move(shape), dtype, source.bytes(payload_size)}};
}

} // namespace

FeatureBundle FeatureBundle::parse(const void* data, std::size_t size) {
    Reader source(data, size);
    if (source.string(kMagic.size()) != std::string(kMagic.data(), kMagic.size()))
        throw std::invalid_argument("invalid OpenFold3 feature section magic");
    if (source.littleEndian<uint32_t>() != kVersion)
        throw std::invalid_argument("unsupported OpenFold3 feature section version");
    const uint32_t count = source.littleEndian<uint32_t>();
    if (count != kExpectedTensorCount)
        throw std::invalid_argument("OpenFold3 feature section must contain exactly 20 tensors");

    FeatureBundle result;
    result.tensors_.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
        auto parsed = readFeature(source);
        if (!result.tensors_.emplace(parsed.name, std::move(parsed.tensor)).second)
            throw std::invalid_argument("duplicate OpenFold3 feature tensor: " + parsed.name);
    }
    if (source.remaining() != 0)
        throw std::invalid_argument("OpenFold3 feature section has trailing bytes");
    return result;
}

const FeatureTensor& FeatureBundle::require(std::string_view name) const {
    const auto found = tensors_.find(std::string(name));
    if (found == tensors_.end())
        throw std::invalid_argument("OpenFold3 feature section is missing tensor: " +
                                    std::string(name));
    return found->second;
}

} // namespace trtmc::openfold3

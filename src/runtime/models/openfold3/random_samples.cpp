/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/openfold3/random_samples.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace trtmc::openfold3 {
namespace {

uint32_t readU32(const std::byte*& cursor, std::size_t& remaining) {
    if (remaining < sizeof(uint32_t))
        throw std::invalid_argument("truncated OpenFold3 random-sample section");
    uint32_t result = 0;
    for (std::size_t index = 0; index < sizeof(uint32_t); ++index)
        result |= static_cast<uint32_t>(std::to_integer<uint8_t>(cursor[index])) << (8U * index);
    cursor += sizeof(uint32_t);
    remaining -= sizeof(uint32_t);
    return result;
}

template <typename T>
void copyArray(const std::byte*& cursor, std::size_t& remaining, std::vector<T>& output,
               std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
        throw std::invalid_argument("OpenFold3 random-sample array size overflows");
    const std::size_t bytes = count * sizeof(T);
    if (bytes > remaining)
        throw std::invalid_argument("truncated OpenFold3 random-sample section");
    output.resize(count);
    std::memcpy(output.data(), cursor, bytes);
    cursor += bytes;
    remaining -= bytes;
}

} // namespace

RandomSamples RandomSamples::parse(const void* data, std::size_t size) {
    if (data == nullptr || size < 20)
        throw std::invalid_argument("truncated OpenFold3 random-sample section");
    auto* cursor = static_cast<const std::byte*>(data);
    std::size_t remaining = size;
    if (std::memcmp(cursor, "OF3R", 4) != 0)
        throw std::invalid_argument("invalid OpenFold3 random-sample magic");
    cursor += 4;
    remaining -= 4;
    if (readU32(cursor, remaining) != 1)
        throw std::invalid_argument("unsupported OpenFold3 random-sample version");
    RandomSamples result;
    result.seed = static_cast<int32_t>(readU32(cursor, remaining));
    result.sampling_steps = static_cast<int32_t>(readU32(cursor, remaining));
    result.padded_atom_count = static_cast<int32_t>(readU32(cursor, remaining));
    if (result.seed != 42 || result.sampling_steps != 200 || result.padded_atom_count <= 0)
        throw std::invalid_argument("OpenFold3 random samples differ from the pinned profile");
    const std::size_t atoms = static_cast<std::size_t>(result.padded_atom_count);
    const std::size_t steps = static_cast<std::size_t>(result.sampling_steps);
    copyArray(cursor, remaining, result.initial, atoms * 3U);
    copyArray(cursor, remaining, result.rotations, steps);
    copyArray(cursor, remaining, result.translations, steps);
    copyArray(cursor, remaining, result.noise, steps * atoms * 3U);
    if (remaining != 0)
        throw std::invalid_argument("OpenFold3 random-sample section has trailing bytes");
    return result;
}

} // namespace trtmc::openfold3

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/boltz2/runtime/random_samples.h"

#include <cstring>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace trtmc::boltz2 {
namespace {

uint32_t readU32(const std::byte*& cursor, std::size_t& remaining) {
    if (remaining < sizeof(uint32_t))
        throw std::invalid_argument("truncated Boltz-2 random-sample section");
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
        throw std::invalid_argument("Boltz-2 random-sample array size overflows");
    const std::size_t bytes = count * sizeof(T);
    if (bytes > remaining)
        throw std::invalid_argument("truncated Boltz-2 random-sample section");
    output.resize(count);
    if (bytes != 0)
        std::memcpy(output.data(), cursor, bytes);
    cursor += bytes;
    remaining -= bytes;
}

constexpr int32_t kMaxSamplingSteps = 1000;

void validateStructureHeader(const RandomSamples& samples) {
    if (samples.seed < 0 || samples.atom_count <= 0 || samples.atom_count > 928 ||
        samples.structure.sampling_steps < 10 ||
        samples.structure.sampling_steps > kMaxSamplingSteps ||
        samples.structure.sample_count < 1 || samples.structure.sample_count > 25) {
        throw std::invalid_argument(
            "Boltz-2 structure random samples are outside supported bounds");
    }
}

void validateAffinityHeader(const RandomSamples& samples) {
    if (samples.affinity.sampling_steps < 0 ||
        samples.affinity.sampling_steps > kMaxSamplingSteps || samples.affinity.sample_count < 0 ||
        samples.affinity.sample_count > 5 ||
        (samples.affinity.sample_count == 0) != (samples.affinity.sampling_steps == 0) ||
        (samples.affinity.sample_count > 0 && samples.affinity.sampling_steps < 10)) {
        throw std::invalid_argument("Boltz-2 affinity random samples are outside supported bounds");
    }
}

void validateHeader(const RandomSamples& samples) {
    validateStructureHeader(samples);
    validateAffinityHeader(samples);
}

std::size_t checkedProduct(std::initializer_list<std::size_t> values) {
    std::size_t result = 1;
    for (const std::size_t value : values) {
        if (value != 0 && result > std::numeric_limits<std::size_t>::max() / value)
            throw std::invalid_argument("Boltz-2 random-sample array size overflows");
        result *= value;
    }
    return result;
}

void copyGroup(const std::byte*& cursor, std::size_t& remaining, RandomSampleGroup& group,
               std::size_t atoms) {
    const auto steps = static_cast<std::size_t>(group.sampling_steps);
    const auto samples = static_cast<std::size_t>(group.sample_count);
    copyArray(cursor, remaining, group.initial, checkedProduct({samples, atoms, 3U}));
    copyArray(cursor, remaining, group.rotations, checkedProduct({samples, steps}));
    copyArray(cursor, remaining, group.translations, checkedProduct({samples, steps}));
    copyArray(cursor, remaining, group.noise, checkedProduct({samples, steps, atoms, 3U}));
}

} // namespace

RandomSamples RandomSamples::parse(const void* data, std::size_t size) {
    if (data == nullptr || size < 32)
        throw std::invalid_argument("truncated Boltz-2 random-sample section");
    auto* cursor = static_cast<const std::byte*>(data);
    std::size_t remaining = size;
    if (std::memcmp(cursor, "B2RN", 4) != 0)
        throw std::invalid_argument("invalid Boltz-2 random-sample magic");
    cursor += 4;
    remaining -= 4;
    if (readU32(cursor, remaining) != 3)
        throw std::invalid_argument("unsupported Boltz-2 random-sample version");
    RandomSamples result;
    result.seed = static_cast<int32_t>(readU32(cursor, remaining));
    result.atom_count = static_cast<int32_t>(readU32(cursor, remaining));
    result.structure.sampling_steps = static_cast<int32_t>(readU32(cursor, remaining));
    result.structure.sample_count = static_cast<int32_t>(readU32(cursor, remaining));
    result.affinity.sampling_steps = static_cast<int32_t>(readU32(cursor, remaining));
    result.affinity.sample_count = static_cast<int32_t>(readU32(cursor, remaining));
    validateHeader(result);
    const std::size_t atoms = static_cast<std::size_t>(result.atom_count);
    copyGroup(cursor, remaining, result.structure, atoms);
    copyGroup(cursor, remaining, result.affinity, atoms);
    if (remaining != 0)
        throw std::invalid_argument("Boltz-2 random-sample section has trailing bytes");
    return result;
}

} // namespace trtmc::boltz2

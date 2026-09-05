/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/openfold3/feature_bundle.h"
#include "runtime/models/openfold3/random_samples.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
void appendInteger(std::vector<std::byte>& output, T value) {
    for (std::size_t byte = 0; byte < sizeof(T); ++byte)
        output.push_back(static_cast<std::byte>((value >> (byte * 8U)) & 0xffU));
}

void append(std::vector<std::byte>& output, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::byte*>(data);
    output.insert(output.end(), bytes, bytes + size);
}

std::vector<std::byte> featureSection() {
    std::vector<std::byte> result;
    append(result, "OF3F", 4);
    appendInteger<uint32_t>(result, 1);
    appendInteger<uint32_t>(result, 20);
    for (uint32_t index = 0; index < 20; ++index) {
        const std::string name = "tensor_" + std::to_string(index);
        appendInteger<uint16_t>(result, static_cast<uint16_t>(name.size()));
        append(result, name.data(), name.size());
        appendInteger<uint8_t>(result, 1);
        appendInteger<uint8_t>(result, 1);
        appendInteger<uint32_t>(result, 1);
        appendInteger<uint64_t>(result, sizeof(float));
        const float value = static_cast<float>(index);
        append(result, &value, sizeof(value));
    }
    return result;
}

std::vector<std::byte> randomSection(uint32_t atoms) {
    std::vector<std::byte> result;
    append(result, "OF3R", 4);
    appendInteger<uint32_t>(result, 1);
    appendInteger<uint32_t>(result, 42);
    appendInteger<uint32_t>(result, 200);
    appendInteger<uint32_t>(result, atoms);
    const std::size_t floats = static_cast<std::size_t>(atoms) * 3U + 200U * 9U + 200U * 3U +
                               200U * static_cast<std::size_t>(atoms) * 3U;
    result.resize(result.size() + floats * sizeof(float));
    return result;
}

} // namespace

int main() {
    try {
        auto feature_bytes = featureSection();
        const auto features =
            trtmc::openfold3::FeatureBundle::parse(feature_bytes.data(), feature_bytes.size());
        if (features.size() != 20 || features.require("tensor_0").dtype != trtmc::DType::kFloat32)
            throw std::runtime_error("feature contract");
        feature_bytes.pop_back();
        try {
            (void)trtmc::openfold3::FeatureBundle::parse(feature_bytes.data(),
                                                         feature_bytes.size());
            throw std::runtime_error("truncated feature section was accepted");
        } catch (const std::invalid_argument&) {
        }
        const auto random_bytes = randomSection(608);
        const auto random =
            trtmc::openfold3::RandomSamples::parse(random_bytes.data(), random_bytes.size());
        if (random.seed != 42 || random.sampling_steps != 200 || random.padded_atom_count != 608)
            throw std::runtime_error("random contract");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "OpenFold3 section test failed: " << error.what() << '\n';
        return 1;
    }
}

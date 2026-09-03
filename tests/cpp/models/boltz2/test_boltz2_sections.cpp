/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/boltz2/feature_bundle.h"
#include "runtime/models/boltz2/prepared_request.h"
#include "runtime/models/boltz2/random_samples.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <typename T>
void appendLittleEndian(std::vector<std::byte>& output, T value) {
    for (std::size_t index = 0; index < sizeof(T); ++index)
        output.push_back(static_cast<std::byte>((value >> (8U * index)) & 0xffU));
}

void appendBytes(std::vector<std::byte>& output, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::byte*>(data);
    output.insert(output.end(), bytes, bytes + size);
}

std::vector<std::byte> featureSection() {
    std::vector<std::byte> result;
    appendBytes(result, "B2FT", 4);
    appendLittleEndian<uint32_t>(result, 1);
    appendLittleEndian<uint32_t>(result, 31);
    for (uint32_t index = 0; index < 31; ++index) {
        const std::string name = "tensor_" + std::to_string(index);
        appendLittleEndian<uint16_t>(result, static_cast<uint16_t>(name.size()));
        appendBytes(result, name.data(), name.size());
        appendLittleEndian<uint8_t>(result, 1);
        appendLittleEndian<uint8_t>(result, 1);
        appendLittleEndian<uint32_t>(result, 1);
        appendLittleEndian<uint64_t>(result, sizeof(float));
        const float value = static_cast<float>(index);
        appendBytes(result, &value, sizeof(value));
    }
    return result;
}

std::vector<std::byte> randomSection(uint32_t atom_count = 928) {
    std::vector<std::byte> result;
    appendBytes(result, "B2RN", 4);
    appendLittleEndian<uint32_t>(result, 1);
    appendLittleEndian<uint32_t>(result, 42);
    appendLittleEndian<uint32_t>(result, 200);
    appendLittleEndian<uint32_t>(result, atom_count);
    const std::size_t floats = static_cast<std::size_t>(atom_count) * 3U + 200U * 9U + 200U * 3U +
                               200U * static_cast<std::size_t>(atom_count) * 3U;
    result.resize(result.size() + floats * sizeof(float));
    return result;
}

std::vector<std::byte> preparedRequestSection(const std::vector<std::byte>& features) {
    const std::string request = "version: 1\n";
    const std::string metadata = "{}";
    const auto random = randomSection(32);
    std::vector<std::byte> result;
    appendBytes(result, "B2RQ", 4);
    appendLittleEndian<uint32_t>(result, 2);
    appendLittleEndian<uint64_t>(result, request.size());
    appendLittleEndian<uint64_t>(result, features.size());
    appendLittleEndian<uint64_t>(result, random.size());
    appendLittleEndian<uint64_t>(result, metadata.size());
    appendBytes(result, request.data(), request.size());
    appendBytes(result, features.data(), features.size());
    appendBytes(result, random.data(), random.size());
    appendBytes(result, metadata.data(), metadata.size());
    return result;
}

} // namespace

int main() {
    try {
        auto features = featureSection();
        const auto parsed_features =
            trtmc::boltz2::FeatureBundle::parse(features.data(), features.size());
        check(parsed_features.size() == 31, "feature count");
        check(parsed_features.require("tensor_0").dtype == trtmc::DType::kFloat32, "feature dtype");
        features.pop_back();
        bool feature_threw = false;
        try {
            (void)trtmc::boltz2::FeatureBundle::parse(features.data(), features.size());
        } catch (const std::invalid_argument&) {
            feature_threw = true;
        }
        check(feature_threw, "truncated feature section fails closed");

        const auto prepared_bytes = preparedRequestSection(featureSection());
        check(trtmc::boltz2::PreparedRequest::isPrepared(prepared_bytes.data(),
                                                         prepared_bytes.size()),
              "prepared request magic");
        const auto prepared =
            trtmc::boltz2::PreparedRequest::parse(prepared_bytes.data(), prepared_bytes.size());
        check(prepared.request == "version: 1\n", "prepared request document");
        check(prepared.features.size() == 31, "prepared request features");
        check(prepared.random_samples.atom_count == 32, "prepared request random samples");
        check(prepared.structure_metadata_json == "{}", "prepared request metadata");
        auto truncated_prepared = prepared_bytes;
        truncated_prepared.pop_back();
        bool prepared_threw = false;
        try {
            (void)trtmc::boltz2::PreparedRequest::parse(truncated_prepared.data(),
                                                        truncated_prepared.size());
        } catch (const std::invalid_argument&) {
            prepared_threw = true;
        }
        check(prepared_threw, "truncated prepared request fails closed");

        auto random = randomSection();
        const auto parsed_random =
            trtmc::boltz2::RandomSamples::parse(random.data(), random.size());
        check(parsed_random.seed == 42 && parsed_random.sampling_steps == 200 &&
                  parsed_random.atom_count == 928,
              "random profile");
        const auto variable_random = randomSection(576);
        const auto parsed_variable =
            trtmc::boltz2::RandomSamples::parse(variable_random.data(), variable_random.size());
        check(parsed_variable.atom_count == 576, "variable-length random profile");
        random[8] = std::byte{41};
        bool random_threw = false;
        try {
            (void)trtmc::boltz2::RandomSamples::parse(random.data(), random.size());
        } catch (const std::invalid_argument&) {
            random_threw = true;
        }
        check(random_threw, "wrong random seed fails closed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Boltz-2 section test failed: " << error.what() << '\n';
        return 1;
    }
}

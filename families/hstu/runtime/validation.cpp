/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/validation.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace trtmc::hstu {

DType parse_dtype(const std::string& name) {
    if (name == "float32")
        return DType::kFloat32;
    if (name == "int32")
        return DType::kInt32;
    if (name == "int64")
        return DType::kInt64;
    throw std::invalid_argument("unsupported feature dtype: " + name);
}

const char* dtype_name(DType dtype) {
    if (dtype == DType::kFloat32)
        return "float32";
    if (dtype == DType::kInt32)
        return "int32";
    if (dtype == DType::kInt64)
        return "int64";
    throw std::invalid_argument("unsupported feature dtype");
}

std::size_t checked_bytes(const std::vector<std::int64_t>& shape, DType dtype) {
    std::size_t size = dtype_size(dtype);
    if (shape.empty() || size == 0)
        throw std::invalid_argument("empty feature shape");
    for (auto dim : shape) {
        if (dim <= 0 ||
            static_cast<std::uint64_t>(dim) > std::numeric_limits<std::size_t>::max() / size)
            throw std::invalid_argument("invalid or overflowing feature shape");
        size *= static_cast<std::size_t>(dim);
    }
    return size;
}

TensorMap validate_inputs(const std::vector<FeatureTensor>& inputs, const nlohmann::json& config) {
    if (inputs.size() != config.at("inputs").size())
        throw std::invalid_argument("wrong number of ranking inputs");
    std::unordered_map<std::string, const FeatureTensor*> by_name;
    for (const auto& input : inputs) {
        if (!by_name.emplace(input.name, &input).second)
            throw std::invalid_argument("duplicate ranking input: " + input.name);
    }
    TensorMap bound;
    std::int64_t batch = -1;
    for (const auto& spec : config.at("inputs")) {
        const auto name = spec.at("name").get<std::string>();
        const auto found = by_name.find(name);
        if (found == by_name.end())
            throw std::invalid_argument("missing input: " + name);
        const auto& input = *found->second;
        const auto expected = spec.at("shape").get<std::vector<std::int64_t>>();
        if (input.shape.size() != expected.size() || input.shape.empty())
            throw std::invalid_argument("wrong input rank: " + name);
        if (batch < 0)
            batch = input.shape.front();
        if (batch < config.value("min_batch_size", std::int64_t{1}) ||
            batch > config.at("max_batch_size").get<std::int64_t>() || input.shape.front() != batch)
            throw std::invalid_argument("invalid or inconsistent batch: " + name);
        for (std::size_t axis = 1; axis < expected.size(); ++axis)
            if (input.shape[axis] != expected[axis])
                throw std::invalid_argument("wrong input shape: " + name);
        if (input.dtype != parse_dtype(spec.at("dtype").get<std::string>()) ||
            input.bytes.size() != checked_bytes(input.shape, input.dtype))
            throw std::invalid_argument("wrong input dtype or byte size: " + name);
        // Family-specific bounds are carried with the exact source schema.
        if (spec.contains("column_vocab_sizes")) {
            const auto limits = spec.at("column_vocab_sizes").get<std::vector<std::int64_t>>();
            if (input.dtype != DType::kInt64 || input.shape.size() != 2 ||
                limits.size() != static_cast<std::size_t>(input.shape[1]))
                throw std::invalid_argument("invalid categorical bounds contract");
            for (std::size_t i = 0; i < input.bytes.size() / sizeof(std::int64_t); ++i) {
                std::int64_t value;
                std::memcpy(&value, input.bytes.data() + i * sizeof(value), sizeof(value));
                if (value < 0 || value >= limits[i % limits.size()])
                    throw std::invalid_argument("categorical ID out of range: " + name);
            }
        }
        if (spec.contains("integer_min")) {
            if (input.dtype != DType::kInt64 && input.dtype != DType::kInt32)
                throw std::invalid_argument("invalid integer bounds contract");
            for (std::size_t offset = 0; offset < input.bytes.size();
                 offset += dtype_size(input.dtype)) {
                std::int64_t value{};
                if (input.dtype == DType::kInt32) {
                    std::int32_t small;
                    std::memcpy(&small, input.bytes.data() + offset, sizeof(small));
                    value = small;
                } else
                    std::memcpy(&value, input.bytes.data() + offset, sizeof(value));
                if (value < spec.at("integer_min").get<std::int64_t>() ||
                    (spec.contains("integer_max") &&
                     value > spec.at("integer_max").get<std::int64_t>()))
                    throw std::invalid_argument("integer feature out of range: " + name);
            }
        }
        if (input.dtype == DType::kFloat32) {
            if (input.dtype != DType::kFloat32)
                throw std::invalid_argument("invalid floating bounds contract");
            for (std::size_t i = 0; i < input.bytes.size() / sizeof(float); ++i) {
                float value;
                std::memcpy(&value, input.bytes.data() + i * sizeof(value), sizeof(value));
                if (!std::isfinite(value) ||
                    (spec.contains("value_min") && (value < spec.at("value_min").get<float>() ||
                                                    value > spec.at("value_max").get<float>())))
                    throw std::invalid_argument("feature value out of range: " + name);
            }
        }
        bound.emplace(
            name, Tensor{const_cast<std::uint8_t*>(input.bytes.data()), input.shape, input.dtype});
    }
    return bound;
}

} // namespace trtmc::hstu

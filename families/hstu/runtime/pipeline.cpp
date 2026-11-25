/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/hstu/runtime/pipeline.h"

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

Pipeline::Pipeline(std::unique_ptr<ITrtModule> engine, nlohmann::json config)
    : engine_(std::move(engine)), config_(std::move(config)) {
    if (!engine_ || !engine_->ok())
        throw std::runtime_error("invalid ranking engine");
    if (config_.at("family") != "hstu")
        throw std::invalid_argument("wrong family runtime contract");
    if (engine_->input_info().size() != config_.at("inputs").size() ||
        engine_->output_info().size() != config_.at("outputs").size())
        throw std::invalid_argument("engine I/O count differs from family contract");
    for (const auto& spec : config_.at("inputs")) {
        const auto name = spec.at("name").get<std::string>();
        if (!engine_->has_input(name) ||
            engine_->tensor_dtype(name) != parse_dtype(spec.at("dtype").get<std::string>()))
            throw std::invalid_argument("engine input differs from family contract: " + name);
    }
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
        if (spec.contains("value_min")) {
            if (input.dtype != DType::kFloat32)
                throw std::invalid_argument("invalid floating bounds contract");
            for (std::size_t i = 0; i < input.bytes.size() / sizeof(float); ++i) {
                float value;
                std::memcpy(&value, input.bytes.data() + i * sizeof(value), sizeof(value));
                if (!std::isfinite(value) || value < spec.at("value_min").get<float>() ||
                    value > spec.at("value_max").get<float>())
                    throw std::invalid_argument("feature value out of range: " + name);
            }
        }
        bound.emplace(
            name, Tensor{const_cast<std::uint8_t*>(input.bytes.data()), input.shape, input.dtype});
    }
    return bound;
}

std::vector<FeatureTensor> Pipeline::score(const std::vector<FeatureTensor>& inputs) {
    const auto bound = validate_inputs(inputs, config_);
    const auto batch = inputs.front().shape.front();
    const auto outputs = engine_->forward(bound);
    std::vector<FeatureTensor> result;
    for (const auto& spec : config_.at("outputs")) {
        const auto name = spec.at("name").get<std::string>();
        const auto found = outputs.find(name);
        if (found == outputs.end())
            throw std::runtime_error("missing engine output: " + name);
        const auto& tensor = found->second;
        auto expected = spec.at("shape").get<std::vector<std::int64_t>>();
        expected.at(0) = batch;
        if (tensor.shape != expected ||
            tensor.dtype != parse_dtype(spec.at("dtype").get<std::string>()))
            throw std::runtime_error("output contract mismatch: " + name);
        FeatureTensor output{name, tensor.dtype, tensor.shape, {}};
        output.bytes.resize(checked_bytes(tensor.shape, tensor.dtype));
        if (!tensor.data)
            throw std::runtime_error("null engine output: " + name);
        std::memcpy(output.bytes.data(), tensor.data, output.bytes.size());
        result.push_back(std::move(output));
    }
    return result;
}
} // namespace trtmc::hstu

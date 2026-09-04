/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/patchtsmixer/runtime/pipeline.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::patchtsmixer {

namespace {

std::int32_t require_positive_int(const std::string& json, const char* key) {
    const std::string needle = std::string("\"") + key + "\"";
    const auto key_position = json.find(needle);
    if (key_position == std::string::npos ||
        json.find(needle, key_position + needle.size()) != std::string::npos) {
        throw std::runtime_error("PatchTSMixer runtime.json must contain one field: " +
                                 std::string(key));
    }
    const auto colon = json.find(':', key_position + needle.size());
    if (colon == std::string::npos)
        throw std::runtime_error("PatchTSMixer runtime.json field has no value: " +
                                 std::string(key));
    auto position = colon + 1;
    while (position < json.size() && std::isspace(static_cast<unsigned char>(json[position])))
        ++position;
    if (position == json.size() || !std::isdigit(static_cast<unsigned char>(json[position])))
        throw std::runtime_error("PatchTSMixer runtime.json field is not a positive integer: " +
                                 std::string(key));
    std::uint64_t value = 0;
    while (position < json.size() && std::isdigit(static_cast<unsigned char>(json[position]))) {
        value = value * 10 + static_cast<unsigned int>(json[position] - '0');
        if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
            throw std::runtime_error("PatchTSMixer runtime.json field is too large: " +
                                     std::string(key));
        ++position;
    }
    if (value == 0)
        throw std::runtime_error("PatchTSMixer runtime.json field must be positive: " +
                                 std::string(key));
    return static_cast<std::int32_t>(value);
}

} // namespace

RuntimeConfig parse_runtime_config(const std::string& json) {
    RuntimeConfig config{
        require_positive_int(json, "context_length"),
        require_positive_int(json, "num_input_channels"),
        require_positive_int(json, "prediction_length"),
    };
    return config;
}

Pipeline::Pipeline(std::unique_ptr<ITrtModule> engine, RuntimeConfig config)
    : engine_(std::move(engine)), config_(config) {
    if (!engine_ || !engine_->ok())
        throw std::runtime_error("PatchTSMixer received an invalid engine");
    if (!engine_->has_input("past_values") || !engine_->has_input("observed_mask"))
        throw std::runtime_error("PatchTSMixer engine input contract does not match");
}

ForecastResult Pipeline::forecast(const ForecastRequest& request) {
    const auto channels = static_cast<std::size_t>(config_.num_input_channels);
    if (request.past_values.empty())
        throw std::invalid_argument("PatchTSMixer requires at least one past timestep");
    if (request.frequency != 0)
        throw std::invalid_argument("PatchTSMixer does not accept a frequency category");
    if (request.past_values.size() % channels != 0) {
        throw std::invalid_argument("PatchTSMixer past values must be divisible by input channels");
    }
    if (!request.observed_mask.empty() &&
        request.observed_mask.size() != request.past_values.size())
        throw std::invalid_argument("PatchTSMixer observed mask must match past values");
    const auto expected = static_cast<std::size_t>(config_.context_length) * channels;
    std::vector<float> padded_values(expected, 0.0F);
    std::vector<float> padded_mask(expected, 0.0F);
    const auto copied = std::min(expected, request.past_values.size());
    const auto source = request.past_values.size() - copied;
    const auto destination = expected - copied;
    std::copy_n(request.past_values.data() + source, copied, padded_values.data() + destination);
    if (request.observed_mask.empty())
        std::fill_n(padded_mask.data() + destination, copied, 1.0F);
    else
        std::copy_n(request.observed_mask.data() + source, copied,
                    padded_mask.data() + destination);

    TensorMap inputs;
    Tensor values;
    values.data = padded_values.data();
    values.shape = {1, config_.context_length, config_.num_input_channels};
    values.dtype = DType::kFloat32;
    inputs.emplace("past_values", values);

    Tensor mask;
    mask.data = padded_mask.data();
    mask.shape = values.shape;
    mask.dtype = DType::kFloat32;
    inputs.emplace("observed_mask", mask);

    auto outputs = engine_->forward(inputs);
    const auto output = outputs.find("prediction_outputs");
    if (output == outputs.end() || output->second.data == nullptr)
        throw std::runtime_error("PatchTSMixer engine did not return prediction_outputs");

    const auto count = static_cast<std::size_t>(config_.prediction_length) *
                       static_cast<std::size_t>(config_.num_input_channels);
    if (output->second.numel() != count)
        throw std::runtime_error("PatchTSMixer output dimensions do not match runtime.json");

    ForecastResult result;
    result.values.resize(count);
    std::memcpy(result.values.data(), output->second.data, count * sizeof(float));
    result.shape = {1, config_.prediction_length, config_.num_input_channels};
    return result;
}

} // namespace trtmc::patchtsmixer

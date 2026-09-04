/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/timesfm/runtime/pipeline.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc::timesfm {

Pipeline::Pipeline(std::unique_ptr<ITrtModule> engine, RuntimeConfig config)
    : engine_(std::move(engine)), config_(config) {
    if (!engine_ || !engine_->ok())
        throw std::runtime_error("TimesFM received an invalid engine");
    if (!engine_->has_input("past_values") || !engine_->has_input("past_values_padding") ||
        !engine_->has_input("freq") || !engine_->has_output("mean_predictions")) {
        throw std::runtime_error("TimesFM engine contract does not match runtime.json");
    }
}

ForecastResult Pipeline::forecast(const ForecastRequest& request) {
    const auto count = static_cast<std::size_t>(config_.context_length);
    if (request.past_values.empty())
        throw std::invalid_argument("TimesFM requires at least one past value");
    if (request.frequency < 0 || request.frequency >= config_.frequency_count)
        throw std::invalid_argument("TimesFM frequency category is outside the built embedding");
    if (!request.observed_mask.empty() &&
        request.observed_mask.size() != request.past_values.size())
        throw std::invalid_argument("TimesFM observed mask must match past values");

    std::vector<float> padded_values(count, 0.0F);
    std::vector<float> padded_mask(count, 0.0F);
    const auto copied = std::min(count, request.past_values.size());
    const auto source = request.past_values.size() - copied;
    const auto destination = count - copied;
    std::copy_n(request.past_values.data() + source, copied, padded_values.data() + destination);
    if (request.observed_mask.empty())
        std::fill_n(padded_mask.data() + destination, copied, 1.0F);
    else
        std::copy_n(request.observed_mask.data() + source, copied,
                    padded_mask.data() + destination);

    std::vector<std::int32_t> padding(count);
    for (std::size_t index = 0; index < count; ++index)
        padding[index] = padded_mask[index] > 0.0F ? 0 : 1;

    Tensor values;
    values.data = padded_values.data();
    values.shape = {1, config_.context_length};
    values.dtype = DType::kFloat32;
    Tensor mask;
    mask.data = padding.data();
    mask.shape = values.shape;
    mask.dtype = DType::kInt32;
    Tensor frequency;
    auto request_frequency = request.frequency;
    frequency.data = &request_frequency;
    frequency.shape = {1};
    frequency.dtype = DType::kInt32;

    TensorMap inputs;
    inputs.emplace("past_values", values);
    inputs.emplace("past_values_padding", mask);
    inputs.emplace("freq", frequency);
    auto outputs = engine_->forward(inputs);
    const auto output = outputs.find("mean_predictions");
    if (output == outputs.end() || output->second.data == nullptr)
        throw std::runtime_error("TimesFM engine omitted mean_predictions");
    if (output->second.numel() != static_cast<std::size_t>(config_.prediction_length))
        throw std::runtime_error("TimesFM output dimensions do not match runtime.json");

    ForecastResult result;
    result.values.resize(static_cast<std::size_t>(config_.prediction_length));
    std::memcpy(result.values.data(), output->second.data, result.values.size() * sizeof(float));
    result.shape = {1, config_.prediction_length};
    return result;
}

} // namespace trtmc::timesfm

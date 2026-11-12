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
namespace {
const internal::ConfigField fields[] = {
    {"frequency", internal::ConfigKind::I64, internal::ConfigValue{std::int64_t{0}},
     "Checkpoint frequency embedding category; defaults to zero."},
};
} // namespace

Pipeline::Pipeline(std::unique_ptr<ITrtModule> engine, RuntimeConfig config)
    : engine_(std::move(engine)), config_(config) {
    if (config_.context_length <= 0 || config_.prediction_length <= 0 ||
        config_.frequency_count <= 0 || config_.tensor_parallel_size <= 0)
        throw std::invalid_argument("TimesFM requires positive runtime dimensions");
    if (!engine_ || !engine_->ok())
        throw std::runtime_error("TimesFM received an invalid engine");
    if (!engine_->has_input("past_values") || !engine_->has_input("past_values_padding") ||
        !engine_->has_input("freq") || !engine_->has_output("mean_predictions")) {
        throw std::runtime_error("TimesFM engine contract does not match runtime.json");
    }
}

std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    return {internal::bind<internal::ISeriesToPointForecast>(*this, fields)};
}

internal::PointForecastResult Pipeline::run(const internal::SeriesToPointForecastRequest& request,
                                            internal::ConfigView config) {
    internal::validate_config(fields, config);
    const auto frequency = *internal::config_get<std::int64_t>(config, fields, "frequency");
    const auto& history = request.history;
    const auto past_values = history.past_values.values;
    const auto observed = history.observed;
    const auto count = static_cast<std::size_t>(config_.context_length);
    if (past_values.empty() || past_values.data() == nullptr)
        throw std::invalid_argument("TimesFM requires at least one past value");
    const auto rows = history.past_values.rows;
    const auto columns = history.past_values.columns;
    if ((rows != 0 || columns != 0) && (columns != 1 || rows != past_values.size()))
        throw std::invalid_argument("TimesFM requires a [time,1] or unspecified history shape");
    if (frequency < 0 || frequency >= config_.frequency_count)
        throw internal::ConfigError("TimesFM frequency category is outside the built embedding");
    if (!observed.empty() && (observed.size() != past_values.size() || !observed.data()))
        throw std::invalid_argument("TimesFM observed mask must match past values");
    for (const auto value : observed)
        if (value > 1)
            throw std::invalid_argument("TimesFM observed mask must contain binary values");

    std::vector<float> padded_values(count, 0.0F);
    std::vector<float> padded_mask(count, 0.0F);
    const auto copied = std::min(count, past_values.size());
    const auto source = past_values.size() - copied;
    const auto destination = count - copied;
    std::copy_n(past_values.data() + source, copied, padded_values.data() + destination);
    if (observed.empty())
        std::fill_n(padded_mask.data() + destination, copied, 1.0F);
    else
        std::copy_n(observed.data() + source, copied, padded_mask.data() + destination);

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
    Tensor frequency_tensor;
    auto request_frequency = static_cast<std::int32_t>(frequency);
    frequency_tensor.data = &request_frequency;
    frequency_tensor.shape = {1};
    frequency_tensor.dtype = DType::kInt32;

    TensorMap inputs;
    inputs.emplace("past_values", values);
    inputs.emplace("past_values_padding", mask);
    inputs.emplace("freq", frequency_tensor);
    auto outputs = engine_->forward(inputs);
    const auto output = outputs.find("mean_predictions");
    if (output == outputs.end() || output->second.data == nullptr)
        throw std::runtime_error("TimesFM engine omitted mean_predictions");
    if (output->second.dtype != DType::kFloat32 ||
        output->second.shape != std::vector<std::int64_t>{1, config_.prediction_length})
        throw std::runtime_error("TimesFM output dimensions do not match runtime.json");

    internal::PointForecastResult result;
    result.values.values.resize(static_cast<std::size_t>(config_.prediction_length));
    std::memcpy(result.values.values.data(), output->second.data,
                result.values.values.size() * sizeof(float));
    result.values.rows = config_.prediction_length;
    result.values.columns = 1;
    for (std::int64_t step = 1; step <= config_.prediction_length; ++step)
        result.axes.horizon_steps.push_back(step);
    return result;
}

} // namespace trtmc::timesfm

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/patchtst/runtime/pipeline.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc::patchtst {
namespace {
const internal::ConfigField fields[] = {
    {"frequency", internal::ConfigKind::I64, internal::ConfigValue{std::int64_t{0}},
     "Only zero is supported; PatchTST has no frequency embedding."},
};

std::vector<std::string> parameter_names(std::string_view distribution) {
    if (distribution == "normal")
        return {"location", "scale"};
    if (distribution == "student_t")
        return {"degrees_of_freedom", "location", "scale"};
    if (distribution == "negative_binomial")
        return {"total_count", "logits"};
    throw std::invalid_argument("PatchTST has an unsupported regression distribution");
}
} // namespace

Pipeline::Pipeline(std::unique_ptr<ITrtModule> module, RuntimeConfig config)
    : module_(std::move(module)), config_(std::move(config)) {
    if (config_.context_length <= 0 || config_.num_input_channels <= 0 ||
        config_.prediction_length <= 0 || config_.tensor_parallel_size <= 0 ||
        config_.num_targets <= 0)
        throw std::invalid_argument("PatchTST requires positive runtime dimensions");
    if (!module_ || !module_->ok())
        throw std::runtime_error("PatchTST received an invalid engine");
    const bool forecast = config_.task == internal::ISeriesToPointForecast::kTask;
    const bool values = config_.task == internal::ISeriesToRegressionValues::kTask;
    const bool distribution = config_.task == internal::ISeriesToRegressionDistribution::kTask;
    if (!forecast && !values && !distribution)
        throw std::invalid_argument("PatchTST has an unsupported primary Task");
    if (config_.output_name != (forecast ? "prediction_outputs" : "regression_outputs"))
        throw std::invalid_argument("PatchTST output name does not match its primary Task");
    if (distribution)
        (void)parameter_names(config_.distribution);
    else if (!config_.distribution.empty())
        throw std::invalid_argument("PatchTST point heads do not return distribution parameters");
    if (!module_->has_input("past_values") || !module_->has_input("past_observed_mask"))
        throw std::runtime_error("PatchTST engine input contract does not match runtime.json");
    if (!module_->has_output(config_.output_name))
        throw std::runtime_error("PatchTST engine output contract does not match runtime.json");
}

std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    if (config_.task == internal::ISeriesToPointForecast::kTask)
        return {internal::bind<internal::ISeriesToPointForecast>(*this, fields)};
    if (config_.task == internal::ISeriesToRegressionValues::kTask)
        return {internal::bind<internal::ISeriesToRegressionValues>(*this, fields)};
    return {internal::bind<internal::ISeriesToRegressionDistribution>(*this, fields)};
}

Tensor Pipeline::evaluate(const internal::SeriesHistory& history, internal::ConfigView config,
                          std::string_view required_task) {
    if (config_.task != required_task)
        throw internal::UnsupportedTask("PatchTST bundle does not implement the requested Task");
    internal::validate_config(fields, config);
    if (*internal::config_get<std::int64_t>(config, fields, "frequency") != 0)
        throw internal::ConfigError("PatchTST does not accept a nonzero frequency category");
    const auto channels = static_cast<std::size_t>(config_.num_input_channels);
    const auto values = history.past_values.values;
    const auto observed = history.observed;
    if (values.empty() || !values.data())
        throw std::invalid_argument("PatchTST requires at least one past timestep");
    if (values.size() % channels != 0)
        throw std::invalid_argument("PatchTST past values must contain complete channel timesteps");
    const auto rows = history.past_values.rows;
    const auto columns = history.past_values.columns;
    if ((rows != 0 || columns != 0) && (columns != channels || rows != values.size() / channels))
        throw std::invalid_argument("PatchTST requires [time,input_channel] or a flat history");
    if (!observed.empty() && (!observed.data() || observed.size() != values.size()))
        throw std::invalid_argument("PatchTST observed mask must match past values");
    for (const auto value : observed)
        if (value > 1)
            throw std::invalid_argument("PatchTST observed mask must contain binary values");
    const auto input_count = static_cast<std::size_t>(config_.context_length) * channels;
    std::vector<float> padded_values(input_count, 0.0F);
    std::vector<float> padded_mask(input_count, 0.0F);
    const auto copied = std::min(input_count, values.size());
    const auto source = values.size() - copied;
    const auto destination = input_count - copied;
    std::copy_n(values.data() + source, copied, padded_values.data() + destination);
    if (observed.empty())
        std::fill_n(padded_mask.data() + destination, copied, 1.0F);
    else
        std::copy_n(observed.data() + source, copied, padded_mask.data() + destination);

    Tensor input;
    input.data = padded_values.data();
    input.shape = {1, config_.context_length, config_.num_input_channels};
    input.dtype = DType::kFloat32;
    Tensor mask;
    mask.data = padded_mask.data();
    mask.shape = input.shape;
    mask.dtype = DType::kFloat32;
    auto outputs = module_->forward({{"past_values", input}, {"past_observed_mask", mask}});
    const auto output = outputs.find(config_.output_name);
    if (output == outputs.end() || !output->second.data)
        throw std::runtime_error("PatchTST engine omitted its declared output");
    if (output->second.dtype != DType::kFloat32)
        throw std::runtime_error("PatchTST engine must export float32 results");
    return output->second;
}

internal::PointForecastResult Pipeline::run(const internal::SeriesToPointForecastRequest& request,
                                            internal::ConfigView config) {
    const auto output = evaluate(request.history, config, internal::ISeriesToPointForecast::kTask);
    if (output.shape !=
        std::vector<std::int64_t>{1, config_.prediction_length, config_.num_input_channels})
        throw std::runtime_error("PatchTST forecast dimensions do not match runtime.json");
    internal::PointForecastResult result;
    result.values.rows = config_.prediction_length;
    result.values.columns = config_.num_input_channels;
    const auto* data = static_cast<const float*>(output.data);
    result.values.values.assign(data, data + output.numel());
    for (std::int64_t step = 1; step <= config_.prediction_length; ++step)
        result.axes.horizon_steps.push_back(step);
    return result;
}

internal::RegressionValuesResult
Pipeline::run(const internal::SeriesToRegressionValuesRequest& request,
              internal::ConfigView config) {
    const auto output =
        evaluate(request.history, config, internal::ISeriesToRegressionValues::kTask);
    if (output.shape != std::vector<std::int64_t>{1, config_.num_targets})
        throw std::runtime_error("PatchTST regression dimensions do not match runtime.json");
    internal::RegressionValuesResult result;
    const auto* data = static_cast<const float*>(output.data);
    result.values.assign(data, data + config_.num_targets);
    return result;
}

internal::RegressionDistributionResult
Pipeline::run(const internal::SeriesToRegressionDistributionRequest& request,
              internal::ConfigView config) {
    const auto output =
        evaluate(request.history, config, internal::ISeriesToRegressionDistribution::kTask);
    const auto names = parameter_names(config_.distribution);
    if (output.shape !=
        std::vector<std::int64_t>{1, config_.num_targets, static_cast<std::int64_t>(names.size())})
        throw std::runtime_error("PatchTST parameter dimensions do not match runtime.json");
    internal::RegressionDistributionResult result;
    result.distribution = config_.distribution == "normal" ? internal::DistributionKind::Normal
                          : config_.distribution == "student_t"
                              ? internal::DistributionKind::StudentT
                              : internal::DistributionKind::NegativeBinomial;
    result.target_count = config_.num_targets;
    const auto* data = static_cast<const float*>(output.data);
    for (std::size_t parameter = 0; parameter < names.size(); ++parameter) {
        internal::DistributionParameter item;
        item.name = names[parameter];
        for (std::int32_t target = 0; target < config_.num_targets; ++target)
            item.values.push_back(
                data[static_cast<std::size_t>(target) * names.size() + parameter]);
        result.parameters.push_back(std::move(item));
    }
    return result;
}

} // namespace trtmc::patchtst

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/chronos_bolt/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc::chronos_bolt {

Pipeline::Pipeline(std::unique_ptr<ITrtModule> engine, RuntimeConfig config)
    : engine_(std::move(engine)), config_(config) {
    if (!engine_ || !engine_->ok())
        throw std::runtime_error("Chronos-Bolt received an invalid engine");
    if (!engine_->has_input("context") || !engine_->has_output("quantile_preds"))
        throw std::runtime_error("Chronos-Bolt engine contract does not match runtime.json");
}

internal::QuantileForecastResult
Pipeline::run(const internal::SeriesToQuantileForecastRequest& request,
              internal::ConfigView config) {
    const auto count = static_cast<std::size_t>(config_.context_length);
    const auto& history = request.history;
    const auto past_values = history.past_values.values;
    if (past_values.empty())
        throw std::invalid_argument("Chronos-Bolt requires at least one past value");
    if (internal::config_get<std::int64_t>(config, fields_, "frequency").value() != 0)
        throw internal::ConfigError("Chronos-Bolt does not accept a frequency category");
    // Chronos-Bolt is univariate: a shape the caller states must be [time, 1].
    if ((history.past_values.rows != 0 || history.past_values.columns != 0) &&
        (history.past_values.columns != 1 || history.past_values.rows != past_values.size()))
        throw std::invalid_argument("Chronos-Bolt history must have exactly one channel");
    if (!history.observed.empty() && history.observed.size() != past_values.size())
        throw std::invalid_argument("Chronos-Bolt observed mask must match past values");

    std::vector<float> values(count, 0.0F);
    std::vector<float> mask(count, 0.0F);
    const auto copied = std::min(count, past_values.size());
    const auto source = past_values.size() - copied;
    const auto destination = count - copied;
    std::copy_n(past_values.data() + source, copied, values.data() + destination);
    if (history.observed.empty())
        std::fill_n(mask.data() + destination, copied, 1.0F);
    else
        for (std::size_t index = 0; index < copied; ++index)
            mask[destination + index] = history.observed[source + index] != 0 ? 1.0F : 0.0F;

    std::vector<float> context(count);
    for (std::size_t index = 0; index < count; ++index) {
        context[index] =
            mask[index] > 0.0F ? values[index] : std::numeric_limits<float>::quiet_NaN();
    }

    Tensor input;
    input.data = context.data();
    input.shape = {1, config_.context_length};
    input.dtype = DType::kFloat32;
    TensorMap inputs;
    inputs.emplace("context", input);

    auto outputs = engine_->forward(inputs);
    const auto output = outputs.find("quantile_preds");
    if (output == outputs.end() || output->second.data == nullptr)
        throw std::runtime_error("Chronos-Bolt engine omitted quantile_preds");
    const auto expected =
        static_cast<std::size_t>(config_.prediction_length) * config_.quantile_levels.size();
    if (output->second.numel() != expected)
        throw std::runtime_error(
            "Chronos-Bolt quantile output dimensions do not match runtime.json");

    // The engine returns [1, quantile, horizon]; with one channel that is exactly the
    // [quantile, horizon, channel] layout of the Task result.
    internal::QuantileForecastResult result;
    result.values.resize(expected);
    std::memcpy(result.values.data(), output->second.data, expected * sizeof(float));
    result.quantile_levels = config_.quantile_levels;
    result.horizon = static_cast<std::uint64_t>(config_.prediction_length);
    result.channels = 1;
    result.axes.horizon_steps.resize(static_cast<std::size_t>(config_.prediction_length));
    std::iota(result.axes.horizon_steps.begin(), result.axes.horizon_steps.end(), std::int64_t{1});
    return result;
}

} // namespace trtmc::chronos_bolt

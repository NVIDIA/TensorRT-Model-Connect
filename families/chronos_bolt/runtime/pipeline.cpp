/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/chronos_bolt/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
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

ForecastResult Pipeline::forecast(const ForecastRequest& request) {
    const auto count = static_cast<std::size_t>(config_.context_length);
    if (request.past_values.empty())
        throw std::invalid_argument("Chronos-Bolt requires at least one past value");
    if (request.frequency != 0)
        throw std::invalid_argument("Chronos-Bolt does not accept a frequency category");
    if (!request.observed_mask.empty() &&
        request.observed_mask.size() != request.past_values.size())
        throw std::invalid_argument("Chronos-Bolt observed mask must match past values");

    std::vector<float> values(count, 0.0F);
    std::vector<float> mask(count, 0.0F);
    const auto copied = std::min(count, request.past_values.size());
    const auto source = request.past_values.size() - copied;
    const auto destination = count - copied;
    std::copy_n(request.past_values.data() + source, copied, values.data() + destination);
    if (request.observed_mask.empty())
        std::fill_n(mask.data() + destination, copied, 1.0F);
    else
        std::copy_n(request.observed_mask.data() + source, copied, mask.data() + destination);

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
    const auto expected = static_cast<std::size_t>(config_.prediction_length) *
                          static_cast<std::size_t>(config_.num_quantiles);
    if (output->second.numel() != expected)
        throw std::runtime_error(
            "Chronos-Bolt quantile output dimensions do not match runtime.json");

    ForecastResult result;
    result.values.resize(expected);
    std::memcpy(result.values.data(), output->second.data, expected * sizeof(float));
    result.shape = {1, config_.prediction_length, config_.num_quantiles};
    return result;
}

} // namespace trtmc::chronos_bolt

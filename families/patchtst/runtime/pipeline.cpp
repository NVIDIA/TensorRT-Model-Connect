/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/patchtst/runtime/pipeline.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc::patchtst {

Pipeline::Pipeline(std::unique_ptr<ITrtModule> module, RuntimeConfig config)
    : module_(std::move(module)), config_(std::move(config)) {
    if (!module_ || !module_->ok())
        throw std::runtime_error("PatchTST received an invalid engine");
    if (!module_->has_input("past_values") || !module_->has_input("past_observed_mask"))
        throw std::runtime_error("PatchTST engine input contract does not match runtime.json");
    if (!module_->has_output(config_.output_name))
        throw std::runtime_error("PatchTST engine output contract does not match runtime.json");
}

ForecastResult Pipeline::forecast(const ForecastRequest& request) {
    const auto channels = static_cast<std::size_t>(config_.num_input_channels);
    if (request.past_values.empty())
        throw std::invalid_argument("PatchTST requires at least one past timestep");
    if (request.frequency != 0)
        throw std::invalid_argument("PatchTST does not accept a frequency category");
    if (request.past_values.size() % channels != 0)
        throw std::invalid_argument("PatchTST past values must be divisible by input channels");
    if (!request.observed_mask.empty() &&
        request.observed_mask.size() != request.past_values.size())
        throw std::invalid_argument("PatchTST observed mask must match past values");
    const auto input_count = static_cast<std::size_t>(config_.context_length) * channels;
    std::vector<float> padded_values(input_count, 0.0F);
    std::vector<float> padded_mask(input_count, 0.0F);
    const auto copied = std::min(input_count, request.past_values.size());
    const auto source = request.past_values.size() - copied;
    const auto destination = input_count - copied;
    std::copy_n(request.past_values.data() + source, copied, padded_values.data() + destination);
    if (request.observed_mask.empty())
        std::fill_n(padded_mask.data() + destination, copied, 1.0F);
    else
        std::copy_n(request.observed_mask.data() + source, copied,
                    padded_mask.data() + destination);

    Tensor values;
    values.data = padded_values.data();
    values.shape = {1, config_.context_length, config_.num_input_channels};
    values.dtype = DType::kFloat32;

    Tensor mask;
    mask.data = padded_mask.data();
    mask.shape = values.shape;
    mask.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs.emplace("past_values", values);
    inputs.emplace("past_observed_mask", mask);
    auto outputs = module_->forward(inputs);
    const auto output = outputs.find(config_.output_name);
    if (output == outputs.end() || output->second.data == nullptr)
        throw std::runtime_error("PatchTST engine omitted its declared output");

    ForecastResult result;
    const auto count = output->second.numel();
    result.values.resize(count);
    std::memcpy(result.values.data(), output->second.data, count * sizeof(float));
    result.shape = output->second.shape;
    return result;
}

} // namespace trtmc::patchtst

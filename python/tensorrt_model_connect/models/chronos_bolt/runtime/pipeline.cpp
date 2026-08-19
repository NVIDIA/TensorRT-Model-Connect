/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

std::vector<float> build_nan_padded_context(const float* src, int32_t src_len,
                                            int32_t context_len) {
    std::vector<float> context(static_cast<std::size_t>(context_len),
                               std::numeric_limits<float>::quiet_NaN());
    if (!src || src_len <= 0 || context_len <= 0)
        return context;

    const int32_t copy_len = std::min(src_len, context_len);
    const int32_t src_offset = src_len - copy_len;
    const int32_t dst_offset = context_len - copy_len;
    std::memcpy(context.data() + dst_offset, src + src_offset,
                static_cast<std::size_t>(copy_len) * sizeof(float));
    return context;
}

TensorMap build_forecast_inputs(TrtModule& forecast, Tensor& context_t) {
    TensorMap inputs;
    if (forecast.has_input("context"))
        inputs["context"] = context_t;
    else if (forecast.has_input("past_values"))
        inputs["past_values"] = context_t;
    else
        inputs["input_0"] = context_t;
    return inputs;
}

const Tensor* select_forecast_tensor(const TensorMap& outputs) {
    for (const auto& [name, tensor] : outputs) {
        if (name.find("forecast") != std::string::npos ||
            name.find("prediction") != std::string::npos ||
            name.find("quantile") != std::string::npos ||
            name.find("output") != std::string::npos || name.find("logit") != std::string::npos) {
            return &tensor;
        }
    }
    if (!outputs.empty())
        return &outputs.begin()->second;
    return nullptr;
}

EmbeddingResult tensor_to_embedding_result(const Tensor& tensor) {
    EmbeddingResult result;
    const auto n = tensor.numel();
    result.data.resize(static_cast<std::size_t>(n));
    std::memcpy(result.data.data(), tensor.data, n * sizeof(float));
    result.dim = static_cast<int32_t>(n);
    return result;
}

} // namespace

ChronosBoltPipeline::ChronosBoltPipeline(std::unique_ptr<TrtModule> forecast,
                                         int32_t context_length, int32_t prediction_length,
                                         int32_t num_quantiles, std::string model_id_str)
    : forecast_(std::move(forecast)), context_length_(context_length),
      prediction_length_(prediction_length), num_quantiles_(num_quantiles),
      model_id_(std::move(model_id_str)) {
    if (!forecast_ || !forecast_->ok())
        throw std::runtime_error("ChronosBoltPipeline: invalid forecast module");
}

EmbeddingResult ChronosBoltPipeline::solve(const float* branch_input, int32_t branch_len,
                                           const float* trunk_input, int32_t trunk_len) {
    if (!branch_input || branch_len <= 0)
        throw std::runtime_error("ChronosBoltPipeline: branch_input is required");

    (void)trunk_input;
    (void)trunk_len;

    const int32_t context_len = std::max(context_length_, branch_len);
    auto context = build_nan_padded_context(branch_input, branch_len, context_len);

    Tensor context_t;
    context_t.data = context.data();
    context_t.shape = {1, static_cast<int64_t>(context_len)};
    context_t.dtype = DType::kFloat32;

    TensorMap inputs = build_forecast_inputs(*forecast_, context_t);
    auto outputs = forecast_->forward(inputs);

    const Tensor* forecast_tensor = select_forecast_tensor(outputs);
    if (!forecast_tensor)
        throw std::runtime_error("ChronosBoltPipeline: no forecast output");
    return tensor_to_embedding_result(*forecast_tensor);
}

} // namespace trtmc

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/timesfm/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace trtmc {

namespace {

void copy_tensor_to_result(const Tensor& tensor, EmbeddingResult& result) {
    const std::size_t n = tensor.numel();
    result.data.resize(n);
    if (n > 0 && tensor.data) {
        std::memcpy(result.data.data(), tensor.data, n * sizeof(float));
    }
    result.dim = static_cast<int32_t>(n);
}

const Tensor* find_named_output(const TensorMap& outputs, const char* name,
                                std::string& selected_name) {
    auto it = outputs.find(name);
    if (it == outputs.end() || !it->second.data || it->second.numel() == 0)
        return nullptr;
    selected_name = name;
    return &it->second;
}

std::vector<float> build_left_padded_series(const float* src, int32_t src_len,
                                            int32_t context_len) {
    std::vector<float> series(static_cast<std::size_t>(context_len), 0.0f);
    if (!src || src_len <= 0 || context_len <= 0) {
        return series;
    }

    const int32_t copy_len = std::min(src_len, context_len);
    const int32_t src_offset = src_len - copy_len;
    const int32_t dst_offset = context_len - copy_len;
    std::memcpy(series.data() + dst_offset, src + src_offset,
                static_cast<std::size_t>(copy_len) * sizeof(float));
    return series;
}

std::vector<int32_t> build_padding_indicator(int32_t src_len, int32_t context_len) {
    std::vector<int32_t> padding(static_cast<std::size_t>(context_len), 1);
    if (src_len <= 0 || context_len <= 0) {
        return padding;
    }

    const int32_t copy_len = std::min(src_len, context_len);
    const int32_t dst_offset = context_len - copy_len;
    std::fill(padding.begin() + dst_offset, padding.end(), 0);
    return padding;
}

int32_t resolve_frequency(int32_t default_freq, const float* trunk_input, int32_t trunk_len) {
    if (!trunk_input || trunk_len <= 0)
        return default_freq;
    return static_cast<int32_t>(std::lround(trunk_input[0]));
}

TensorMap build_timesfm_inputs(TrtModule& model, std::vector<float>& series,
                               std::vector<int32_t>& padding, int32_t& freq, int32_t context_len) {
    Tensor series_t;
    series_t.data = series.data();
    series_t.shape = {1, context_len};
    series_t.dtype = DType::kFloat32;

    Tensor padding_t;
    padding_t.data = padding.data();
    padding_t.shape = {1, context_len};
    padding_t.dtype = DType::kInt32;

    Tensor freq_t;
    freq_t.data = &freq;
    freq_t.shape = {1};
    freq_t.dtype = DType::kInt32;

    TensorMap inputs;
    if (model.has_input("past_values"))
        inputs["past_values"] = series_t;
    if (model.has_input("past_values_padding"))
        inputs["past_values_padding"] = padding_t;
    if (model.has_input("freq"))
        inputs["freq"] = freq_t;
    return inputs;
}

void truncate_result(EmbeddingResult& result, int32_t prediction_length) {
    if (prediction_length <= 0 || result.dim <= prediction_length)
        return;
    result.data.resize(static_cast<std::size_t>(prediction_length));
    result.dim = prediction_length;
}

} // namespace

TimesFmPipeline::TimesFmPipeline(std::unique_ptr<TrtModule> model, int32_t default_freq,
                                 int32_t prediction_length, std::string model_id_str)
    : model_(std::move(model)), default_freq_(default_freq), prediction_length_(prediction_length),
      model_id_(std::move(model_id_str)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("TimesFmPipeline: invalid TRT module");
}

int32_t TimesFmPipeline::infer_input_length(const TrtModule& module, const std::string& name,
                                            int32_t fallback) {
    for (const auto& info : module.input_info()) {
        if (info.name != name || info.shape.empty())
            continue;
        const int64_t tail = info.shape.back();
        if (tail > 0)
            return static_cast<int32_t>(tail);
    }
    return fallback;
}

const Tensor* TimesFmPipeline::select_forecast_output(const TensorMap& outputs,
                                                      std::string& selected_name) {
    static const char* kPreferredOutputs[] = {
        "output0", "mean_predictions", "mean", "forecast", "prediction",
    };
    for (const char* name : kPreferredOutputs) {
        if (const Tensor* tensor = find_named_output(outputs, name, selected_name))
            return tensor;
    }

    for (const auto& [name, tensor] : outputs) {
        if (tensor.data && tensor.numel() > 0) {
            selected_name = name;
            return &tensor;
        }
    }

    selected_name.clear();
    return nullptr;
}

EmbeddingResult TimesFmPipeline::solve(const float* branch_input, int32_t branch_len,
                                       const float* trunk_input, int32_t trunk_len) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("TimesFmPipeline: invalid TRT module");
    if (!branch_input || branch_len <= 0)
        throw std::runtime_error("TimesFmPipeline::solve requires a non-empty branch_input");

    // Interpret branch_input as a flattened single series. trunk_input is
    // treated as an optional metadata channel: if provided, the first value
    // overrides the default frequency index. Otherwise we fall back to the
    // bundle/default freq.
    int32_t freq = resolve_frequency(default_freq_, trunk_input, trunk_len);

    const int32_t context_len = infer_input_length(*model_, "past_values", branch_len);
    auto series = build_left_padded_series(branch_input, branch_len, context_len);
    auto padding = build_padding_indicator(branch_len, context_len);
    TensorMap inputs = build_timesfm_inputs(*model_, series, padding, freq, context_len);

    auto outputs = model_->forward(inputs);

    std::string selected_name;
    const Tensor* forecast = select_forecast_output(outputs, selected_name);
    if (!forecast)
        throw std::runtime_error("TimesFmPipeline: no forecast output found in engine");

    EmbeddingResult result;
    copy_tensor_to_result(*forecast, result);
    truncate_result(result, prediction_length_);
    return result;
}

} // namespace trtmc

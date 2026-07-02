/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/patchtst/pipeline.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {

namespace {

int32_t infer_input_channels(const TrtModule& module, int32_t fallback) {
    for (const auto& info : module.input_info()) {
        if (info.name != "past_values")
            continue;
        if (info.shape.size() >= 3 && info.shape.back() > 0)
            return static_cast<int32_t>(info.shape.back());
    }
    return std::max(fallback, 1);
}

int32_t infer_context_length(const TrtModule& module, int32_t fallback) {
    for (const auto& info : module.input_info()) {
        if (info.name != "past_values")
            continue;
        if (info.shape.size() >= 3 && info.shape[1] > 0)
            return static_cast<int32_t>(info.shape[1]);
        if (info.shape.size() == 2 && info.shape[0] > 0)
            return static_cast<int32_t>(info.shape[0]);
    }
    return std::max(fallback, 1);
}

std::vector<float> build_left_padded_window(const float* src, int32_t src_len,
                                            std::size_t expected_len) {
    std::vector<float> out(expected_len, 0.0f);
    if (!src || src_len <= 0 || expected_len == 0)
        return out;

    const std::size_t copy_len = std::min(expected_len, static_cast<std::size_t>(src_len));
    const std::size_t src_offset = static_cast<std::size_t>(src_len) - copy_len;
    const std::size_t dst_offset = expected_len - copy_len;

    std::memcpy(out.data() + dst_offset, src + src_offset, copy_len * sizeof(float));
    return out;
}

std::vector<float> build_observed_mask(int32_t src_len, std::size_t expected_len) {
    std::vector<float> mask(expected_len, 0.0f);
    if (src_len <= 0 || expected_len == 0)
        return mask;

    const std::size_t copy_len = std::min(expected_len, static_cast<std::size_t>(src_len));
    const std::size_t dst_offset = expected_len - copy_len;
    std::fill(mask.begin() + static_cast<std::ptrdiff_t>(dst_offset), mask.end(), 1.0f);
    return mask;
}

bool name_matches(const std::string& name, const std::vector<const char*>& needles) {
    for (const char* needle : needles) {
        if (name.find(needle) != std::string::npos)
            return true;
    }
    return false;
}

const Tensor* select_primary_output(const TensorMap& outputs, const std::string& task_type) {
    std::vector<const char*> preferred;
    if (task_type == "classification") {
        preferred = {"prediction_logits", "logits", "output", "prediction", "score"};
    } else if (task_type == "regression") {
        preferred = {"regression_outputs",
                     "prediction_outputs",
                     "prediction_logits",
                     "logits",
                     "output",
                     "prediction",
                     "score"};
    } else {
        preferred = {"prediction_outputs", "prediction", "output", "logits", "score"};
    }

    for (const auto& [name, tensor] : outputs) {
        if (name_matches(name, preferred))
            return &tensor;
    }

    if (!outputs.empty())
        return &outputs.begin()->second;
    return nullptr;
}

} // namespace

PatchTSTPipeline::PatchTSTPipeline(std::unique_ptr<TrtModule> module, std::string task_type,
                                   int32_t context_length, int32_t num_input_channels,
                                   int32_t prediction_length, int32_t num_targets,
                                   std::string model_id_str)
    : module_(std::move(module)), task_type_(std::move(task_type)), context_length_(context_length),
      num_input_channels_(num_input_channels), prediction_length_(prediction_length),
      num_targets_(num_targets), model_id_(std::move(model_id_str)) {
    if (!module_ || !module_->ok())
        throw std::runtime_error("PatchTSTPipeline: invalid TRT module");

    context_length_ = infer_context_length(*module_, context_length_);
    num_input_channels_ = infer_input_channels(*module_, num_input_channels_);
}

EmbeddingResult PatchTSTPipeline::solve(const float* branch_input, int32_t branch_len,
                                        const float* trunk_input, int32_t trunk_len) {
    // PatchTST is a pure numeric sequence model, not a branch/trunk operator.
    // The runtime contract therefore treats branch_input as the flattened
    // past_values window and ignores trunk_input by design.
    (void)trunk_input;
    (void)trunk_len;

    if (!branch_input || branch_len <= 0)
        throw std::runtime_error("PatchTSTPipeline: branch_input must be non-empty");

    const int32_t channels = std::max(num_input_channels_, 1);
    if (channels > 1 && branch_len % channels != 0) {
        throw std::runtime_error("PatchTSTPipeline: flattened branch_input length must be "
                                 "divisible by num_input_channels");
    }

    const int32_t context_length =
        (context_length_ > 0) ? context_length_ : std::max(branch_len / channels, 1);
    const std::size_t expected_len =
        static_cast<std::size_t>(context_length) * static_cast<std::size_t>(channels);

    auto values = build_left_padded_window(branch_input, branch_len, expected_len);
    auto observed_mask = build_observed_mask(branch_len, expected_len);

    Tensor values_t;
    values_t.data = values.data();
    values_t.shape = {1, context_length, channels};
    values_t.dtype = DType::kFloat32;

    TensorMap inputs;
    inputs["past_values"] = values_t;

    if (module_->has_input("past_observed_mask")) {
        Tensor mask_t;
        mask_t.data = observed_mask.data();
        mask_t.shape = {1, context_length, channels};
        mask_t.dtype = DType::kFloat32;
        inputs["past_observed_mask"] = mask_t;
    }

    auto outputs = module_->forward(inputs);
    const Tensor* tensor = select_primary_output(outputs, task_type_);
    if (!tensor)
        throw std::runtime_error("PatchTSTPipeline: no TRT outputs produced");

    EmbeddingResult result;
    const auto n = tensor->numel();
    result.data.resize(n);
    std::memcpy(result.data.data(), tensor->data, n * sizeof(float));
    result.dim = static_cast<int32_t>(n);
    return result;
}

} // namespace trtmc

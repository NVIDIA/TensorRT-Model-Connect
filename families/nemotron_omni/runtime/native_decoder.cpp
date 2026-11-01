/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "native_decoder.h"

#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::nemotron_omni {
namespace {
void checked(cudaError_t code) {
    if (code != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(code));
}
} // namespace

NativeDecoder::NativeDecoder(const FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument(
            "Omni uses its built cache capacity, not a runtime byte override");
    const auto metadata = context.reader.read_section("runtime.json");
    const auto config = nlohmann::json::parse(metadata);
    if (config.at("format") != 1 ||
        config.at("config").at("model_type") != "NemotronH_Nano_Omni_Reasoning_V3")
        throw std::invalid_argument("Invalid native Omni bundle schema");
    const auto& language = config.at("config").at("llm_config");
    hidden_ = language.at("hidden_size");
    vocab_ = language.at("vocab_size");
    capacity_ = config.at("max_cache_length");
    if (hidden_ <= 0 || vocab_ <= 0 || capacity_ <= 0)
        throw std::invalid_argument("Invalid native Omni dimensions");
    const auto plan = context.reader.read_section("decoder.plan");
    module_ = context.backend.create_module(plan.data(), plan.size(), {});
    if (!module_ || !module_->ok())
        throw std::runtime_error("Cannot load Omni decoder engine");
    module_->set_timing_label("omni_decoder");
    dtype_ = module_->tensor_dtype("input_embedding");
    if ((dtype_ != DType::kFloat32 && dtype_ != DType::kFloat16) ||
        module_->tensor_shape("input_embedding") != std::vector<int64_t>{1, hidden_} ||
        module_->tensor_shape("logits") != std::vector<int64_t>{1, vocab_} ||
        module_->tensor_dtype("logits") != DType::kFloat32)
        throw std::invalid_argument("Omni decoder tensors differ from bundle metadata");
    const auto& state = config.at("state");
    auto add = [&](const std::string& input, const std::string& output, std::vector<int64_t> shape,
                   bool append) {
        if (!module_->has_input(input) || !module_->has_output(output) ||
            module_->tensor_shape(input) != shape)
            throw std::invalid_argument("Invalid Omni decoder state: " + input);
        auto expected = shape;
        if (append)
            expected.front() = 1;
        const auto dtype = module_->tensor_dtype(input);
        if (module_->tensor_shape(output) != expected || module_->tensor_dtype(output) != dtype)
            throw std::invalid_argument("Invalid Omni decoder state output: " + output);
        states_.push_back({output, DeviceTensor(shape, dtype, module_->stream()), append});
        if (!states_.back().storage.ok())
            throw std::runtime_error("Cannot allocate Omni decoder state");
        module_->bind_external(input, states_.back().storage.data());
    };
    for (int i = 0; i < state.at("num_mamba_layers").get<int>(); ++i) {
        const auto suffix = "_" + std::to_string(i);
        add("conv_state" + suffix, "present_conv" + suffix,
            {state.at("conv_dim").get<int>(), state.at("mamba_d_conv").get<int>()}, false);
        add("ssm_state" + suffix, "present_ssm" + suffix,
            {state.at("mamba_nheads").get<int>(), state.at("mamba_head_dim").get<int>(),
             state.at("mamba_d_state").get<int>()},
            false);
    }
    const int kv =
        language.at("num_key_value_heads").get<int>() * language.at("head_dim").get<int>();
    for (int i = 0; i < state.at("num_attention_layers").get<int>(); ++i)
        for (const auto* kind : {"k", "v"}) {
            const auto suffix = std::string(kind) + "_" + std::to_string(i);
            add("cache_" + suffix, "present_" + suffix, {capacity_, kv}, true);
        }
    reset();
}

void NativeDecoder::reset() {
    module_->reset_execution_context();
    for (auto& state : states_)
        checked(
            cudaMemsetAsync(state.storage.data(), 0, state.storage.nbytes(), module_->stream()));
    module_->sync();
    position_ = 0;
}

std::vector<float> NativeDecoder::step(const std::vector<float>& embedding) {
    if (embedding.size() != static_cast<std::size_t>(hidden_) || position_ >= capacity_ ||
        !std::all_of(embedding.begin(), embedding.end(), [](float x) { return std::isfinite(x); }))
        throw std::invalid_argument("Invalid or over-capacity Omni decoder embedding");
    std::vector<__half> half;
    void* data = const_cast<float*>(embedding.data());
    if (dtype_ == DType::kFloat16) {
        half.reserve(embedding.size());
        for (const float value : embedding) {
            const auto converted = __float2half_rn(value);
            if (!std::isfinite(__half2float(converted)))
                throw std::invalid_argument("Omni embedding cannot be represented as fp16");
            half.push_back(converted);
        }
        data = half.data();
    }
    std::vector<float> mask(static_cast<std::size_t>(capacity_) + 1,
                            std::numeric_limits<float>::lowest());
    std::fill_n(mask.begin(), position_, 0.0F);
    mask.back() = 0.0F;
    TensorMap inputs{{"input_embedding", {data, {1, hidden_}, dtype_}},
                     {"attention_mask", {mask.data(), {1, capacity_ + 1}, DType::kFloat32}}};
    if (module_->has_input("position_id"))
        inputs.emplace("position_id", TensorMap::mapped_type{&position_, {1}, DType::kInt32});
    module_->forward_async(inputs);
    std::vector<float> logits(vocab_);
    checked(cudaMemcpyAsync(logits.data(), module_->device_ptr("logits"),
                            logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                            module_->stream()));
    for (auto& state : states_) {
        const auto bytes =
            state.append ? state.storage.nbytes() / capacity_ : state.storage.nbytes();
        auto* destination = static_cast<unsigned char*>(state.storage.data());
        if (state.append)
            destination += static_cast<std::size_t>(position_) * bytes;
        checked(cudaMemcpyAsync(destination, module_->device_ptr(state.output), bytes,
                                cudaMemcpyDeviceToDevice, module_->stream()));
    }
    module_->sync();
    if (!std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }))
        throw std::runtime_error("Omni decoder produced nonfinite logits; reset before retrying");
    ++position_;
    return logits;
}
} // namespace trtmc::nemotron_omni

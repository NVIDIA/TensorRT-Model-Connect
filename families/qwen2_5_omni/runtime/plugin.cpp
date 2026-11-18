/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen2_5_omni/runtime/tokenizer.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>

namespace trtmc::qwen2_5_omni {
namespace {
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || found->length == 0)
        throw std::runtime_error(std::string("Qwen2.5 Omni Thinker bundle lacks ") + name);
    return reader.read_section(name);
}

void require_tensor(ITrtModule& engine, const std::string& name, const std::vector<int64_t>& shape,
                    DType dtype, bool output) {
    if (!(output ? engine.has_output(name) : engine.has_input(name)) ||
        engine.tensor_shape(name) != shape || engine.tensor_dtype(name) != dtype ||
        !engine.device_ptr(name))
        throw std::runtime_error("Invalid Qwen2.5 Omni Thinker tensor " + name);
}

void check_copy(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Qwen2.5 Omni Thinker CUDA copy failed: ") +
                                 cudaGetErrorString(status));
}

class Pipeline final : public internal::IModel, public internal::ITextContinuation {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, std::unique_ptr<ITokenizer> tokenizer,
             nlohmann::json cfg)
        : engine_(std::move(engine)), tokenizer_(std::move(tokenizer)), cfg_(std::move(cfg)),
          fields_{{"max_new_tokens", internal::ConfigKind::I64, std::nullopt,
                   "Maximum generated tokens; defaults to min(32, remaining context)"},
                  {"do_sample", internal::ConfigKind::Bool, internal::ConfigValue{false},
                   "Greedy decoding only"}} {
        const int32_t length = cfg_.at("sequence_length"), vocab = cfg_.at("vocab_size"),
                      layers = cfg_.at("layers"), heads = cfg_.at("kv_heads"),
                      dim = cfg_.at("head_dim"), eos = cfg_.at("eos_id");
        if (length < 2 || vocab <= 0 || layers <= 0 || heads <= 0 || dim <= 0 || eos < 0 ||
            eos >= vocab)
            throw std::runtime_error("Invalid Qwen2.5 Omni Thinker profile");
        require_tensor(*engine_, "input_ids", {1}, DType::kInt32, false);
        require_tensor(*engine_, "position", {1}, DType::kInt32, false);
        require_tensor(*engine_, "logits", {1, vocab}, DType::kFloat32, true);
        for (int32_t i = 0; i < layers; ++i) {
            const auto suffix = std::to_string(i);
            for (const auto* prefix : {"cache_k_", "cache_v_", "present_k_", "present_v_"})
                require_tensor(*engine_, prefix + suffix, {1, heads, length, dim}, DType::kFloat32,
                               prefix[0] == 'p');
        }
    }
    const char* task() const noexcept override { return "text_continuation"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {
            internal::bind<internal::ITextContinuation>(*this, {fields_.data(), fields_.size()})};
    }
    internal::TextResult run(const internal::TextContinuationRequest& request,
                             internal::ConfigView config) override {
        const Span<const internal::ConfigField> fields{fields_.data(), fields_.size()};
        internal::validate_config(fields, config);
        const auto requested_max = internal::config_get<int64_t>(config, fields, "max_new_tokens");
        if (*internal::config_get<bool>(config, fields, "do_sample"))
            throw std::invalid_argument("Qwen2.5 Omni Thinker currently supports greedy decoding");
        const int32_t length = cfg_.at("sequence_length"), vocab = cfg_.at("vocab_size");
        const int32_t layers = cfg_.at("layers"), heads = cfg_.at("kv_heads"),
                      dim = cfg_.at("head_dim");
        std::vector<int32_t> ids;
        if (const auto* text = std::get_if<std::string_view>(&request.prefix))
            ids = tokenizer_->encode(std::string(*text));
        else {
            const auto tokens = std::get<Span<const int32_t>>(request.prefix);
            ids.assign(tokens.begin(), tokens.end());
        }
        if (ids.empty() || ids.size() >= static_cast<std::size_t>(length))
            throw std::invalid_argument("Qwen2.5 Omni Thinker prefix does not fit its KV profile");
        const auto remaining = static_cast<int64_t>(length - ids.size());
        const auto max_new = requested_max.value_or(std::min<int64_t>(32, remaining));
        if (max_new <= 0 || max_new > remaining)
            throw std::invalid_argument(
                "Qwen2.5 Omni Thinker prefix and generation exceed the built KV profile");
        for (const auto id : ids)
            if (id < 0 || id >= vocab)
                throw std::invalid_argument("Invalid Qwen2.5 Omni Thinker token ID");
        const auto stream = engine_->stream();
        const auto cache_count = static_cast<std::size_t>(heads) * length * dim;
        std::vector<DeviceTensor> keys, values;
        keys.reserve(layers);
        values.reserve(layers);
        for (int32_t i = 0; i < layers; ++i) {
            keys.push_back(DeviceTensor::zeros({1, heads, length, dim}, DType::kFloat32, stream));
            values.push_back(DeviceTensor::zeros({1, heads, length, dim}, DType::kFloat32, stream));
            if (!keys.back().ok() || !values.back().ok())
                throw std::runtime_error("Qwen2.5 Omni Thinker could not allocate device KV state");
        }
        DeviceTensor token_input({1}, DType::kInt32, stream),
            position_input({1}, DType::kInt32, stream);
        std::vector<float> logits(vocab);
        int32_t position = 0;
        auto step = [&](int32_t token) {
            if (position >= length)
                throw std::runtime_error("Qwen2.5 Omni Thinker exhausted its KV profile");
            if (!token_input.copy_from_host(&token) || !position_input.copy_from_host(&position))
                throw std::runtime_error("Qwen2.5 Omni Thinker could not upload token inputs");
            DeviceTensorMap inputs{{"input_ids", &token_input}, {"position", &position_input}};
            for (int32_t i = 0; i < layers; ++i) {
                inputs.emplace("cache_k_" + std::to_string(i), &keys[i]);
                inputs.emplace("cache_v_" + std::to_string(i), &values[i]);
            }
            engine_->forward_device_async(inputs);
            check_copy(cudaMemcpyAsync(logits.data(), engine_->device_ptr("logits"),
                                       logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                       stream));
            for (int32_t i = 0; i < layers; ++i) {
                const auto suffix = std::to_string(i);
                check_copy(
                    cudaMemcpyAsync(keys[i].data(), engine_->device_ptr("present_k_" + suffix),
                                    cache_count * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                check_copy(
                    cudaMemcpyAsync(values[i].data(), engine_->device_ptr("present_v_" + suffix),
                                    cache_count * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            }
            engine_->sync();
            for (const auto value : logits)
                if (!std::isfinite(value))
                    throw std::runtime_error("Qwen2.5 Omni Thinker logit is nonfinite");
            ++position;
        };
        using Clock = std::chrono::steady_clock;
        const auto prefill_start = Clock::now();
        for (const auto token : ids)
            step(token);
        const auto decode_start = Clock::now();
        const int32_t eos = cfg_.at("eos_id");
        internal::TextResult result;
        result.prefill_ms =
            std::chrono::duration<double, std::milli>(decode_start - prefill_start).count();
        for (int64_t i = 0; i < max_new; ++i) {
            const auto next = static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) -
                                                   logits.begin());
            result.token_ids.push_back(next);
            if (next == eos)
                break;
            if (i + 1 < max_new)
                step(next);
        }
        result.decode_ms =
            std::chrono::duration<double, std::milli>(Clock::now() - decode_start).count();
        result.text = tokenizer_->decode(result.token_ids);
        return result;
    }

  private:
    std::unique_ptr<ITrtModule> engine_;
    std::unique_ptr<ITokenizer> tokenizer_;
    nlohmann::json cfg_;
    std::vector<internal::ConfigField> fields_;
};
} // namespace
} // namespace trtmc::qwen2_5_omni

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("qwen2_5_omni does not support --kv-cache-size");
    using namespace trtmc;
    if (context.reader.info().task != "text_continuation")
        throw std::invalid_argument(
            "Qwen2.5 Omni Thinker uses a family-owned fixed KV profile for text continuation");
    const auto raw = qwen2_5_omni::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    const int32_t length = cfg.at("sequence_length"), layers = cfg.at("layers");
    const int32_t heads = cfg.at("kv_heads"), dim = cfg.at("head_dim"),
                  vocab = cfg.at("vocab_size");
    const int32_t eos = cfg.at("eos_id");
    if (length <= 0 || length > 1024 || layers <= 0 || heads <= 0 || dim <= 0 || vocab <= 0 ||
        eos < 0 || eos >= vocab)
        throw std::runtime_error("Invalid Qwen2.5 Omni Thinker runtime dimensions");
    const auto plan = qwen2_5_omni::section(context.reader, "model.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok())
        throw std::runtime_error("Qwen2.5 Omni Thinker engine failed to load");
    const auto raw_tokenizer = qwen2_5_omni::section(context.reader, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(raw_tokenizer.data(), raw_tokenizer.size(), false);
    return new qwen2_5_omni::Pipeline(std::move(engine), std::move(tokenizer), cfg);
}

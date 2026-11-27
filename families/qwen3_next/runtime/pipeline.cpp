/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/qwen3_next/runtime/diagnostics.h"
#include "families/qwen3_next/runtime/routing.h"
#include "families/qwen3_next/runtime/sampler.h"
#include "families/qwen3_next/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::qwen3_next {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Cache {
    std::vector<std::uint16_t> k, v, conv;
    std::vector<float> recurrent;
    std::int64_t length{1};
    bool dummy{true};
};

class Pipeline final : public ITextGeneration, public Diagnostics {
  public:
    explicit Pipeline(const FamilyContext& context) {
        const auto raw = context.reader.read_section("runtime.json");
        config_ = nlohmann::json::parse(raw.begin(), raw.end());
        if (config_.value("routing_contract", "") != "reference_topk_retry_v1")
            throw std::runtime_error("Qwen3Next bundle has an incompatible routing contract");
        experts_ = config_.at("num_experts");
        experts_per_token_ = config_.at("num_experts_per_tok");
        hidden_ = config_.at("hidden_size");
        heads_ = config_.at("num_key_value_heads");
        dim_ = config_.at("head_dim");
        vocab_ = config_.at("vocab_size");
        limit_ = config_.at("max_sequence_length");
        eos_ = {config_.at("eos_token_id").get<std::int32_t>()};
        conv_dim_ = config_.at("conv_dim");
        history_ = config_.at("linear_conv_kernel_dim").get<int>() - 1;
        linear_heads_ = config_.at("linear_num_value_heads");
        key_dim_ = config_.at("linear_key_head_dim");
        value_dim_ = config_.at("linear_value_head_dim");
        types_ = config_.at("layer_types").get<std::vector<std::string>>();
        if (hidden_ <= 0 || heads_ <= 0 || dim_ <= 0 || vocab_ <= 0 || limit_ <= 0 ||
            types_.empty() || eos_.empty() || conv_dim_ <= 0 || history_ <= 0 ||
            linear_heads_ <= 0 || key_dim_ <= 0 || value_dim_ <= 0)
            throw std::runtime_error("invalid Qwen3Next runtime dimensions");
        embedding_ = context.reader.read_section("embedding.bf16");
        if (embedding_.size() != static_cast<std::size_t>(hidden_) * vocab_ * 2)
            throw std::runtime_error("invalid Qwen3Next embedding section");
        const auto tokenizer = context.reader.read_section("tokenizer.json");
        tokenizer_ = CreateBpeTokenizer(tokenizer.data(), tokenizer.size(), false);
        auto load = [&](const std::string& name) {
            auto plan = context.reader.read_section(name);
            auto module = context.backend.create_module(plan.data(), plan.size(), {});
            if (!module || !module->ok())
                throw std::runtime_error("cannot load Qwen3Next engine " + name);
            return module;
        };
        for (std::size_t i = 0; i < types_.size(); ++i)
            layers_.push_back(load("layers/" + std::to_string(i) + ".plan"));
        head_ = load("head.plan");
    }

    std::int32_t default_max_new_tokens() const override { return 128; }

    LogitTrace trace_logits(const std::string& prompt, std::int32_t steps) override {
        LogitTrace trace;
        trace.prompt_ids = tokenizer_->encode(prompt);
        if (steps < 1 || trace.prompt_ids.empty() ||
            trace.prompt_ids.size() + static_cast<std::size_t>(steps) >
                static_cast<std::size_t>(limit_))
            throw std::invalid_argument("invalid Qwen3Next logit trace request");
        reset();
        std::vector<float> logits;
        for (auto token : trace.prompt_ids)
            logits = step(token);
        for (int i = 0; i < steps; ++i) {
            trace.rows.push_back(logits);
            auto token = static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) -
                                                   logits.begin());
            trace.generated_ids.push_back(token);
            if (std::find(eos_.begin(), eos_.end(), token) != eos_.end())
                break;
            if (i + 1 < steps)
                logits = step(token);
        }
        trace.routing_retries = routing_retries_;
        return trace;
    }

    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg) override {
        if (cfg.max_new_tokens < 0)
            throw std::invalid_argument("negative Qwen3Next output budget");
        if (cfg.repetition_penalty != 1.0F)
            throw std::invalid_argument("Qwen3Next does not support repetition penalties");
        if (!cfg.lora_adapter_id.empty() || cfg.stop_on_boxed_answer)
            throw std::invalid_argument("unsupported Qwen3Next adapter/stop condition");
        std::string text = prompt;
        if (cfg.use_chat_template) {
            text = cfg.system_prompt.empty()
                       ? ""
                       : "<|im_start|>system\n" + cfg.system_prompt + "<|im_end|>\n";
            text += "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
        } else if (!cfg.system_prompt.empty()) {
            throw std::invalid_argument("Qwen3Next system_prompt requires the chat template");
        }
        auto ids = tokenizer_->encode(text);
        if (ids.empty())
            throw std::invalid_argument("Qwen3Next requires a non-empty prompt");
        if (ids.size() + static_cast<std::size_t>(cfg.max_new_tokens) >
            static_cast<std::size_t>(limit_))
            throw std::invalid_argument("Qwen3Next request exceeds the built context capacity");
        reset();
        TextResult result;
        if (cfg.max_new_tokens == 0)
            return result;
        auto params = qwen3_next_sampling_params_from_config(cfg, eos_.front());
        auto sampler = create_qwen3_next_sampler(params);
        const auto start = Clock::now();
        std::vector<float> logits;
        for (auto id : ids)
            logits = step(id);
        result.prefill_ms = elapsed(start);
        const auto decode_start = Clock::now();
        for (int i = 0; i < cfg.max_new_tokens; ++i) {
            const auto selected = sampler->sample(logits.data(), vocab_, params).token_id;
            result.token_ids.push_back(selected);
            const bool stop = cfg.eos_token_id >= 0
                                  ? selected == cfg.eos_token_id
                                  : std::find(eos_.begin(), eos_.end(), selected) != eos_.end();
            if (stop || i + 1 == cfg.max_new_tokens)
                break;
            logits = step(selected);
        }
        result.decode_ms = elapsed(decode_start);
        result.text = tokenizer_->decode(result.token_ids);
        return result;
    }

  private:
    void reset() {
        position_ = 0;
        routing_retries_ = 0;
        caches_.assign(layers_.size(), {});
        for (std::size_t i = 0; i < caches_.size(); ++i) {
            auto& cache = caches_[i];
            if (types_[i] == "linear_attention") {
                cache.conv.assign(static_cast<std::size_t>(conv_dim_) * history_, 0);
                cache.recurrent.assign(
                    static_cast<std::size_t>(linear_heads_) * key_dim_ * value_dim_, 0.0F);
            } else {
                cache.k.assign(static_cast<std::size_t>(heads_) * dim_, 0);
                cache.v.assign(static_cast<std::size_t>(heads_) * dim_, 0);
            }
        }
    }

    TensorMap route_layer(ITrtModule& layer, TensorMap inputs) {
        std::int32_t force = 0;
        std::vector<std::int32_t> forced(static_cast<std::size_t>(experts_per_token_), 0);
        inputs["forced_experts"] = {forced.data(), {experts_per_token_}, DType::kInt32};
        inputs["force_routing"] = {&force, {1}, DType::kInt32};
        auto outputs = layer.forward(inputs);
        const auto* probabilities =
            static_cast<const float*>(outputs.at("router_probabilities").data);
        forced = select_experts(probabilities, experts_, experts_per_token_);
        const auto matches = [&] {
            const auto* actual =
                static_cast<const std::int32_t*>(outputs.at("selected_experts").data);
            return std::equal(forced.begin(), forced.end(), actual);
        };
        if (!matches()) {
            force = 1;
            ++routing_retries_;
            inputs.at("forced_experts").data = forced.data();
            // The first forward is speculative: original hidden and cache inputs
            // are still intact. Only the final forward may update sequence state.
            outputs = layer.forward(inputs);
            if (!matches())
                throw std::runtime_error("Qwen3Next forced expert routing was not applied");
        }
        return outputs;
    }

    std::vector<float> step(std::int32_t token) {
        if (token < 0 || token >= vocab_ || position_ >= limit_)
            throw std::out_of_range("Qwen3Next token or position outside model capacity");
        std::vector<std::uint16_t> hidden(hidden_);
        std::memcpy(hidden.data(),
                    embedding_.data() + static_cast<std::size_t>(token) * hidden_ * 2, hidden_ * 2);
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            auto& cache = caches_[i];
            TensorMap inputs;
            inputs["hidden"] = {hidden.data(), {1, hidden_}, DType::kBFloat16};
            if (types_[i] == "linear_attention") {
                inputs["conv_state"] = {cache.conv.data(), {conv_dim_, history_}, DType::kBFloat16};
                inputs["recurrent_state"] = {
                    cache.recurrent.data(), {linear_heads_, key_dim_, value_dim_}, DType::kFloat32};
                const auto outputs = route_layer(*layers_[i], inputs);
                std::memcpy(hidden.data(), outputs.at("hidden_out").data, hidden_ * 2);
                std::memcpy(cache.conv.data(), outputs.at("present_conv").data,
                            cache.conv.size() * 2);
                std::memcpy(cache.recurrent.data(), outputs.at("present_recurrent").data,
                            cache.recurrent.size() * sizeof(float));
            } else {
                std::vector<float> mask(static_cast<std::size_t>(cache.length) + 1, 0.0F);
                if (cache.dummy)
                    mask.front() = -1e30F;
                inputs["position"] = {&position_, {1}, DType::kInt32};
                inputs["past_k"] = {cache.k.data(), {heads_, cache.length, dim_}, DType::kBFloat16};
                inputs["past_v"] = {cache.v.data(), {heads_, cache.length, dim_}, DType::kBFloat16};
                inputs["mask"] = {mask.data(), {1, 1, cache.length + 1}, DType::kFloat32};
                const auto outputs = route_layer(*layers_[i], inputs);
                std::memcpy(hidden.data(), outputs.at("hidden_out").data, hidden_ * 2);
                ++cache.length;
                cache.k.resize(static_cast<std::size_t>(heads_) * cache.length * dim_);
                cache.v.resize(cache.k.size());
                std::memcpy(cache.k.data(), outputs.at("present_k").data, cache.k.size() * 2);
                std::memcpy(cache.v.data(), outputs.at("present_v").data, cache.v.size() * 2);
            }
        }
        ++position_;
        TensorMap inputs;
        inputs["hidden"] = {hidden.data(), {1, hidden_}, DType::kBFloat16};
        const auto output = head_->forward(inputs).at("logits");
        const auto* data = static_cast<const float*>(output.data);
        return {data, data + vocab_};
    }

    nlohmann::json config_;
    std::uint64_t routing_retries_{0};
    std::int32_t experts_, experts_per_token_;
    std::int32_t hidden_, heads_, dim_, vocab_, limit_, position_{0};
    std::int32_t conv_dim_, history_, linear_heads_, key_dim_, value_dim_;
    std::vector<std::int32_t> eos_;
    std::vector<std::string> types_;
    std::vector<char> embedding_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::vector<std::unique_ptr<ITrtModule>> layers_;
    std::unique_ptr<ITrtModule> head_;
    std::vector<Cache> caches_;
};
} // namespace

ITask* create_pipeline(const FamilyContext& context) {
    return new Pipeline(context);
}
} // namespace trtmc::qwen3_next

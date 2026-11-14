/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/flint/runtime/diagnostics.h"
#include "families/flint/runtime/sampler.h"
#include "families/flint/runtime/tokenizer.h"
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

namespace trtmc::flint {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::string clean_text(std::string text) {
    text.erase(std::remove(text.begin(), text.end(), '\0'), text.end());
    for (const std::string marker : {"<think>", "</think>"}) {
        for (auto pos = text.find(marker); pos != std::string::npos; pos = text.find(marker))
            text.erase(pos, marker.size());
    }
    for (const std::string role : {"begin_of_text", "end_of_text", "system", "developer", "user",
                                   "assistant", "observation"}) {
        const auto marker = "<|" + role + "|>";
        const auto replacement = "&lt;|" + role + "|&gt;";
        for (auto pos = text.find(marker); pos != std::string::npos;
             pos = text.find(marker, pos + replacement.size()))
            text.replace(pos, marker.size(), replacement);
    }
    return text;
}

struct Cache {
    std::vector<std::uint16_t> k, v;
    std::int64_t length{1};
    bool dummy{true};
};

class Pipeline final : public ITextGeneration, public Diagnostics {
  public:
    explicit Pipeline(const FamilyContext& context) {
        const auto raw = context.reader.read_section("runtime.json");
        config_ = nlohmann::json::parse(raw.begin(), raw.end());
        hidden_ = config_.at("hidden_size");
        heads_ = config_.at("num_key_value_heads");
        dim_ = config_.at("head_dim");
        vocab_ = config_.at("vocab_size");
        limit_ = config_.at("max_sequence_length");
        eos_ = config_.at("eos_token_id").get<std::vector<std::int32_t>>();
        types_ = config_.at("layer_types").get<std::vector<std::string>>();
        if (hidden_ <= 0 || heads_ <= 0 || dim_ <= 0 || vocab_ <= 0 || limit_ <= 0 ||
            types_.empty() || eos_.empty())
            throw std::runtime_error("invalid Flint runtime dimensions");
        embedding_ = context.reader.read_section("embedding.bf16");
        if (embedding_.size() != static_cast<std::size_t>(hidden_) * vocab_ * 2)
            throw std::runtime_error("invalid Flint embedding section");
        const auto tokenizer = context.reader.read_section("tokenizer.json");
        tokenizer_ = CreateBpeTokenizer(tokenizer.data(), tokenizer.size(), false);
        auto load = [&](const std::string& name) {
            auto plan = context.reader.read_section(name);
            auto module = context.backend.create_module(plan.data(), plan.size(), {});
            if (!module || !module->ok())
                throw std::runtime_error("cannot load Flint engine " + name);
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
            throw std::invalid_argument("invalid Flint logit trace request");
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
        return trace;
    }

    TextResult generate(const std::string& prompt, const TextGenerationConfig& cfg) override {
        if (cfg.max_new_tokens < 0)
            throw std::invalid_argument("negative Flint output budget");
        if (!cfg.lora_adapter_id.empty() || cfg.stop_on_boxed_answer)
            throw std::invalid_argument("unsupported Flint adapter/stop condition");
        std::string text = prompt;
        if (cfg.use_chat_template) {
            text = cfg.system_prompt.empty() ? "" : "<|system|>" + clean_text(cfg.system_prompt);
            text += "<|user|>" + clean_text(prompt) + "<|assistant|>";
        } else if (!cfg.system_prompt.empty()) {
            throw std::invalid_argument("Flint system_prompt requires the chat template");
        }
        auto ids = tokenizer_->encode(text);
        if (ids.empty())
            throw std::invalid_argument("Flint requires a non-empty prompt");
        if (ids.size() + static_cast<std::size_t>(cfg.max_new_tokens) >
            static_cast<std::size_t>(limit_))
            throw std::invalid_argument("Flint request exceeds the built context capacity");
        reset();
        TextResult result;
        if (cfg.max_new_tokens == 0)
            return result;
        auto params = flint_sampling_params_from_config(cfg, eos_.front());
        auto sampler = create_flint_sampler(params);
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
        caches_.assign(layers_.size(), {});
        for (auto& cache : caches_) {
            cache.k.assign(static_cast<std::size_t>(heads_) * dim_, 0);
            cache.v.assign(static_cast<std::size_t>(heads_) * dim_, 0);
        }
    }

    void save_cache(std::vector<std::uint16_t>& target, const Tensor& output, std::int64_t length,
                    std::int64_t keep) {
        target.resize(static_cast<std::size_t>(heads_) * keep * dim_);
        const auto* source = static_cast<const std::uint16_t*>(output.data);
        for (std::int64_t h = 0; h < heads_; ++h)
            std::memcpy(target.data() + h * keep * dim_,
                        source + (h * length + length - keep) * dim_,
                        static_cast<std::size_t>(keep) * dim_ * 2);
    }

    std::vector<float> step(std::int32_t token) {
        if (token < 0 || token >= vocab_ || position_ >= limit_)
            throw std::out_of_range("Flint token or position outside model capacity");
        std::vector<std::uint16_t> hidden(hidden_);
        std::memcpy(hidden.data(),
                    embedding_.data() + static_cast<std::size_t>(token) * hidden_ * 2, hidden_ * 2);
        for (std::size_t i = 0; i < layers_.size(); ++i) {
            auto& cache = caches_[i];
            std::vector<float> mask(static_cast<std::size_t>(cache.length) + 1, 0.0F);
            if (cache.dummy)
                mask.front() = -1e30F;
            TensorMap inputs;
            inputs["hidden"] = {hidden.data(), {1, hidden_}, DType::kBFloat16};
            inputs["position"] = {&position_, {1}, DType::kInt32};
            inputs["past_k"] = {cache.k.data(), {heads_, cache.length, dim_}, DType::kBFloat16};
            inputs["past_v"] = {cache.v.data(), {heads_, cache.length, dim_}, DType::kBFloat16};
            inputs["mask"] = {mask.data(), {1, 1, cache.length + 1}, DType::kFloat32};
            const auto outputs = layers_[i]->forward(inputs);
            std::memcpy(hidden.data(), outputs.at("hidden_out").data, hidden_ * 2);
            const auto length = cache.length + 1;
            // Keep the last window-1 real tokens for the next sliding step.
            const std::int64_t keep =
                types_[i] == "sliding_attention"
                    ? std::min<std::int64_t>(length, config_.at("sliding_window").get<int>() - 1)
                    : length;
            save_cache(cache.k, outputs.at("present_k"), length, keep);
            save_cache(cache.v, outputs.at("present_v"), length, keep);
            cache.dummy = cache.dummy && keep == length;
            cache.length = keep;
        }
        ++position_;
        TensorMap inputs;
        inputs["hidden"] = {hidden.data(), {1, hidden_}, DType::kBFloat16};
        const auto output = head_->forward(inputs).at("logits");
        const auto* data = static_cast<const float*>(output.data);
        return {data, data + vocab_};
    }

    nlohmann::json config_;
    std::int32_t hidden_, heads_, dim_, vocab_, limit_, position_{0};
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
} // namespace trtmc::flint

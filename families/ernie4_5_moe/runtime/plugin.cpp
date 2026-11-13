/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/ernie4_5_moe/runtime/tokenizer.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::ernie4_5_moe {
namespace {
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || found->length == 0)
        throw std::runtime_error(std::string("ERNIE bundle lacks ") + name);
    return reader.read_section(name);
}
Tensor tensor(void* data, std::initializer_list<int64_t> shape, DType dtype) {
    Tensor result;
    result.data = data;
    result.shape = shape;
    result.dtype = dtype;
    return result;
}
const Tensor& output(const TensorMap& outputs, const std::string& name, std::size_t count) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || !found->second.data || found->second.dtype != DType::kFloat32 ||
        found->second.numel() != count)
        throw std::runtime_error("Invalid ERNIE output " + name);
    return found->second;
}

class Pipeline final : public internal::IModel, public internal::ITextContinuation {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, std::unique_ptr<ITokenizer> tokenizer,
             nlohmann::json cfg)
        : engine_(std::move(engine)), tokenizer_(std::move(tokenizer)), cfg_(std::move(cfg)),
          fields_{{"max_new_tokens", internal::ConfigKind::I64, internal::ConfigValue{int64_t{32}},
                   "Maximum generated tokens"},
                  {"do_sample", internal::ConfigKind::Bool, internal::ConfigValue{false},
                   "Greedy decoding only"}} {}
    const char* task() const noexcept override { return "text_continuation"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {
            internal::bind<internal::ITextContinuation>(*this, {fields_.data(), fields_.size()})};
    }
    internal::TextResult run(const internal::TextContinuationRequest& request,
                             internal::ConfigView config) override {
        const Span<const internal::ConfigField> fields{fields_.data(), fields_.size()};
        internal::validate_config(fields, config);
        const auto max_new = *internal::config_get<int64_t>(config, fields, "max_new_tokens");
        if (*internal::config_get<bool>(config, fields, "do_sample"))
            throw std::invalid_argument("ERNIE currently supports greedy decoding");
        const int32_t length = cfg_.at("sequence_length"), vocab = cfg_.at("vocab_size");
        const int32_t layers = cfg_.at("layers"), heads = cfg_.at("kv_heads"),
                      dim = cfg_.at("head_dim");
        if (max_new <= 0 || max_new > length)
            throw std::invalid_argument("Invalid ERNIE generation length");
        std::vector<int32_t> ids;
        if (const auto* text = std::get_if<std::string_view>(&request.prefix))
            ids = tokenizer_->encode(std::string(*text));
        else {
            const auto tokens = std::get<Span<const int32_t>>(request.prefix);
            ids.assign(tokens.begin(), tokens.end());
        }
        if (ids.empty() || ids.size() + max_new > static_cast<std::size_t>(length))
            throw std::invalid_argument("ERNIE prefix and generation exceed the built KV profile");
        for (const auto id : ids)
            if (id < 0 || id >= vocab)
                throw std::invalid_argument("Invalid ERNIE token ID");
        const auto cache_count = static_cast<std::size_t>(heads) * length * dim;
        std::vector<std::vector<float>> keys(layers, std::vector<float>(cache_count, 0));
        std::vector<std::vector<float>> values(layers, std::vector<float>(cache_count, 0));
        std::vector<float> logits;
        int32_t position = 0;
        auto step = [&](int32_t token) {
            if (position >= length)
                throw std::runtime_error("ERNIE exhausted its KV profile");
            TensorMap inputs{{"input_ids", tensor(&token, {1}, DType::kInt32)},
                             {"position", tensor(&position, {1}, DType::kInt32)}};
            for (int32_t i = 0; i < layers; ++i) {
                inputs.emplace("cache_k_" + std::to_string(i),
                               tensor(keys[i].data(), {1, heads, length, dim}, DType::kFloat32));
                inputs.emplace("cache_v_" + std::to_string(i),
                               tensor(values[i].data(), {1, heads, length, dim}, DType::kFloat32));
            }
            const auto predicted = engine_->forward(inputs);
            const auto& scores = output(predicted, "logits", vocab);
            const auto* data = static_cast<const float*>(scores.data);
            logits.assign(data, data + vocab);
            for (const auto value : logits)
                if (!std::isfinite(value))
                    throw std::runtime_error("ERNIE logit is nonfinite");
            for (int32_t i = 0; i < layers; ++i) {
                const auto& key = output(predicted, "present_k_" + std::to_string(i), cache_count);
                const auto& value =
                    output(predicted, "present_v_" + std::to_string(i), cache_count);
                std::copy_n(static_cast<const float*>(key.data), cache_count, keys[i].data());
                std::copy_n(static_cast<const float*>(value.data), cache_count, values[i].data());
            }
            ++position;
        };
        for (const auto token : ids)
            step(token);
        const int32_t eos = cfg_.at("eos_id");
        internal::TextResult result;
        for (int64_t i = 0; i < max_new; ++i) {
            const auto next = static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) -
                                                   logits.begin());
            result.token_ids.push_back(next);
            if (next == eos)
                break;
            if (i + 1 < max_new)
                step(next);
        }
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
} // namespace trtmc::ernie4_5_moe

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    if (context.kv_cache_size_bytes || context.reader.info().task != "text_continuation")
        throw std::invalid_argument(
            "ERNIE uses a family-owned fixed KV profile for text continuation");
    const auto raw = ernie4_5_moe::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    const int32_t length = cfg.at("sequence_length"), layers = cfg.at("layers");
    const int32_t heads = cfg.at("kv_heads"), dim = cfg.at("head_dim"),
                  vocab = cfg.at("vocab_size");
    const int32_t eos = cfg.at("eos_id");
    if (length <= 0 || length > 1024 || layers <= 0 || heads <= 0 || dim <= 0 || vocab <= 0 ||
        eos < 0 || eos >= vocab)
        throw std::runtime_error("Invalid ERNIE runtime dimensions");
    const auto plan = ernie4_5_moe::section(context.reader, "model.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok())
        throw std::runtime_error("ERNIE engine failed to load");
    const auto raw_tokenizer = ernie4_5_moe::section(context.reader, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(raw_tokenizer.data(), raw_tokenizer.size(), true);
    return new ernie4_5_moe::Pipeline(std::move(engine), std::move(tokenizer), cfg);
}

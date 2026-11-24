/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/kolibri1/runtime/tokenizer.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::kolibri1 {
namespace {
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || found->length == 0)
        throw std::runtime_error(std::string("Kolibri1 bundle lacks ") + name);
    return reader.read_section(name);
}
Tensor tensor(void* data, std::initializer_list<int64_t> shape, DType dtype) {
    Tensor result;
    result.data = data;
    result.shape = shape;
    result.dtype = dtype;
    return result;
}
Tensor output(const TensorMap& outputs, const std::string& name, std::size_t count,
              DType dtype = DType::kFloat32) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || !found->second.data || found->second.dtype != dtype ||
        found->second.numel() != count)
        throw std::runtime_error("Invalid Kolibri1 output " + name);
    return found->second;
}

class Pipeline final : public internal::IModel, public internal::ITextContinuation {
  public:
    Pipeline(IBackend& backend, std::unique_ptr<ITrtModule> embedding,
             std::vector<std::unique_ptr<ITrtModule>> layers,
             std::vector<std::vector<char>> streamed_plans, std::unique_ptr<ITrtModule> head,
             std::unique_ptr<ITokenizer> tokenizer, nlohmann::json cfg)
        : backend_(backend), embedding_(std::move(embedding)), layers_(std::move(layers)),
          streamed_plans_(std::move(streamed_plans)), head_(std::move(head)),
          tokenizer_(std::move(tokenizer)), cfg_(std::move(cfg)),
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
            throw std::invalid_argument("Kolibri1 currently supports greedy decoding");
        const int32_t length = cfg_.at("sequence_length"), vocab = cfg_.at("vocab_size");
        const int32_t layers = cfg_.at("layers"), heads = cfg_.at("kv_heads"),
                      dim = cfg_.at("head_dim");
        if (max_new <= 0 || max_new > length)
            throw std::invalid_argument("Invalid Kolibri1 generation length");
        std::vector<int32_t> ids;
        if (const auto* text = std::get_if<std::string_view>(&request.prefix))
            ids = tokenizer_->encode(std::string(*text));
        else {
            const auto tokens = std::get<Span<const int32_t>>(request.prefix);
            ids.assign(tokens.begin(), tokens.end());
        }
        if (ids.empty() || ids.size() + max_new > static_cast<std::size_t>(length))
            throw std::invalid_argument(
                "Kolibri1 prefix and generation exceed the built KV profile");
        for (const auto id : ids)
            if (id < 0 || id >= vocab)
                throw std::invalid_argument("Invalid Kolibri1 token ID");
        const int32_t width = cfg_.at("hidden_size");
        const auto cache_count = static_cast<std::size_t>(heads) * length * dim;
        std::vector<std::vector<uint16_t>> keys(layers, std::vector<uint16_t>(cache_count, 0));
        std::vector<std::vector<uint16_t>> values(layers, std::vector<uint16_t>(cache_count, 0));
        std::vector<uint16_t> hidden(width);
        std::vector<float> logits;
        int32_t position = 0;
        auto step = [&](int32_t token) {
            if (position >= length)
                throw std::runtime_error("Kolibri1 exhausted its KV profile");
            auto embedded =
                embedding_->forward({{"input_ids", tensor(&token, {1}, DType::kInt32)}});
            const auto& initial = output(embedded, "result", width, DType::kFloat16);
            std::copy_n(static_cast<const uint16_t*>(initial.data), width, hidden.data());
            for (int32_t i = 0; i < layers; ++i) {
                TensorMap inputs{
                    {"hidden", tensor(hidden.data(), {1, width}, DType::kFloat16)},
                    {"position", tensor(&position, {1}, DType::kInt32)},
                    {"cache_k", tensor(keys[i].data(), {1, heads, length, dim}, DType::kFloat16)},
                    {"cache_v",
                     tensor(values[i].data(), {1, heads, length, dim}, DType::kFloat16)}};
                std::unique_ptr<ITrtModule> transient;
                ITrtModule* layer = nullptr;
                if (static_cast<std::size_t>(i) < layers_.size()) {
                    layer = layers_[i].get();
                } else {
                    const auto& plan = streamed_plans_.at(i - layers_.size());
                    transient = backend_.create_module(plan.data(), plan.size(), {});
                    if (!transient || !transient->ok())
                        throw std::runtime_error("Kolibri1 streamed engine failed to load");
                    layer = transient.get();
                }
                const auto predicted = layer->forward(inputs);
                const auto& result = output(predicted, "result", width, DType::kFloat16);
                const auto& key = output(predicted, "present_k", cache_count, DType::kFloat16);
                const auto& value = output(predicted, "present_v", cache_count, DType::kFloat16);
                std::copy_n(static_cast<const uint16_t*>(result.data), width, hidden.data());
                std::copy_n(static_cast<const uint16_t*>(key.data), cache_count, keys[i].data());
                std::copy_n(static_cast<const uint16_t*>(value.data), cache_count,
                            values[i].data());
            }
            const auto predicted =
                head_->forward({{"hidden", tensor(hidden.data(), {1, width}, DType::kFloat16)}});
            const auto& scores = output(predicted, "logits", vocab);
            const auto* data = static_cast<const float*>(scores.data);
            logits.assign(data, data + vocab);
            for (const auto value : logits)
                if (!std::isfinite(value))
                    throw std::runtime_error("Kolibri1 logit is nonfinite");
            ++position;
        };
        for (const auto token : ids)
            step(token);
        const auto eos = cfg_.at("eos_ids").get<std::vector<int32_t>>();
        internal::TextResult result;
        for (int64_t i = 0; i < max_new; ++i) {
            const auto next = static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) -
                                                   logits.begin());
            result.token_ids.push_back(next);
            if (std::find(eos.begin(), eos.end(), next) != eos.end())
                break;
            if (i + 1 < max_new)
                step(next);
        }
        result.text = tokenizer_->decode(result.token_ids);
        return result;
    }

  private:
    IBackend& backend_;
    std::unique_ptr<ITrtModule> embedding_;
    std::vector<std::unique_ptr<ITrtModule>> layers_;
    std::vector<std::vector<char>> streamed_plans_;
    std::unique_ptr<ITrtModule> head_;
    std::unique_ptr<ITokenizer> tokenizer_;
    nlohmann::json cfg_;
    std::vector<internal::ConfigField> fields_;
};
} // namespace
} // namespace trtmc::kolibri1

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("kolibri1 does not support --kv-cache-size");
    using namespace trtmc;
    if (context.reader.info().task != "text_continuation")
        throw std::invalid_argument(
            "Kolibri1 uses a family-owned fixed KV profile for text continuation");
    const auto raw = kolibri1::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    const int32_t length = cfg.at("sequence_length"), layers = cfg.at("layers");
    const int32_t heads = cfg.at("kv_heads"), dim = cfg.at("head_dim"),
                  vocab = cfg.at("vocab_size");
    const int32_t width = cfg.at("hidden_size");
    const auto eos = cfg.at("eos_ids").get<std::vector<int32_t>>();
    if (length <= 0 || length > 513 || layers <= 0 || layers > 128 || heads <= 0 || heads > 128 ||
        dim <= 0 || dim > 1024 || width <= 0 || width > 65536 || vocab <= 0 || eos.empty())
        throw std::runtime_error("Invalid Kolibri1 runtime dimensions");
    for (auto id : eos)
        if (id < 0 || id >= vocab)
            throw std::runtime_error("Invalid Kolibri1 EOS token");
    auto load = [&](const std::string& name) {
        const auto plan = kolibri1::section(context.reader, name.c_str());
        auto engine = context.backend.create_module(plan.data(), plan.size(), {});
        if (!engine || !engine->ok())
            throw std::runtime_error("Kolibri1 engine failed to load: " + name);
        return engine;
    };
    // Keep half the decoder engines resident and retain the remaining serialized
    // plans on the host. Stream one transient engine at a time without rereading
    // its large plan for every token. KV state remains explicit and family-owned.
    auto embedding = load("embedding.plan");
    std::vector<std::unique_ptr<ITrtModule>> engines;
    std::vector<std::vector<char>> streamed_plans;
    const int32_t resident_layers = (layers + 1) / 2;
    for (int32_t i = 0; i < layers; ++i) {
        const auto name = "layer." + std::to_string(i) + ".plan";
        if (i < resident_layers)
            engines.push_back(load(name));
        else
            streamed_plans.push_back(kolibri1::section(context.reader, name.c_str()));
    }
    auto head = load("head.plan");
    const auto raw_tokenizer = kolibri1::section(context.reader, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(raw_tokenizer.data(), raw_tokenizer.size(), false);
    return new kolibri1::Pipeline(context.backend, std::move(embedding), std::move(engines),
                                  std::move(streamed_plans), std::move(head), std::move(tokenizer),
                                  cfg);
}

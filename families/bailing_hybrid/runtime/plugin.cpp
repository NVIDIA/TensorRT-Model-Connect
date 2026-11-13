/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/bailing_hybrid/runtime/tokenizer.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::bailing_hybrid {
namespace {
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || found->length == 0) throw std::runtime_error(std::string("Ling3 bundle lacks ") + name);
    return reader.read_section(name);
}
Tensor tensor(void* data, std::initializer_list<int64_t> shape, DType dtype) {
    Tensor result;
    result.data = data; result.shape = shape; result.dtype = dtype;
    return result;
}
Tensor output(const TensorMap& outputs, const std::string& name, std::size_t count, DType dtype = DType::kFloat32) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || !found->second.data || found->second.dtype != dtype ||
        found->second.numel() != count) throw std::runtime_error("Invalid Ling3 output " + name);
    return found->second;
}

class Pipeline final : public internal::IModel, public internal::ITextContinuation {
  public:
    Pipeline(std::unique_ptr<ITrtModule> embedding, std::vector<std::unique_ptr<ITrtModule>> layers,
             std::unique_ptr<ITrtModule> head, std::unique_ptr<ITokenizer> tokenizer, nlohmann::json cfg)
        : embedding_(std::move(embedding)), layers_(std::move(layers)), head_(std::move(head)),
          tokenizer_(std::move(tokenizer)), cfg_(std::move(cfg)), fields_{
            {"max_new_tokens", internal::ConfigKind::I64, internal::ConfigValue{int64_t{32}}, "Maximum generated tokens"},
            {"do_sample", internal::ConfigKind::Bool, internal::ConfigValue{false}, "Greedy decoding only"}} {}
    const char* task() const noexcept override { return "text_continuation"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ITextContinuation>(*this, {fields_.data(), fields_.size()})};
    }
    internal::TextResult run(const internal::TextContinuationRequest& request,
                            internal::ConfigView config) override {
        const Span<const internal::ConfigField> fields{fields_.data(), fields_.size()};
        internal::validate_config(fields, config);
        const auto max_new = *internal::config_get<int64_t>(config, fields, "max_new_tokens");
        if (*internal::config_get<bool>(config, fields, "do_sample"))
            throw std::invalid_argument("Ling3 currently supports greedy decoding");
        const int32_t length = cfg_.at("sequence_length"), vocab = cfg_.at("vocab_size");
        const int32_t layers = cfg_.at("layers"), heads = cfg_.at("heads"), dim = cfg_.at("head_dim");
        if (max_new <= 0 || max_new > length) throw std::invalid_argument("Invalid Ling3 generation length");
        std::vector<int32_t> ids;
        if (const auto* text = std::get_if<std::string_view>(&request.prefix))
            ids = tokenizer_->encode(std::string(*text));
        else {
            const auto tokens = std::get<Span<const int32_t>>(request.prefix);
            ids.assign(tokens.begin(), tokens.end());
        }
        if (ids.empty() || ids.size() + max_new > static_cast<std::size_t>(length))
            throw std::invalid_argument("Ling3 prefix and generation exceed the built KV profile");
        for (const auto id : ids)
            if (id < 0 || id >= vocab) throw std::invalid_argument("Invalid Ling3 token ID");
        const int32_t width = cfg_.at("hidden_size");
        const int32_t key_dim = cfg_.at("key_dim"), value_dim = cfg_.at("value_dim"), kernel = cfg_.at("conv_kernel");
        const auto kinds = cfg_.at("layer_types").get<std::vector<std::string>>();
        const auto key_count = static_cast<std::size_t>(heads) * length * key_dim;
        const auto value_count = static_cast<std::size_t>(heads) * length * value_dim;
        const auto state_count = static_cast<std::size_t>(heads) * dim * dim;
        const auto conv_count = static_cast<std::size_t>(heads) * dim * (kernel - 1);
        struct State {
            std::vector<float> key, value, recurrent;
            std::array<std::vector<float>, 3> convolution;
        };
        std::vector<State> states(layers);
        for (int32_t i = 0; i < layers; ++i) {
            if (kinds[i] == "mla") {
                states[i].key.resize(key_count, 0); states[i].value.resize(value_count, 0);
            } else {
                states[i].recurrent.resize(state_count, 0);
                for (auto& history : states[i].convolution) history.resize(conv_count, 0);
            }
        }
        std::vector<float> hidden(width);
        std::vector<float> logits;
        int32_t position = 0;
        auto step = [&](int32_t token) {
            if (position >= length) throw std::runtime_error("Ling3 exhausted its KV profile");
            const auto embedded = embedding_->forward({{"input_ids", tensor(&token, {1}, DType::kInt32)}});
            const auto initial = output(embedded, "result", width);
            std::copy_n(static_cast<const float*>(initial.data), width, hidden.data());
            for (int32_t i = 0; i < layers; ++i) {
                auto& state = states[i];
                TensorMap inputs{{"hidden", tensor(hidden.data(), {1, width}, DType::kFloat32)}};
                if (kinds[i] == "mla") {
                    inputs.emplace("position", tensor(&position, {1}, DType::kInt32));
                    inputs.emplace("cache_k", tensor(state.key.data(), {1, heads, length, key_dim}, DType::kFloat32));
                    inputs.emplace("cache_v", tensor(state.value.data(), {1, heads, length, value_dim}, DType::kFloat32));
                } else {
                    inputs.emplace("state", tensor(state.recurrent.data(), {heads, dim, dim}, DType::kFloat32));
                    for (int j = 0; j < 3; ++j)
                        inputs.emplace(std::string("conv_") + "qkv"[j], tensor(state.convolution[j].data(), {heads * dim, kernel - 1}, DType::kFloat32));
                }
                const auto predicted = layers_[i]->forward(inputs);
                const auto result = output(predicted, "result", width);
                std::copy_n(static_cast<const float*>(result.data), width, hidden.data());
                auto retain = [&](const std::string& name, std::vector<float>& storage) {
                    const auto view = output(predicted, name, storage.size());
                    std::copy_n(static_cast<const float*>(view.data), storage.size(), storage.data());
                };
                if (kinds[i] == "mla") {
                    retain("present_k", state.key); retain("present_v", state.value);
                } else {
                    retain("present_state", state.recurrent);
                    for (int j = 0; j < 3; ++j) retain(std::string("present_conv_") + "qkv"[j], state.convolution[j]);
                }
            }
            const auto predicted = head_->forward({{"hidden", tensor(hidden.data(), {1, width}, DType::kFloat32)}});
            const auto& scores = output(predicted, "logits", vocab);
            const auto* data = static_cast<const float*>(scores.data);
            logits.assign(data, data + vocab);
            for (const auto value : logits)
                if (!std::isfinite(value)) throw std::runtime_error("Ling3 logit is nonfinite");
            ++position;
        };
        for (const auto token : ids) step(token);
        const auto eos = cfg_.at("eos_ids").get<std::vector<int32_t>>();
        internal::TextResult result;
        for (int64_t i = 0; i < max_new; ++i) {
            const auto next = static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
            result.token_ids.push_back(next);
            if (std::find(eos.begin(), eos.end(), next) != eos.end()) break;
            if (i + 1 < max_new) step(next);
        }
        result.text = tokenizer_->decode(result.token_ids);
        return result;
    }
  private:
    std::unique_ptr<ITrtModule> embedding_;
    std::vector<std::unique_ptr<ITrtModule>> layers_;
    std::unique_ptr<ITrtModule> head_;
    std::unique_ptr<ITokenizer> tokenizer_;
    nlohmann::json cfg_;
    std::vector<internal::ConfigField> fields_;
};
} // namespace
} // namespace trtmc::bailing_hybrid

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    if (context.kv_cache_size_bytes || context.reader.info().task != "text_continuation")
        throw std::invalid_argument("Ling3 uses a family-owned fixed KV profile for text continuation");
    const auto raw = bailing_hybrid::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    const int32_t length = cfg.at("sequence_length"), layers = cfg.at("layers");
    const int32_t heads = cfg.at("heads"), dim = cfg.at("head_dim"), vocab = cfg.at("vocab_size");
    const int32_t width = cfg.at("hidden_size");
    const auto eos = cfg.at("eos_ids").get<std::vector<int32_t>>();
    const auto kinds = cfg.at("layer_types").get<std::vector<std::string>>();
    const int32_t kernel = cfg.at("conv_kernel"), key_dim = cfg.at("key_dim"), value_dim = cfg.at("value_dim");
    if (length <= 0 || length > 512 || layers <= 0 || layers > 128 || heads <= 0 || heads > 128 ||
        dim <= 0 || dim > 256 || width <= 0 || width > 65536 || vocab <= 0 || eos.empty() ||
        kernel < 2 || kernel > 16 || key_dim <= 0 || key_dim > 512 || value_dim <= 0 || value_dim > 512 ||
        kinds.size() != static_cast<std::size_t>(layers))
        throw std::runtime_error("Invalid Ling3 runtime dimensions");
    for (const auto& kind : kinds)
        if (kind != "mla" && kind != "kda") throw std::runtime_error("Invalid Ling3 state kind");
    for (auto id : eos)
        if (id < 0 || id >= vocab) throw std::runtime_error("Invalid Ling3 EOS token");
    auto load = [&](const std::string& name) {
        const auto plan = bailing_hybrid::section(context.reader, name.c_str());
        auto engine = context.backend.create_module(plan.data(), plan.size(), {});
        if (!engine || !engine->ok()) throw std::runtime_error("Ling3 engine failed to load: " + name);
        return engine;
    };
    // Read and deserialize one plan at a time; never retain a full CPU weight copy.
    auto embedding = load("embedding.plan");
    std::vector<std::unique_ptr<ITrtModule>> engines;
    for (int32_t i = 0; i < layers; ++i) engines.push_back(load("layer." + std::to_string(i) + ".plan"));
    auto head = load("head.plan");
    const auto raw_tokenizer = bailing_hybrid::section(context.reader, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(raw_tokenizer.data(), raw_tokenizer.size(), false);
    return new bailing_hybrid::Pipeline(std::move(embedding), std::move(engines), std::move(head), std::move(tokenizer), cfg);
}

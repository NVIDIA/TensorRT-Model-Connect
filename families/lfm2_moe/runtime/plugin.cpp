/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lfm2_moe/runtime/tokenizer.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::lfm2_moe {
namespace {
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || found->length == 0)
        throw std::runtime_error(std::string("LFM2 MoE bundle lacks ") + name);
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
        throw std::runtime_error("Invalid LFM2 MoE output " + name);
    return found->second;
}

class Pipeline final : public internal::IModel, public internal::ITextContinuation {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, std::unique_ptr<ITokenizer> tokenizer,
             nlohmann::json cfg)
        : engine_(std::move(engine)), tokenizer_(std::move(tokenizer)), cfg_(std::move(cfg)),
          fields_{{"max_new_tokens", internal::ConfigKind::I64, internal::ConfigValue{int64_t{32}},
                   "Maximum generated tokens"},
                  {"repetition_penalty", internal::ConfigKind::F64,
                   internal::ConfigValue{cfg_.value("repetition_penalty", 1.0)},
                   "Penalty over prefix and generated tokens"},
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
        const auto requested_penalty =
            *internal::config_get<double>(config, fields, "repetition_penalty");
        const float penalty = static_cast<float>(requested_penalty);
        if (!std::isfinite(penalty) || penalty <= 0)
            throw std::invalid_argument("LFM2 MoE repetition penalty must be positive and finite");
        if (*internal::config_get<bool>(config, fields, "do_sample"))
            throw std::invalid_argument("LFM2 MoE currently supports greedy decoding");
        const int32_t length = cfg_.at("sequence_length"), vocab = cfg_.at("vocab_size");
        const int32_t layers = cfg_.at("layers"), heads = cfg_.at("kv_heads"),
                      dim = cfg_.at("head_dim");
        if (max_new <= 0 || max_new > length)
            throw std::invalid_argument("Invalid LFM2 MoE generation length");
        std::vector<int32_t> ids;
        if (const auto* text = std::get_if<std::string_view>(&request.prefix))
            ids = tokenizer_->encode(std::string(*text));
        else {
            const auto tokens = std::get<Span<const int32_t>>(request.prefix);
            ids.assign(tokens.begin(), tokens.end());
        }
        if (ids.empty() || ids.size() + max_new > static_cast<std::size_t>(length))
            throw std::invalid_argument(
                "LFM2 MoE prefix and generation exceed the built KV profile");
        for (const auto id : ids)
            if (id < 0 || id >= vocab)
                throw std::invalid_argument("Invalid LFM2 MoE token ID");
        std::vector<bool> seen(vocab, false);
        for (const auto id : ids)
            seen[id] = true;
        const auto cache_count = static_cast<std::size_t>(heads) * length * dim;
        const auto kinds = cfg_.at("layer_types").get<std::vector<std::string>>();
        const int32_t width = cfg_.at("hidden_size"), kernel = cfg_.at("conv_kernel");
        const auto conv_count = static_cast<std::size_t>(width) * (kernel - 1);
        std::vector<std::vector<float>> keys(layers), values(layers), convs(layers);
        for (int32_t i = 0; i < layers; ++i) {
            if (kinds[i] == "conv")
                convs[i].assign(conv_count, 0);
            else {
                keys[i].assign(cache_count, 0);
                values[i].assign(cache_count, 0);
            }
        }
        std::vector<float> logits;
        int32_t position = 0;
        auto step = [&](int32_t token) {
            if (position >= length)
                throw std::runtime_error("LFM2 MoE exhausted its KV profile");
            TensorMap inputs{{"input_ids", tensor(&token, {1}, DType::kInt32)},
                             {"position", tensor(&position, {1}, DType::kInt32)}};
            for (int32_t i = 0; i < layers; ++i) {
                if (kinds[i] == "conv") {
                    inputs.emplace(
                        "cache_conv_" + std::to_string(i),
                        tensor(convs[i].data(), {1, width, kernel - 1}, DType::kFloat32));
                    continue;
                }
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
                    throw std::runtime_error("LFM2 MoE logit is nonfinite");
            for (int32_t i = 0; i < layers; ++i) {
                if (kinds[i] == "conv") {
                    const auto& state =
                        output(predicted, "present_conv_" + std::to_string(i), conv_count);
                    std::copy_n(static_cast<const float*>(state.data), conv_count, convs[i].data());
                    continue;
                }
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
            for (int32_t id = 0; id < vocab; ++id)
                if (seen[id])
                    logits[id] = logits[id] < 0 ? logits[id] * penalty : logits[id] / penalty;
            const auto next = static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) -
                                                   logits.begin());
            result.token_ids.push_back(next);
            seen[next] = true;
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
} // namespace trtmc::lfm2_moe

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("lfm2_moe does not support --kv-cache-size");
    using namespace trtmc;
    if (context.reader.info().task != "text_continuation")
        throw std::invalid_argument(
            "LFM2 MoE uses a family-owned fixed KV profile for text continuation");
    const auto raw = lfm2_moe::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    const int32_t length = cfg.at("sequence_length"), layers = cfg.at("layers");
    const int32_t heads = cfg.at("kv_heads"), dim = cfg.at("head_dim"),
                  vocab = cfg.at("vocab_size");
    const int32_t eos = cfg.at("eos_id");
    if (length <= 0 || length > 1024 || layers <= 0 || heads <= 0 || dim <= 0 || vocab <= 0 ||
        eos < 0 || eos >= vocab)
        throw std::runtime_error("Invalid LFM2 MoE runtime dimensions");
    const auto kinds = cfg.at("layer_types").get<std::vector<std::string>>();
    const int32_t width = cfg.at("hidden_size"), kernel = cfg.at("conv_kernel");
    if (kinds.size() != static_cast<std::size_t>(layers) || width <= 0 || kernel < 2 ||
        !std::all_of(
            kinds.begin(), kinds.end(),
            [](const auto& kind) { return kind == "conv" || kind == "full_attention"; }))
        throw std::runtime_error("Invalid LFM2 MoE recurrent state dimensions");
    const auto plan = lfm2_moe::section(context.reader, "model.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok())
        throw std::runtime_error("LFM2 MoE engine failed to load");
    const auto raw_tokenizer = lfm2_moe::section(context.reader, "tokenizer.json");
    auto tokenizer =
        lfm2_moe::CreateBpeTokenizer(raw_tokenizer.data(), raw_tokenizer.size(), false);
    return new lfm2_moe::Pipeline(std::move(engine), std::move(tokenizer), cfg);
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/seamless_m4t_v2/runtime/tokenizer.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/text.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::seamless_m4t_v2 {
namespace {
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || found->length == 0)
        throw std::runtime_error(std::string("SeamlessM4T v2 bundle lacks ") + name);
    return reader.read_section(name);
}
Tensor tensor(void* data, std::initializer_list<int64_t> shape, DType dtype) {
    Tensor result;
    result.data = data;
    result.shape = shape;
    result.dtype = dtype;
    return result;
}
const Tensor& output(const TensorMap& outputs, const char* name, std::size_t count) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || !found->second.data || found->second.dtype != DType::kFloat32 ||
        found->second.numel() != count)
        throw std::runtime_error(std::string("Invalid SeamlessM4T v2 output ") + name);
    return found->second;
}

class Pipeline final : public internal::IModel, public internal::ITextTranslation {
  public:
    Pipeline(std::unique_ptr<ITrtModule> encoder, std::unique_ptr<ITrtModule> decoder,
             std::unique_ptr<ITokenizer> tokenizer, nlohmann::json cfg)
        : encoder_(std::move(encoder)), decoder_(std::move(decoder)),
          tokenizer_(std::move(tokenizer)), cfg_(std::move(cfg)),
          fields_{{"max_new_tokens", internal::ConfigKind::I64,
                   internal::ConfigValue{
                       std::min<int64_t>(64, cfg_.at("sequence_length").get<int64_t>() - 2)},
                   "Maximum generated tokens; greedy decoding"}} {}
    const char* task() const noexcept override { return "text_translation"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {
            internal::bind<internal::ITextTranslation>(*this, {fields_.data(), fields_.size()})};
    }
    internal::TextResult run(const internal::TextTranslationRequest& request,
                             internal::ConfigView config) override {
        const Span<const internal::ConfigField> fields{fields_.data(), fields_.size()};
        internal::validate_config(fields, config);
        const auto max_new = *internal::config_get<int64_t>(config, fields, "max_new_tokens");
        const int32_t length = cfg_.at("sequence_length"), vocab = cfg_.at("vocab_size");
        const int32_t width = cfg_.at("source_width"), pad = cfg_.at("pad_id"),
                      eos = cfg_.at("eos_id");
        if (request.source_text.empty() || !request.target_language ||
            request.target_language->empty() || max_new <= 0 || max_new > length - 2)
            throw std::invalid_argument("SeamlessM4T v2 requires nonempty text, target language "
                                        "and a valid generation length");
        auto language_id = [&](std::string value) {
            if (value.size() > 4 && value.substr(0, 2) == "__" &&
                value.substr(value.size() - 2) == "__")
                value = value.substr(2, value.size() - 4);
            const auto& languages = cfg_.at("languages");
            if (!languages.contains(value))
                throw std::invalid_argument("Unsupported SeamlessM4T v2 language: " + value);
            return languages.at(value).get<int32_t>();
        };
        const int32_t source_language =
            language_id(std::string(request.source_language.value_or("eng")));
        const int32_t target_language = language_id(std::string(*request.target_language));
        auto source_ids = tokenizer_->encode(std::string(request.source_text));
        source_ids.insert(source_ids.begin(), source_language);
        source_ids.push_back(eos);
        if (source_ids.size() > static_cast<std::size_t>(length))
            throw std::invalid_argument("SeamlessM4T v2 source exceeds its built text profile");
        const auto count = source_ids.size();
        source_ids.resize(length, pad);
        std::vector<int32_t> source_mask(length, 0), source_positions(length, pad);
        int32_t position = pad;
        for (std::size_t i = 0; i < count; ++i) {
            if (source_ids[i] < 0 || source_ids[i] >= vocab)
                throw std::invalid_argument("Invalid SeamlessM4T v2 source token ID");
            source_mask[i] = 1;
            if (source_ids[i] != pad)
                source_positions[i] = ++position;
        }
        const auto encoded = encoder_->forward(
            {{"input_ids", tensor(source_ids.data(), {length}, DType::kInt32)},
             {"attention_mask", tensor(source_mask.data(), {length}, DType::kInt32)},
             {"position_ids", tensor(source_positions.data(), {length}, DType::kInt32)}});
        const auto& memory =
            output(encoded, "encoder_hidden", static_cast<std::size_t>(length) * width);
        const auto* memory_data = static_cast<const float*>(memory.data);
        std::vector<float> source(memory_data, memory_data + memory.numel());
        std::vector<int32_t> ids(length, pad), valid(length, 0), positions(length, pad);
        ids[0] = cfg_.at("bos_id");
        ids[1] = target_language;
        valid[0] = valid[1] = 1;
        positions[0] = pad + 1;
        positions[1] = pad + 2;
        internal::TextResult result;
        for (int32_t step = 0; step < max_new; ++step) {
            int32_t last = step + 1;
            const auto decoded = decoder_->forward(
                {{"input_ids", tensor(ids.data(), {length}, DType::kInt32)},
                 {"attention_mask", tensor(valid.data(), {length}, DType::kInt32)},
                 {"position_ids", tensor(positions.data(), {length}, DType::kInt32)},
                 {"encoder_hidden", tensor(source.data(), {length, width}, DType::kFloat32)},
                 {"encoder_mask", tensor(source_mask.data(), {length}, DType::kInt32)},
                 {"last_index", tensor(&last, {1}, DType::kInt32)}});
            const auto& logits = output(decoded, "logits", vocab);
            const auto* values = static_cast<const float*>(logits.data);
            for (int32_t j = 0; j < vocab; ++j)
                if (!std::isfinite(values[j]))
                    throw std::runtime_error("SeamlessM4T v2 logit is nonfinite");
            const auto next =
                static_cast<int32_t>(std::max_element(values, values + vocab) - values);
            result.token_ids.push_back(next);
            if (next == eos)
                break;
            ids[step + 2] = next;
            valid[step + 2] = 1;
            positions[step + 2] = next == pad ? pad : pad + step + 3;
        }
        result.text = tokenizer_->decode(result.token_ids);
        return result;
    }

  private:
    std::unique_ptr<ITrtModule> encoder_, decoder_;
    std::unique_ptr<ITokenizer> tokenizer_;
    nlohmann::json cfg_;
    std::vector<internal::ConfigField> fields_;
};
} // namespace
} // namespace trtmc::seamless_m4t_v2

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("seamless_m4t_v2 does not support --kv-cache-size");
    using namespace trtmc;
    if (context.reader.info().task != "text_translation")
        throw std::invalid_argument(
            "SeamlessM4T v2 requires speech transcription without an external KV cache");
    const auto raw = seamless_m4t_v2::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    const int32_t length = cfg.at("sequence_length"), width = cfg.at("source_width");
    const int32_t vocab = cfg.at("vocab_size"), bos = cfg.at("bos_id"), eos = cfg.at("eos_id"),
                  pad = cfg.at("pad_id");
    if (length < 3 || width <= 0 || vocab <= 0 || bos < 0 || bos >= vocab || eos < 0 ||
        eos >= vocab || pad < 0 || pad >= vocab || cfg.at("source_language") != "eng" ||
        cfg.at("languages").empty())
        throw std::runtime_error("Invalid SeamlessM4T v2 runtime profile");
    for (const auto& item : cfg.at("languages").items()) {
        const int32_t id = item.value();
        if (item.key().empty() || id < 0 || id >= vocab)
            throw std::runtime_error("Invalid SeamlessM4T v2 language map");
    }
    auto load = [&](const char* name) {
        const auto plan = seamless_m4t_v2::section(context.reader, name);
        auto engine = context.backend.create_module(plan.data(), plan.size(), {});
        if (!engine || !engine->ok())
            throw std::runtime_error(std::string("SeamlessM4T v2 failed to load ") + name);
        return engine;
    };
    const auto raw_tokenizer = seamless_m4t_v2::section(context.reader, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(raw_tokenizer.data(), raw_tokenizer.size(), false, vocab);
    return new seamless_m4t_v2::Pipeline(load("encoder.plan"), load("decoder.plan"),
                                         std::move(tokenizer), cfg);
}

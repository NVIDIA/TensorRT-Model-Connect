/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/longformer/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"
#include <algorithm>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::longformer {
namespace {
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || found->length == 0)
        throw std::runtime_error(std::string("Longformer bundle is missing ") + name);
    return reader.read_section(name);
}

class Pipeline final : public internal::IModel, public internal::ITextToTokenFeatures {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, std::unique_ptr<ITokenizer> tokenizer,
             int32_t length, int32_t width, int32_t vocab, int32_t pad)
        : engine_(std::move(engine)), tokenizer_(std::move(tokenizer)), length_(length),
          width_(width), vocab_(vocab), pad_(pad) {}
    const char* task() const noexcept override { return "text_to_token_features"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ITextToTokenFeatures>(*this)};
    }
    internal::TokenFeaturesResult run(const internal::TextToTokenFeaturesRequest& request,
                                      internal::ConfigView config) override {
        if (!config.empty()) throw internal::ConfigError("Longformer has no runtime config");
        std::vector<int32_t> ids;
        const bool text = std::holds_alternative<std::string_view>(request.text);
        if (text) {
            ids = tokenizer_->encode(std::string(std::get<std::string_view>(request.text)));
        } else {
            const auto source = std::get<Span<const int32_t>>(request.text);
            ids.assign(source.begin(), source.end());
        }
        if (ids.empty() || ids.size() > static_cast<std::size_t>(length_))
            throw std::invalid_argument("Longformer input exceeds the built sequence profile");
        for (const auto id : ids) {
            if (id < 0 || id >= vocab_ || id == pad_)
                throw std::invalid_argument("Longformer requires valid, unpadded token IDs");
        }
        const auto count = ids.size();
        ids.resize(length_, pad_);
        std::vector<int32_t> valid(length_, 0);
        std::fill(valid.begin(), valid.begin() + count, 1);
        Tensor input, mask;
        input.data = ids.data(); input.shape = {length_}; input.dtype = DType::kInt32;
        mask.data = valid.data(); mask.shape = {length_}; mask.dtype = DType::kInt32;
        auto outputs = engine_->forward({{"input_ids", input}, {"attention_mask", mask}});
        const auto found = outputs.find("hidden_states");
        if (found == outputs.end() || !found->second.data ||
            found->second.dtype != DType::kFloat32 ||
            found->second.shape != std::vector<int64_t>{length_, width_})
            throw std::runtime_error("Longformer engine returned invalid token features");
        const auto* values = static_cast<const float*>(found->second.data);
        internal::TokenFeaturesResult result;
        result.features.values.assign(values, values + count * width_);
        result.features.rows = count; result.features.columns = width_;
        for (std::size_t index = 0; index < count; ++index) {
            internal::FeatureToken token;
            token.token_id = ids[index]; token.token_index = index;
            token.input_index = text && (index == 0 || index + 1 == count) ? -1 : 0;
            result.tokens.push_back(token);
        }
        return result;
    }
  private:
    std::unique_ptr<ITrtModule> engine_;
    std::unique_ptr<ITokenizer> tokenizer_;
    int32_t length_, width_, vocab_, pad_;
};
} // namespace
} // namespace trtmc::longformer

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    if (context.kv_cache_size_bytes != 0 || context.reader.info().task != "text_to_token_features")
        throw std::invalid_argument("Longformer requires the token-feature task without a KV cache");
    const auto data = longformer::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(data.begin(), data.end());
    const int32_t length = cfg.at("sequence_length"), width = cfg.at("hidden_size");
    const int32_t vocab = cfg.at("vocab_size"), pad = cfg.at("pad_token_id");
    if (length <= 0 || length > 4096 || width <= 0 || vocab <= 0 || pad < 0 || pad >= vocab)
        throw std::runtime_error("Invalid Longformer runtime dimensions");
    const auto plan = longformer::section(context.reader, "model.plan");
    auto engine = context.backend.create_module(plan.data(),plan.size(),{});
    if (!engine || !engine->ok()) throw std::runtime_error("Longformer engine failed to load");
    const auto tokenizer_json = longformer::section(context.reader,"tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(tokenizer_json.data(),tokenizer_json.size(),true);
    return new longformer::Pipeline(std::move(engine),std::move(tokenizer),length,width,vocab,pad);
}

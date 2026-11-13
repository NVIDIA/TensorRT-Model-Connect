/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/jina_embeddings_v3/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"
#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::jina_embeddings_v3 {
namespace {
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || found->length == 0)
        throw std::runtime_error(std::string("Jina embedding bundle is missing ") + name);
    return reader.read_section(name);
}

class Pipeline final : public internal::IModel, public internal::ITextToEmbedding {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, std::unique_ptr<ITokenizer> tokenizer,
             int32_t length, int32_t width, int32_t vocab, int32_t pad)
        : engine_(std::move(engine)), tokenizer_(std::move(tokenizer)), length_(length),
          width_(width), vocab_(vocab), pad_(pad) {}
    const char* task() const noexcept override { return "text_to_embedding"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ITextToEmbedding>(*this)};
    }
    internal::SemanticEmbeddingResult run(const internal::TextToEmbeddingRequest& request,
                                          internal::ConfigView config) override {
        if (!config.empty()) throw internal::ConfigError("Jina embedding has no runtime config");
        int32_t adapter_id = 0;
        switch (request.role) {
        case internal::EmbeddingRole::Default: break;
        case internal::EmbeddingRole::Query: adapter_id = 1; break;
        case internal::EmbeddingRole::Document: adapter_id = 2; break;
        default: throw std::invalid_argument("Unknown Jina embedding role");
        }
        auto ids = tokenizer_->encode(std::string(request.text));
        if (ids.empty() || ids.size() > static_cast<std::size_t>(length_))
            throw std::invalid_argument("Jina embedding requires a nonempty input within the built profile");
        for (const auto id : ids) {
            if (id < 0 || id >= vocab_)
                throw std::invalid_argument("Jina tokenizer returned an invalid token ID");
        }
        const auto count = ids.size();
        ids.resize(length_, pad_);
        std::vector<int32_t> valid(length_, 0);
        std::fill(valid.begin(), valid.begin() + count, 1);
        Tensor input, mask;
        input.data = ids.data(); input.shape = {length_}; input.dtype = DType::kInt32;
        mask.data = valid.data(); mask.shape = {length_}; mask.dtype = DType::kInt32;
        Tensor adapter;
        adapter.data = &adapter_id; adapter.shape = {1}; adapter.dtype = DType::kInt32;
        auto outputs = engine_->forward({{"input_ids", input}, {"attention_mask", mask},
                                         {"adapter_id", adapter}});
        const auto found = outputs.find("embedding");
        if (found == outputs.end() || !found->second.data ||
            found->second.dtype != DType::kFloat32 ||
            found->second.shape != std::vector<int64_t>{1, width_})
            throw std::runtime_error("Jina engine returned an invalid embedding");
        const auto* values = static_cast<const float*>(found->second.data);
        internal::SemanticEmbeddingResult result;
        result.values.assign(values, values + width_);
        for (const auto value : result.values) {
            if (!std::isfinite(value)) throw std::runtime_error("Jina embedding is nonfinite");
        }
        result.embedding_space = "jina_embeddings_v3";
        result.pooling = "mean";
        result.normalization = "l2";
        return result;
    }
  private:
    std::unique_ptr<ITrtModule> engine_;
    std::unique_ptr<ITokenizer> tokenizer_;
    int32_t length_, width_, vocab_, pad_;
};
} // namespace
} // namespace trtmc::jina_embeddings_v3

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    if (context.kv_cache_size_bytes != 0 || context.reader.info().task != "text_to_embedding")
        throw std::invalid_argument("Jina requires text embedding without a KV cache");
    const auto data = jina_embeddings_v3::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(data.begin(), data.end());
    const int32_t length = cfg.at("sequence_length"), width = cfg.at("hidden_size");
    const int32_t vocab = cfg.at("vocab_size"), pad = cfg.at("pad_token_id");
    if (length <= 0 || length > 8192 || width <= 0 || vocab <= 0 || pad < 0 || pad >= vocab)
        throw std::runtime_error("Invalid Jina embedding runtime dimensions");
    const auto plan = jina_embeddings_v3::section(context.reader, "model.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok()) throw std::runtime_error("Jina engine failed to load");
    const auto tokenizer_json = jina_embeddings_v3::section(context.reader, "tokenizer.json");
    auto tokenizer = CreateUnigramTokenizer(tokenizer_json.data(), tokenizer_json.size(), true);
    return new jina_embeddings_v3::Pipeline(std::move(engine), std::move(tokenizer), length, width,
        vocab, pad);
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/nemotron_embed/runtime/tokenizer.h"
#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::nemotron_embed {
namespace {
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || found->length == 0)
        throw std::runtime_error(std::string("Nemotron embedding bundle is missing ") + name);
    return reader.read_section(name);
}

class Pipeline final : public internal::IModel, public internal::ITextToEmbedding {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, std::unique_ptr<ITokenizer> tokenizer,
             int32_t length, int32_t width, int32_t vocab, int32_t pad, std::string query,
             std::string document)
        : engine_(std::move(engine)), tokenizer_(std::move(tokenizer)), length_(length),
          width_(width), vocab_(vocab), pad_(pad), query_(std::move(query)),
          document_(std::move(document)) {}
    const char* task() const noexcept override { return "text_to_embedding"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ITextToEmbedding>(*this)};
    }
    internal::SemanticEmbeddingResult run(const internal::TextToEmbeddingRequest& request,
                                          internal::ConfigView config) override {
        if (!config.empty())
            throw internal::ConfigError("Nemotron embedding has no runtime config");
        if (request.text.empty())
            throw std::invalid_argument("Nemotron embedding requires nonempty request text");
        std::string text;
        switch (request.role) {
        case internal::EmbeddingRole::Default:
            break;
        case internal::EmbeddingRole::Query:
            text = query_;
            break;
        case internal::EmbeddingRole::Document:
            text = document_;
            break;
        default:
            throw std::invalid_argument("Unknown Nemotron embedding role");
        }
        text.append(request.text);
        auto ids = tokenizer_->encode(text);
        if (ids.empty() || ids.size() > static_cast<std::size_t>(length_))
            throw std::invalid_argument(
                "Nemotron embedding requires a nonempty input within the built profile");
        for (const auto id : ids) {
            if (id < 0 || id >= vocab_)
                throw std::invalid_argument("Nemotron tokenizer returned an invalid token ID");
        }
        const auto count = ids.size();
        ids.resize(length_, pad_);
        std::vector<int32_t> valid(length_, 0);
        std::fill(valid.begin(), valid.begin() + count, 1);
        Tensor input, mask;
        input.data = ids.data();
        input.shape = {length_};
        input.dtype = DType::kInt32;
        mask.data = valid.data();
        mask.shape = {length_};
        mask.dtype = DType::kInt32;
        auto outputs = engine_->forward({{"input_ids", input}, {"attention_mask", mask}});
        const auto found = outputs.find("embedding");
        if (found == outputs.end() || !found->second.data ||
            found->second.dtype != DType::kFloat32 ||
            found->second.shape != std::vector<int64_t>{1, width_})
            throw std::runtime_error("Nemotron engine returned an invalid embedding");
        const auto* values = static_cast<const float*>(found->second.data);
        internal::SemanticEmbeddingResult result;
        result.values.assign(values, values + width_);
        for (const auto value : result.values) {
            if (!std::isfinite(value))
                throw std::runtime_error("Nemotron embedding is nonfinite");
        }
        result.embedding_space = "nemotron_embed";
        result.pooling = "mean";
        result.normalization = "l2";
        return result;
    }

  private:
    std::unique_ptr<ITrtModule> engine_;
    std::unique_ptr<ITokenizer> tokenizer_;
    int32_t length_, width_, vocab_, pad_;
    std::string query_, document_;
};
} // namespace
} // namespace trtmc::nemotron_embed

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("nemotron_embed does not support --kv-cache-size");
    using namespace trtmc;
    if (context.reader.info().task != "text_to_embedding")
        throw std::invalid_argument("Nemotron requires text embedding without a KV cache");
    const auto data = nemotron_embed::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(data.begin(), data.end());
    const int32_t length = cfg.at("sequence_length"), width = cfg.at("hidden_size");
    const int32_t vocab = cfg.at("vocab_size"), pad = cfg.at("pad_token_id");
    if (length <= 0 || length > 16384 || width <= 0 || vocab <= 0 || pad < 0 || pad >= vocab)
        throw std::runtime_error("Invalid Nemotron embedding runtime dimensions");
    const auto plan = nemotron_embed::section(context.reader, "model.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok())
        throw std::runtime_error("Nemotron engine failed to load");
    for (const auto* name : {"input_ids", "attention_mask"}) {
        if (!engine->has_input(name) ||
            engine->tensor_shape(name) != std::vector<int64_t>{length} ||
            engine->tensor_dtype(name) != DType::kInt32)
            throw std::runtime_error(
                "Nemotron engine input does not match runtime sequence_length");
    }
    const auto tokenizer_json = nemotron_embed::section(context.reader, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), true);
    return new nemotron_embed::Pipeline(std::move(engine), std::move(tokenizer), length, width,
                                        vocab, pad, cfg.at("prompts").at("query"),
                                        cfg.at("prompts").at("document"));
}

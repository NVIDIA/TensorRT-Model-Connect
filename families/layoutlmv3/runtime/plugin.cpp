/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/layoutlmv3/runtime/document_input.h"
#include "families/layoutlmv3/runtime/image_preprocess.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"
namespace trtmc::layoutlmv3 {
namespace {
class Pipeline final : public internal::IModel, public internal::IDocumentWordsToTokenFeatures {
  public:
    Pipeline(std::unique_ptr<ITrtModule> engine, std::unique_ptr<ITokenizer> tokenizer,
             nlohmann::json cfg)
        : engine_(std::move(engine)), tokenizer_(std::move(tokenizer)), cfg_(std::move(cfg)) {}
    const char* task() const noexcept override { return "document_words_to_token_features"; }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IDocumentWordsToTokenFeatures>(*this)};
    }
    internal::DocumentTokenFeaturesResult
    run(const internal::DocumentWordsToTokenFeaturesRequest& input,
        internal::ConfigView config) override {
        internal::validate_config({}, config);
        auto encoded = encode_document(*tokenizer_, input, cfg_);
        auto pixels = image_pixels(input.image);
        const int64_t length = cfg_.at("sequence_length"), width = cfg_.at("hidden_size"),
                      total = length + 197;
        const auto tensor = [](std::vector<int32_t>& data, std::vector<int64_t> shape) {
            Tensor result;
            result.data = data.data();
            result.shape = std::move(shape);
            result.dtype = DType::kInt32;
            return result;
        };
        Tensor image;
        image.data = pixels.data();
        image.shape = {1, 3, 224, 224};
        image.dtype = DType::kFloat32;
        const auto outputs =
            engine_->forward({{"input_ids", tensor(encoded.ids, {length})},
                              {"attention_mask", tensor(encoded.valid, {length})},
                              {"position_ids", tensor(encoded.positions, {length})},
                              {"layout_indices", tensor(encoded.layout, {length, 6})},
                              {"relative_x", tensor(encoded.relative_x, {total, total})},
                              {"relative_y", tensor(encoded.relative_y, {total, total})},
                              {"pixel_values", image}});
        const auto found = outputs.find("features");
        if (found == outputs.end() || found->second.dtype != DType::kFloat32 ||
            found->second.numel() != static_cast<uint64_t>(length * width) || !found->second.data)
            throw std::runtime_error("Invalid LayoutLMv3 document feature output");
        internal::DocumentTokenFeaturesResult result;
        result.features.rows = length;
        result.features.columns = width;
        result.source_word_count = input.words.size();
        const auto* values = static_cast<const float*>(found->second.data);
        result.features.values.assign(values, values + length * width);
        result.tokens = std::move(encoded.mapping);
        return result;
    }

  private:
    std::unique_ptr<ITrtModule> engine_;
    std::unique_ptr<ITokenizer> tokenizer_;
    nlohmann::json cfg_;
};
std::vector<char> section(const BundleReader& reader, const char* name) {
    const auto* found = reader.find_section(name);
    if (!found || !found->length)
        throw std::runtime_error(std::string("LayoutLMv3 bundle lacks ") + name);
    return reader.read_section(name);
}
} // namespace
} // namespace trtmc::layoutlmv3
extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    using namespace trtmc;
    if (context.kv_cache_size_bytes ||
        context.reader.info().task != "document_words_to_token_features")
        throw std::invalid_argument(
            "LayoutLMv3 requires document features without an external cache");
    const auto raw = layoutlmv3::section(context.reader, "runtime.json");
    const auto cfg = nlohmann::json::parse(raw.begin(), raw.end());
    const int64_t length = cfg.at("sequence_length"), width = cfg.at("hidden_size"),
                  vocab = cfg.at("vocab_size");
    if (length < 4 || length > 512 || width <= 0 || width > 8192 || vocab <= 0 ||
        cfg.at("image_size") != 224 || cfg.at("spatial_buckets").size() != 2001)
        throw std::runtime_error("Invalid LayoutLMv3 document profile");
    const auto plan = layoutlmv3::section(context.reader, "model.plan");
    auto engine = context.backend.create_module(plan.data(), plan.size(), {});
    if (!engine || !engine->ok())
        throw std::runtime_error("LayoutLMv3 engine failed to load");
    const auto data = layoutlmv3::section(context.reader, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(data.data(), data.size(), false);
    return new layoutlmv3::Pipeline(std::move(engine), std::move(tokenizer), cfg);
}

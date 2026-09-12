/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/family_factory.h"

#include <cmath>
#include <limits>
#include <string>

namespace {
using namespace trtmc;
using namespace trtmc::internal;

class FeaturesFixture final : public IModel,
                              public ITextToTokenFeatures,
                              public ITextPairToTokenFeatures,
                              public ITextToPooledFeatures,
                              public ITextToHeadScores,
                              public ITextToEmbedding,
                              public ITitleBodyToEmbedding,
                              public IMaskedTextToTokenScores,
                              public ITextPairToPretrainingRelationScores,
                              public ITextToReplacedTokenScores,
                              public ITextPredictionPositionsToTokenScores,
                              public IImageToTokenFeatures,
                              public IImageToTokenAndPooledFeatures,
                              public IImageToSpatialFeatures,
                              public IImageToPooledFeatures,
                              public IImageToEmbedding,
                              public IImageTextToEmbedding,
                              public ITextPairToRelevance,
                              public ITextImageToRelevance,
                              public ITextImageTextToRelevance,
                              public IImageToClassScores,
                              public ITextQueryDocumentsToRelevance,
                              public IBatchImageToClassScores,
                              public IBatchImageToTokenFeatures,
                              public IBatchImageToSpatialFeatures,
                              public IBatchImageToPooledFeatures,
                              public IBatchTextToEmbedding,
                              public IBatchTextToTokenFeatures,
                              public IBatchImageToTokenAndPooledFeatures {
  public:
    explicit FeaturesFixture(std::string mode) : mode_(std::move(mode)) {}
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInstance> task_bindings() override {
        if (mode_ == "none")
            return {};
        return {
            bind<IBatchImageToTokenFeatures>(*this, fields_for(IBatchImageToTokenFeatures::kTask)),
            bind<IBatchImageToSpatialFeatures>(*this,
                                               fields_for(IBatchImageToSpatialFeatures::kTask)),
            bind<IBatchImageToPooledFeatures>(*this,
                                              fields_for(IBatchImageToPooledFeatures::kTask)),
            bind<IBatchTextToEmbedding>(*this, fields_for(IBatchTextToEmbedding::kTask)),
            bind<IBatchTextToTokenFeatures>(*this, fields_for(IBatchTextToTokenFeatures::kTask)),
            bind<IBatchImageToTokenAndPooledFeatures>(
                *this, fields_for(IBatchImageToTokenAndPooledFeatures::kTask)),
            bind<IBatchImageToClassScores>(*this, fields_for(IBatchImageToClassScores::kTask)),
            bind<ITextToTokenFeatures>(*this, fields_for(ITextToTokenFeatures::kTask)),
            bind<ITextPairToTokenFeatures>(*this, fields_for(ITextPairToTokenFeatures::kTask)),
            bind<ITextToPooledFeatures>(*this, fields_for(ITextToPooledFeatures::kTask)),
            bind<ITextToHeadScores>(*this, fields_for(ITextToHeadScores::kTask)),
            bind<ITextToEmbedding>(*this, fields_for(ITextToEmbedding::kTask)),
            bind<ITitleBodyToEmbedding>(*this, fields_for(ITitleBodyToEmbedding::kTask)),
            bind<IMaskedTextToTokenScores>(*this, fields_for(IMaskedTextToTokenScores::kTask)),
            bind<ITextPairToPretrainingRelationScores>(
                *this, fields_for(ITextPairToPretrainingRelationScores::kTask)),
            bind<ITextToReplacedTokenScores>(*this, fields_for(ITextToReplacedTokenScores::kTask)),
            bind<ITextPredictionPositionsToTokenScores>(
                *this, fields_for(ITextPredictionPositionsToTokenScores::kTask)),
            bind<IImageToTokenFeatures>(*this, fields_for(IImageToTokenFeatures::kTask)),
            bind<IImageToTokenAndPooledFeatures>(*this,
                                                 fields_for(IImageToTokenAndPooledFeatures::kTask)),
            bind<IImageToSpatialFeatures>(*this, fields_for(IImageToSpatialFeatures::kTask)),
            bind<IImageToPooledFeatures>(*this, fields_for(IImageToPooledFeatures::kTask)),
            bind<IImageToEmbedding>(*this, fields_for(IImageToEmbedding::kTask)),
            bind<IImageTextToEmbedding>(*this, fields_for(IImageTextToEmbedding::kTask)),
            bind<ITextPairToRelevance>(*this, fields_for(ITextPairToRelevance::kTask)),
            bind<ITextImageToRelevance>(*this, fields_for(ITextImageToRelevance::kTask)),
            bind<ITextImageTextToRelevance>(*this, fields_for(ITextImageTextToRelevance::kTask)),
            bind<IImageToClassScores>(*this, fields_for(IImageToClassScores::kTask)),
            bind<ITextQueryDocumentsToRelevance>(*this,
                                                 fields_for(ITextQueryDocumentsToRelevance::kTask)),
        };
    }
    trtmc::Span<const ConfigField> fields_for(std::string_view task_id) const {
        static const ConfigField single[] = {
            {"scale", ConfigKind::F64, ConfigValue{1.0}, "Synthetic fixture scale."}};
        static const ConfigField batch[] = {
            {"scale", ConfigKind::F64, ConfigValue{1.0}, "Synthetic batch scale."},
            {"annotate", ConfigKind::Bool, ConfigValue{true}, "Add a synthetic probe offset."}};
        static const ConfigField tagged_batch[] = {
            {"scale", ConfigKind::F64, ConfigValue{1.0}, "Synthetic batch scale."},
            {"annotate", ConfigKind::Bool, ConfigValue{true}, "Add a synthetic probe offset."},
            {"tag", ConfigKind::String, ConfigValue{std::string_view{"default"}},
             "Synthetic metadata tag."}};
        static const ConfigField head[] = {
            {"scale", ConfigKind::F64, ConfigValue{1.0}, "Synthetic head score scale."},
            {"representation", ConfigKind::String, ConfigValue{std::string_view{"raw"}},
             "Synthetic family-owned representation: raw, first or unit."}};
        if (task_id == ITextToHeadScores::kTask)
            return head;
        if (task_id == IBatchImageToClassScores::kTask || task_id == IBatchTextToEmbedding::kTask)
            return tagged_batch;
        if (task_id.find("batch_") == 0)
            return batch;
        return single;
    }

    TokenFeaturesResult run(const TextToTokenFeaturesRequest& request, ConfigView config) override {
        auto result = token_features(1, request.text, scale(config));
        if (mode_.rfind("padding", 0) == 0)
            append_padding(result, 7);
        if (mode_ == "bad_shape")
            result.features.values.pop_back();
        return result;
    }
    TokenFeaturesResult run(const TextPairToTokenFeaturesRequest& request,
                            ConfigView config) override {
        const auto factor = scale(config);
        if (const auto* text = std::get_if<FeatureTextPair>(&request.pair)) {
            TokenFeaturesResult result{{{float(2 * factor), float(text->first.size()),
                                         float(20 * factor), float(text->second.size())},
                                        2,
                                        2},
                                       {{11, 0, 0, true, 0, text->first.size()},
                                        {12, 1, 1, true, 0, text->second.size()}}};
            if (mode_.rfind("padding", 0) == 0) {
                result.features.values.insert(result.features.values.end(), {8.25F, -4.5F});
                ++result.features.rows;
                result.tokens.push_back({101, -1, 2, false, 0, 0});
                append_padding(result, 3);
            }
            return result;
        }
        const auto& tokens = std::get<FeatureTokenizedPair>(request.pair);
        TokenFeaturesResult result;
        result.features.columns = 2;
        for (size_t i = 0; i < tokens.token_ids.size(); ++i) {
            const bool padding = !tokens.attention_mask.empty() && tokens.attention_mask[i] == 0;
            if (padding && mode_.rfind("padding", 0) != 0)
                continue;
            if (!padding && (tokens.segment_ids[i] < 0 || tokens.segment_ids[i] > 1))
                throw std::invalid_argument("fixture accepts segment IDs zero and one");
            result.features.values.insert(
                result.features.values.end(),
                {padding ? 37.5F : float(2 * factor), float(tokens.token_ids[i])});
            result.tokens.push_back({tokens.token_ids[i],
                                     padding ? kFeaturePaddingInputIndex : tokens.segment_ids[i], i,
                                     false, 0, 0});
        }
        result.features.rows = result.tokens.size();
        return result;
    }
    PooledFeaturesResult run(const TextToPooledFeaturesRequest& request,
                             ConfigView config) override {
        return {{float(3 * scale(config)), float(text_size(request.text))}, "mean", "none"};
    }
    HeadScoresResult run(const TextToHeadScoresRequest& request, ConfigView config) override {
        const auto factor =
            config_get<double>(config, fields_for(ITextToHeadScores::kTask), "scale").value();
        if (!std::isfinite(factor) || factor < 0)
            throw ConfigError("scale must be finite and nonnegative");
        HeadScoresResult result{{float(-2 * factor), float(text_size(request.text)),
                                 float(3 * factor), float(-4 * factor)},
                                {1, 2, 2},
                                ScoreKind::Logit,
                                "none",
                                "none"};
        const auto representation =
            config_get<std::string_view>(config, fields_for(ITextToHeadScores::kTask),
                                         "representation")
                .value();
        if (representation == "first")
            result = {{float(-2 * factor)}, {1}, ScoreKind::Logit, "first_token", "none"};
        else if (representation == "unit")
            result = {{-0.6F, 0.8F}, {2}, ScoreKind::Unbounded, "mean", "l2"};
        else if (representation != "raw")
            throw ConfigError("unknown synthetic head representation");
        if (mode_ == "head_bad_shape")
            result.shape = {3, 2};
        if (mode_ == "head_zero_dim")
            result.shape = {0, 4};
        if (mode_ == "head_empty_shape")
            result.shape.clear();
        if (mode_ == "head_overflow")
            result.shape = {std::numeric_limits<uint64_t>::max(), 2};
        if (mode_ == "head_bad_kind")
            result.kind = static_cast<ScoreKind>(99);
        if (mode_ == "head_missing_pooling")
            result.pooling.clear();
        if (mode_ == "head_missing_normalization")
            result.normalization.clear();
        return result;
    }
    SemanticEmbeddingResult run(const TextToEmbeddingRequest& request, ConfigView config) override {
        return embedding(float(4 * scale(config)), float(static_cast<uint32_t>(request.role)));
    }
    SemanticEmbeddingResult run(const TitleBodyToEmbeddingRequest& request,
                                ConfigView config) override {
        return embedding(float(5 * scale(config) + request.title.size()),
                         float(request.body.size()));
    }
    VocabularyScoresResult run(const MaskedTextToTokenScoresRequest& request,
                               ConfigView config) override {
        const auto factor = scale(config);
        return {{{float(6 * factor), float(text_size(request.text)), -1}, 1, 3},
                {{42, 0, 2, false, 0, 0}},
                "fixture.vocabulary"};
    }
    LabelScoresResult run(const TextPairToPretrainingRelationScoresRequest& request,
                          ConfigView config) override {
        (void)scale(config);
        if (request.first.empty() || request.second.empty())
            throw std::invalid_argument("fixture requires pair");
        return {{0.25F, 0.75F}, {"next", "not_next"}, ScoreKind::Probability, "fixture.relation"};
    }
    ReplacedTokenScoresResult run(const TextToReplacedTokenScoresRequest&,
                                  ConfigView config) override {
        const auto factor = scale(config);
        return {{float(-2 * factor), float(2 * factor)},
                {{31, 0, 0, false, 0, 0}, {32, 0, 1, false, 0, 0}}};
    }
    VocabularyScoresResult run(const TextPredictionPositionsToTokenScoresRequest& request,
                               ConfigView config) override {
        const auto factor = scale(config);
        VocabularyScoresResult result;
        result.logits.rows = request.prediction_positions.size();
        result.logits.columns = 3;
        result.vocabulary_id = "fixture.xlnet";
        for (const auto position : request.prediction_positions) {
            result.logits.values.insert(
                result.logits.values.end(),
                {float(9 * factor), float(position), float(request.blocked_attention[position])});
            result.positions.push_back({request.token_ids[position], 0, position, false, 0, 0});
        }
        return result;
    }
    ImageTokenFeaturesResult run(const ImageToTokenFeaturesRequest& request,
                                 ConfigView config) override {
        const auto value = float(10 * scale(config));
        return {{{value, pixel(request.image), value + 1, 0, value + 2, 0}, 3, 2},
                {{ImageFeatureTokenRole::Class, 0, 0, 0, 0, 0, 0},
                 {ImageFeatureTokenRole::Register, 0, 0, 0, 0, 0, 0},
                 {ImageFeatureTokenRole::Patch, 0, 0, 0, 0, 1, 1}},
                1,
                1};
    }
    ImageTokenAndPooledFeaturesResult run(const ImageToTokenAndPooledFeaturesRequest& request,
                                          ConfigView config) override {
        const auto marker = float(20 * scale(config));
        const auto invocation = static_cast<float>(++joint_calls_);
        ImageTokenAndPooledFeaturesResult result{
            {{{marker, invocation}, 1, 2},
             {{ImageFeatureTokenRole::Class, 0, 0, 0, 0, 0, 0}},
             1,
             1},
            {{marker, invocation, pixel(request.image)}, "cls", "none"}};
        if (mode_ == "global_pooled" || mode_ == "unknown_image_role") {
            result = {
                {{{marker, invocation, marker - 1, invocation - 1, marker + 1, invocation + 1},
                  3,
                  2},
                 {{ImageFeatureTokenRole::GlobalPooled, 0, 0, 0, 0, 0, 0},
                  {ImageFeatureTokenRole::Patch, 0, 0, 0, 0, 0.5F, 1},
                  {ImageFeatureTokenRole::Patch, 0, 1, 0.5F, 0, 1, 1}},
                 1,
                 2},
                {{marker, invocation}, "mean", "none"}};
            if (mode_ == "unknown_image_role")
                result.tokens.tokens[0].role = static_cast<ImageFeatureTokenRole>(999);
        }
        if (mode_ == "missing_pooler")
            result.pooled.values.clear();
        return result;
    }
    SpatialFeaturesResult run(const ImageToSpatialFeaturesRequest& request,
                              ConfigView config) override {
        return {{{"stage0", {float(11 * scale(config)), pixel(request.image)}, 2, 1, 1, 2, 2}},
                2,
                2,
                {2, 4, -3, 5}};
    }
    PooledFeaturesResult run(const ImageToPooledFeaturesRequest& request,
                             ConfigView config) override {
        return {{float(12 * scale(config)), pixel(request.image)}, "cls", "none"};
    }
    SemanticEmbeddingResult run(const ImageToEmbeddingRequest& request,
                                ConfigView config) override {
        return embedding(float(13 * scale(config)), pixel(request.image));
    }
    SemanticEmbeddingResult run(const ImageTextToEmbeddingRequest& request,
                                ConfigView config) override {
        auto result = embedding(float(14 * scale(config)), pixel(request.image));
        result.values.push_back(float(request.text.size()));
        return result;
    }
    RelevanceResult run(const TextPairToRelevanceRequest& request, ConfigView config) override {
        return {float((15 + request.query.size() + 10 * request.document.size()) * scale(config)),
                ScoreKind::Unbounded};
    }
    RelevanceResult run(const TextImageToRelevanceRequest& request, ConfigView config) override {
        return {float((16 + request.query.size() + pixel(request.document)) * scale(config)),
                ScoreKind::Unbounded};
    }
    RelevanceResult run(const TextImageTextToRelevanceRequest& request,
                        ConfigView config) override {
        return {float((17 + request.query.size() + 10 * request.document_text.size() +
                       pixel(request.image)) *
                      scale(config)),
                ScoreKind::Unbounded};
    }
    LabelScoresResult run(const ImageToClassScoresRequest& request, ConfigView config) override {
        ++scalar_class_calls_;
        return class_identity({{float(18 * scale(config)), pixel(request.image)},
                               {"left", "right"},
                               ScoreKind::Logit,
                               "fixture.classes"});
    }

    std::vector<LabelScoresResult>
    run_batch(const BatchImageToClassScoresRequest& request) override {
        // Synthetic protocol batch: one method invocation, never scalar Tasks.
        // The marker is not evidence of a GPU batch or a throughput claim.
        const auto options = batch_options(request);
        const auto invocation = static_cast<float>(++batch_calls_);
        std::vector<LabelScoresResult> results;
        for (size_t i = 0; i < request.items.size(); ++i)
            results.push_back({{static_cast<float>(21 * options[i].scale),
                                probe(pixel(request.items[i].input.image), options[i]), invocation,
                                static_cast<float>(scalar_class_calls_)},
                               {"first", "second", "invocation", "scalar_calls"},
                               ScoreKind::Logit,
                               "fixture." + options[i].tag + ".classes"});
        if (mode_ == "bad_batch_count")
            results.pop_back();
        if (!results.empty())
            results.back() = class_identity(std::move(results.back()));
        return results;
    }

    std::vector<ImageTokenFeaturesResult>
    run_batch(const BatchImageToTokenFeaturesRequest& request) override {
        const auto options = batch_options(request);
        const auto invocation = static_cast<float>(++batch_calls_);
        std::vector<ImageTokenFeaturesResult> out;
        for (size_t i = 0; i < request.items.size(); ++i)
            out.push_back({{{static_cast<float>(22 * options[i].scale),
                             probe(pixel(request.items[i].input.image), options[i]), invocation,
                             static_cast<float>(220 * options[i].scale), 0, 0},
                            2,
                            3},
                           {{ImageFeatureTokenRole::Class, 0, 0, 0, 0, 0, 0},
                            {ImageFeatureTokenRole::Patch, 0, 0, 0, 0, 1, 1}},
                           1,
                           1});
        return out;
    }
    std::vector<SpatialFeaturesResult>
    run_batch(const BatchImageToSpatialFeaturesRequest& request) override {
        const auto options = batch_options(request);
        const auto invocation = static_cast<float>(++batch_calls_);
        std::vector<SpatialFeaturesResult> out;
        for (size_t i = 0; i < request.items.size(); ++i) {
            SpatialFeaturesResult result;
            for (size_t stage = 0; stage <= i; ++stage) {
                std::vector<float> values(3 * (i + 1), 0);
                values[0] = static_cast<float>(23 * options[i].scale);
                values[1] = probe(pixel(request.items[i].input.image), options[i]);
                values[2] = invocation;
                result.maps.push_back(
                    {"stage" + std::to_string(stage), std::move(values), 3, 1, i + 1, 2, 2});
            }
            result.processed_image_height = 2;
            result.processed_image_width = 2 * (i + 1);
            result.source_to_processed = {2, 4, -3, 5};
            out.push_back(std::move(result));
        }
        return out;
    }
    std::vector<PooledFeaturesResult>
    run_batch(const BatchImageToPooledFeaturesRequest& request) override {
        const auto options = batch_options(request);
        const auto invocation = static_cast<float>(++batch_calls_);
        std::vector<PooledFeaturesResult> out;
        for (size_t i = 0; i < request.items.size(); ++i)
            out.push_back({{static_cast<float>(24 * options[i].scale),
                            probe(pixel(request.items[i].input.image), options[i]), invocation},
                           "cls",
                           "none"});
        return out;
    }
    std::vector<SemanticEmbeddingResult>
    run_batch(const BatchTextToEmbeddingRequest& request) override {
        const auto options = batch_options(request);
        const auto invocation = static_cast<float>(++batch_calls_);
        std::vector<SemanticEmbeddingResult> out;
        for (size_t i = 0; i < request.items.size(); ++i)
            out.push_back(
                {{static_cast<float>(25 * options[i].scale),
                  probe(static_cast<float>(request.items[i].input.text.size()), options[i]),
                  static_cast<float>(request.items[i].input.role), invocation},
                 mode_ == "unknown_embedding_space" ? ""
                                                    : "fixture." + options[i].tag + ".embedding",
                 "mean",
                 "none"});
        return out;
    }
    std::vector<TokenFeaturesResult>
    run_batch(const BatchTextToTokenFeaturesRequest& request) override {
        const auto options = batch_options(request);
        const auto invocation = static_cast<float>(++batch_calls_);
        std::vector<TokenFeaturesResult> out;
        for (size_t i = 0; i < request.items.size(); ++i) {
            const auto& source = request.items[i].input.text;
            const auto* text = std::get_if<std::string_view>(&source);
            const auto length = text_size(source);
            TokenFeaturesResult result;
            result.features = {{}, length, 3};
            for (size_t position = 0; position < length; ++position) {
                const int64_t id = text ? static_cast<uint8_t>((*text)[position])
                                        : std::get<Span<const int32_t>>(source)[position];
                result.features.values.insert(result.features.values.end(),
                                              {static_cast<float>(26 * options[i].scale),
                                               probe(static_cast<float>(id), options[i]),
                                               invocation});
                result.tokens.push_back({id, 0, position, text != nullptr, position, position + 1});
            }
            if (mode_.rfind("padding", 0) == 0)
                append_padding(result, length);
            out.push_back(std::move(result));
        }
        return out;
    }
    std::vector<ImageTokenAndPooledFeaturesResult>
    run_batch(const BatchImageToTokenAndPooledFeaturesRequest& request) override {
        const auto options = batch_options(request);
        const auto invocation = static_cast<float>(++batch_calls_);
        std::vector<ImageTokenAndPooledFeaturesResult> out;
        for (size_t i = 0; i < request.items.size(); ++i) {
            const auto marker = static_cast<float>(27 * options[i].scale);
            ImageTokenAndPooledFeaturesResult result{
                {{{marker, probe(pixel(request.items[i].input.image), options[i]), invocation},
                  1,
                  3},
                 {{ImageFeatureTokenRole::Class, 0, 0, 0, 0, 0, 0}},
                 1,
                 1},
                {{marker, invocation}, "cls", "none"}};
            if (mode_ == "missing_batch_pooler" && i == 1)
                result.pooled.values.clear();
            out.push_back(std::move(result));
        }
        return out;
    }

    DocumentRelevanceResult run(const TextQueryDocumentsToRelevanceRequest& request,
                                ConfigView config) override {
        const auto factor = scale(config);
        DocumentRelevanceResult result;
        if (request.documents.empty())
            return result;
        // One list-operation marker, not a claim about TensorRT engine enqueues.
        const auto invocation = ++query_document_calls_;
        for (size_t i = 0; i < request.documents.size(); ++i)
            result.scores.push_back(float(factor * (1000 * invocation + 100 * request.query.size() +
                                                    10 * i + request.documents[i].size())));
        if (mode_ == "bad_count")
            result.scores.pop_back();
        return result;
    }

  private:
    LabelScoresResult class_identity(LabelScoresResult result) const {
        if (mode_ == "unnamed_classes" || mode_ == "ordinal_classes")
            result.labels.clear();
        if (mode_ == "unnamed_classes" || mode_ == "named_classes" || mode_ == "blank_classes")
            result.vocabulary_id.clear();
        if (mode_ == "blank_classes")
            result.labels.assign(result.scores.size(), "");
        if (mode_ == "empty_classes")
            result.scores.clear();
        return result;
    }

    uint64_t joint_calls_{0};
    uint64_t batch_calls_{0};
    uint64_t scalar_class_calls_{0};

    struct BatchOptions {
        double scale{1};
        bool annotate{true};
        std::string tag{"default"};
    };
    static float probe(float value, const BatchOptions& options) {
        return value + (options.annotate ? 100.0F : 0.0F);
    }
    template <class Request>
    std::vector<BatchOptions> batch_options(const Request& request) const {
        std::vector<BatchOptions> out;
        for (size_t i = 0; i < request.items.size(); ++i) {
            try {
                BatchOptions options;
                bool seen_scale = false, seen_annotate = false, seen_tag = false;
                for (const auto& entry : request.items[i].config) {
                    if (entry.name == "scale" && !seen_scale &&
                        std::holds_alternative<double>(entry.value)) {
                        options.scale = std::get<double>(entry.value);
                        seen_scale = true;
                    } else if (entry.name == "annotate" && !seen_annotate &&
                               std::holds_alternative<bool>(entry.value)) {
                        options.annotate = std::get<bool>(entry.value);
                        seen_annotate = true;
                    } else if (entry.name == "tag" && !seen_tag &&
                               std::holds_alternative<std::string_view>(entry.value) &&
                               (std::is_same_v<Request, BatchImageToClassScoresRequest> ||
                                std::is_same_v<Request, BatchTextToEmbeddingRequest>)) {
                        options.tag = std::get<std::string_view>(entry.value);
                        seen_tag = true;
                    } else
                        throw ConfigError("unknown, duplicate or mistyped batch config");
                }
                if (!std::isfinite(options.scale) || options.scale < 0)
                    throw ConfigError("batch scale must be finite and nonnegative");
                out.push_back(std::move(options));
            } catch (const ConfigError& error) {
                throw ConfigError("batch item[" + std::to_string(i) + "]: " + error.what());
            }
        }
        return out;
    }

    double scale(ConfigView options) const {
        double result = 1;
        bool seen = false;
        for (const auto& entry : options) {
            if (entry.name != "scale" || seen || config_kind(entry.value) != ConfigKind::F64)
                throw ConfigError("invalid or duplicate fixture scale");
            result = std::get<double>(entry.value);
            seen = true;
        }
        if (!std::isfinite(result) || result < 0)
            throw ConfigError("scale must be finite and nonnegative");
        return result;
    }
    static size_t text_size(const TextSource& source) {
        if (const auto* text = std::get_if<std::string_view>(&source))
            return text->size();
        return std::get<Span<const int32_t>>(source).size();
    }
    void append_padding(TokenFeaturesResult& result, uint64_t position) const {
        // Deliberately nonzero raw padded rows; transport must not trim or zero them.
        for (uint64_t column = 0; column < result.features.columns; ++column)
            result.features.values.push_back(column == 0 ? 37.5F : -91.25F);
        ++result.features.rows;
        result.tokens.push_back({99, mode_ == "padding_bad_index" ? -3 : kFeaturePaddingInputIndex,
                                 position, mode_ == "padding_offsets", 0,
                                 mode_ == "padding_offsets" ? 1U : 0U});
    }
    static TokenFeaturesResult token_features(float marker, const TextSource& source,
                                              double factor) {
        int64_t token_id = 17;
        if (const auto* ids = std::get_if<Span<const int32_t>>(&source); ids && !ids->empty())
            token_id = (*ids)[0];
        return {{{float(marker * factor), float(text_size(source))}, 1, 2},
                {{token_id, 0, 0, false, 0, 0}}};
    }
    SemanticEmbeddingResult embedding(float first, float second) const {
        return {{first, second},
                mode_ == "unknown_embedding_space" ? "" : "fixture.embedding",
                "mean",
                "none"};
    }
    static float pixel(const ImageView& image) {
        if (image.format == ImageFormat::Float32)
            return static_cast<const float*>(image.data)[0];
        return static_cast<const uint8_t*>(image.data)[0] / 255.0F;
    }
    std::string mode_;
    size_t query_document_calls_{0};
};

class MissingFixture final : public IModel {
  public:
    const char* task() const noexcept override { return "missing"; }
    std::vector<TaskInstance> task_bindings() override { return {}; }
};

} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.reader.info().family != "features_fixture")
        throw std::runtime_error("wrong feature fixture family");
    if (context.reader.info().task == "missing")
        return new MissingFixture;
    return new FeaturesFixture(context.reader.info().task);
}

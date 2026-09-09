/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/core.hpp"
#include "trtmc/features.h"
#include "trtmc/image.hpp"
#include "trtmc/matrix.hpp"
#include "trtmc/scores.hpp"
#include "trtmc/text.hpp"

#include <algorithm>
#include <cmath>

namespace trtmc {

enum class EmbeddingRole : uint32_t { Default = 0, Query = 1, Document = 2 };
using FeatureToken = trtmc_feature_token_v1;
// FeatureToken::input_index uses TRTMC_FEATURE_INPUT_PADDING / SPECIAL or 0/1.
// Padding rows, when returned by the family, remain visible and are not zeroed.
using ImageFeatureToken = trtmc_image_feature_token_v1;
using SpatialFeatureMapView = trtmc_spatial_feature_map_v1;

struct TextQueryDocumentsToRelevanceRequest {
    std::string query;
    std::vector<std::string> documents;
};

struct TextToTokenFeaturesRequest {
    TextSource text;
};
struct FeatureTextPair {
    std::string first;
    std::string second;
};
struct FeatureTokenizedPair {
    std::vector<int32_t> token_ids;
    std::vector<int32_t> segment_ids;
    std::vector<uint8_t> attention_mask;
};
struct TextPairToTokenFeaturesRequest {
    TextPairToTokenFeaturesRequest(std::string first, std::string second)
        : pair(FeatureTextPair{std::move(first), std::move(second)}) {}
    TextPairToTokenFeaturesRequest(FeatureTokenizedPair tokens) : pair(std::move(tokens)) {}
    std::variant<FeatureTextPair, FeatureTokenizedPair> pair;
};
struct TextToPooledFeaturesRequest {
    TextSource text;
};
struct TextToEmbeddingRequest {
    std::string text;
    EmbeddingRole role{EmbeddingRole::Default};
};
struct TitleBodyToEmbeddingRequest {
    std::string title;
    std::string body;
};
struct MaskedTextToTokenScoresRequest {
    TextSource text;
};
struct TextPairToPretrainingRelationScoresRequest {
    std::string first;
    std::string second;
};
struct TextToReplacedTokenScoresRequest {
    TextSource text;
};
struct TextPredictionPositionsToTokenScoresRequest {
    std::vector<int64_t> token_ids;
    std::vector<uint8_t> attention_mask;
    std::vector<int64_t> segment_ids;
    std::vector<uint8_t> blocked_attention;
    std::vector<uint64_t> prediction_positions;
};
struct ImageToTokenFeaturesRequest {
    ImageInput image;
};
struct ImageToTokenAndPooledFeaturesRequest {
    ImageInput image;
};
struct ImageToSpatialFeaturesRequest {
    ImageInput image;
};
struct ImageToPooledFeaturesRequest {
    ImageInput image;
};
struct ImageToEmbeddingRequest {
    ImageInput image;
};
struct ImageTextToEmbeddingRequest {
    ImageInput image;
    std::string text;
};
struct TextPairToRelevanceRequest {
    std::string query;
    std::string document;
};
struct TextImageToRelevanceRequest {
    std::string query;
    ImageInput document;
};
struct TextImageTextToRelevanceRequest {
    std::string query;
    ImageInput image;
    std::string document_text;
};
struct ImageToClassScoresRequest {
    ImageInput image;
};

namespace detail {
template <class Wire>
class FeatureResultOwner {
  public:
    FeatureResultOwner(std::shared_ptr<ModelState> state, trtmc_result* result) noexcept
        : owner_(std::move(state), result) {}
    FeatureResultOwner(const FeatureResultOwner&) = delete;
    FeatureResultOwner& operator=(const FeatureResultOwner&) = delete;
    FeatureResultOwner(FeatureResultOwner&& other) noexcept
        : owner_(std::move(other.owner_)), view_(std::exchange(other.view_, {})) {}
    FeatureResultOwner& operator=(FeatureResultOwner&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            view_ = std::exchange(other.view_, {});
        }
        return *this;
    }
    Wire& wire_view() noexcept { return view_; }
    const Wire& wire_view() const noexcept { return view_; }

  protected:
    ResultOwner owner_;
    Wire view_{};
};
inline FloatMatrixView feature_matrix(trtmc_f32_matrix_view_v1 matrix) noexcept {
    return {{matrix.data, static_cast<std::size_t>(matrix.count)}, matrix.rows, matrix.columns};
}
} // namespace detail

class TokenFeaturesResult : public detail::FeatureResultOwner<trtmc_token_features_view_v1> {
  public:
    using FeatureResultOwner::FeatureResultOwner;
    FloatMatrixView features() const noexcept { return detail::feature_matrix(view_.features); }
    Span<const FeatureToken> tokens() const noexcept {
        return {view_.tokens, static_cast<std::size_t>(view_.token_count)};
    }
};
class PooledFeaturesResult : public detail::FeatureResultOwner<trtmc_pooled_features_view_v1> {
  public:
    using FeatureResultOwner::FeatureResultOwner;
    Span<const float> values() const noexcept {
        return {view_.values, static_cast<std::size_t>(view_.count)};
    }
    std::string_view pooling() const { return detail::string_view(view_.pooling); }
    std::string_view normalization() const { return detail::string_view(view_.normalization); }
};
class SemanticEmbeddingResult
    : public detail::FeatureResultOwner<trtmc_semantic_embedding_view_v1> {
  public:
    using FeatureResultOwner::FeatureResultOwner;
    Span<const float> values() const noexcept {
        return {view_.values, static_cast<std::size_t>(view_.count)};
    }
    std::string_view embedding_space() const { return detail::string_view(view_.embedding_space); }
    std::string_view pooling() const { return detail::string_view(view_.pooling); }
    std::string_view normalization() const { return detail::string_view(view_.normalization); }
};
class VocabularyScoresResult : public detail::FeatureResultOwner<trtmc_vocabulary_scores_view_v1> {
  public:
    using FeatureResultOwner::FeatureResultOwner;
    FloatMatrixView logits() const noexcept { return detail::feature_matrix(view_.logits); }
    Span<const FeatureToken> positions() const noexcept {
        return {view_.positions, static_cast<std::size_t>(view_.position_count)};
    }
    std::string_view vocabulary_id() const { return detail::string_view(view_.vocabulary_id); }
};

struct RankedTokenCandidate {
    int64_t token_id;
    float logit; // Raw model score, not a softmax probability.
};
struct RankedMaskPosition {
    FeatureToken position;
    std::vector<RankedTokenCandidate> candidates;
};

// Pure caller-side ranking of the family's already-selected positions. Ties
// use ascending token ID; +/-infinity are ordered, NaN is rejected. max_candidates
// is a nonnegative upper bound. No tokenization, softmax, filtering or decoding.
inline std::vector<RankedMaskPosition>
rank_masked_tokens(const trtmc_vocabulary_scores_view_v1& input, int64_t max_candidates = 5) {
    const auto require = [](bool condition, const char* message) {
        if (!condition)
            throw Error(TRTMC_INVALID_ARGUMENT, message);
    };
    require(max_candidates >= 0, "candidate count must be nonnegative");
    const auto& matrix = input.logits;
    const auto limit = static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
    require(matrix.rows == input.position_count && matrix.rows <= limit / sizeof(FeatureToken),
            "mask position count differs from logits rows or exceeds host storage");
    require(matrix.columns <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
                (!matrix.rows ||
                 (matrix.columns && matrix.rows <= limit / sizeof(float) / matrix.columns)),
            "vocabulary shape is empty or exceeds host storage");
    require(matrix.count == matrix.rows * matrix.columns && (!matrix.count || matrix.data) &&
                (!input.position_count || input.positions),
            "vocabulary scores require matching data and position arrays");
    for (uint64_t i = 0; i < matrix.count; ++i)
        require(!std::isnan(matrix.data[i]), "cannot rank NaN vocabulary scores");
    const auto kept = static_cast<size_t>(std::min<uint64_t>(max_candidates, matrix.columns));
    std::vector<RankedMaskPosition> result;
    result.reserve(static_cast<size_t>(matrix.rows));
    std::vector<RankedTokenCandidate> candidates;
    if (kept && matrix.rows)
        candidates.reserve(static_cast<size_t>(matrix.columns));
    for (uint64_t row = 0; row < matrix.rows; ++row) {
        RankedMaskPosition ranked{input.positions[row], {}};
        if (kept) {
            candidates.clear();
            for (uint64_t id = 0; id < matrix.columns; ++id)
                candidates.push_back(
                    {static_cast<int64_t>(id), matrix.data[row * matrix.columns + id]});
            std::partial_sort(candidates.begin(), candidates.begin() + kept, candidates.end(),
                              [](const auto& a, const auto& b) {
                                  return a.logit == b.logit ? a.token_id < b.token_id
                                                            : a.logit > b.logit;
                              });
            ranked.candidates.assign(candidates.begin(), candidates.begin() + kept);
        }
        result.push_back(std::move(ranked));
    }
    return result;
}
inline std::vector<RankedMaskPosition> rank_masked_tokens(const VocabularyScoresResult& input,
                                                          int64_t max_candidates = 5) {
    return rank_masked_tokens(input.wire_view(), max_candidates);
}

class ReplacedTokenScoresResult
    : public detail::FeatureResultOwner<trtmc_replaced_token_scores_view_v1> {
  public:
    using FeatureResultOwner::FeatureResultOwner;
    Span<const float> logits() const noexcept {
        return {view_.logits, static_cast<std::size_t>(view_.token_count)};
    }
    Span<const FeatureToken> tokens() const noexcept {
        return {view_.tokens, static_cast<std::size_t>(view_.token_count)};
    }
};
class ImageTokenFeaturesResult
    : public detail::FeatureResultOwner<trtmc_image_token_features_view_v1> {
  public:
    using FeatureResultOwner::FeatureResultOwner;
    FloatMatrixView features() const noexcept { return detail::feature_matrix(view_.features); }
    Span<const ImageFeatureToken> tokens() const noexcept {
        return {view_.tokens, static_cast<std::size_t>(view_.token_count)};
    }
    uint64_t grid_rows() const noexcept { return view_.grid_rows; }
    uint64_t grid_columns() const noexcept { return view_.grid_columns; }
};
class ImageTokenAndPooledFeaturesResult
    : public detail::FeatureResultOwner<trtmc_image_token_and_pooled_features_view_v1> {
  public:
    using FeatureResultOwner::FeatureResultOwner;
    FloatMatrixView features() const noexcept {
        return detail::feature_matrix(view_.tokens.features);
    }
    Span<const ImageFeatureToken> tokens() const noexcept {
        return {view_.tokens.tokens, static_cast<size_t>(view_.tokens.token_count)};
    }
    uint64_t grid_rows() const noexcept { return view_.tokens.grid_rows; }
    uint64_t grid_columns() const noexcept { return view_.tokens.grid_columns; }
    Span<const float> pooled_values() const noexcept {
        return {view_.pooled.values, static_cast<size_t>(view_.pooled.count)};
    }
    std::string_view pooling() const { return detail::string_view(view_.pooled.pooling); }
    std::string_view normalization() const {
        return detail::string_view(view_.pooled.normalization);
    }
};
class SpatialFeaturesResult : public detail::FeatureResultOwner<trtmc_spatial_features_view_v1> {
  public:
    using FeatureResultOwner::FeatureResultOwner;
    Span<const SpatialFeatureMapView> maps() const noexcept {
        return {view_.maps, static_cast<std::size_t>(view_.map_count)};
    }
    uint64_t processed_image_height() const noexcept { return view_.processed_image_height; }
    uint64_t processed_image_width() const noexcept { return view_.processed_image_width; }
    trtmc_feature_image_transform_v1 source_to_processed() const noexcept {
        return view_.source_to_processed;
    }
};
class RelevanceResult : public detail::FeatureResultOwner<trtmc_relevance_view_v1> {
  public:
    using FeatureResultOwner::FeatureResultOwner;
    float score() const noexcept { return view_.score; }
    uint32_t kind() const noexcept { return view_.kind; }
};
class DocumentRelevanceResult
    : public detail::FeatureResultOwner<trtmc_document_relevance_view_v1> {
  public:
    using FeatureResultOwner::FeatureResultOwner;
    Span<const float> scores() const noexcept {
        return {view_.scores, static_cast<std::size_t>(view_.count)};
    }
    uint32_t kind() const noexcept { return view_.kind; }
};

class TextQueryDocumentsToRelevance {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_QUERY_DOCUMENTS_TO_RELEVANCE;
    static constexpr uint32_t kMajor = 1, kMinor = 0;
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 ||
            table->byte_size < sizeof(trtmc_text_query_documents_to_relevance_api_v1))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible document relevance Task table");
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, 1, 0);
    }
    DocumentRelevanceResult run(const TextQueryDocumentsToRelevanceRequest& input,
                                const Config& config = {}) const {
        std::vector<trtmc_string_view> documents;
        documents.reserve(input.documents.size());
        for (const auto& document : input.documents)
            documents.push_back(detail::c_string(document));
        const trtmc_text_query_documents_to_relevance_request_v1 request{
            detail::c_string(input.query), {documents.data(), documents.size()}};
        const auto entries = config.c_entries();
        const auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = api_->run(state_->handle, &request, &options, &raw, &error);
        DocumentRelevanceResult result(state_, raw);
        detail::check(state_->api, status, error);
        error = nullptr;
        status = api_->result_view(raw, &result.wire_view(), &error);
        detail::check(state_->api, status, error);
        return result;
    }

  private:
    friend class Model;
    TextQueryDocumentsToRelevance(std::shared_ptr<detail::ModelState> state,
                                  const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_text_query_documents_to_relevance_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_text_query_documents_to_relevance_api_v1* api_;
};
namespace detail {

inline trtmc_text_to_token_features_request_v1
feature_request(const TextToTokenFeaturesRequest& input) {
    return {wire_text_source(input.text)};
}
inline trtmc_text_pair_to_token_features_request_v1
feature_request(const TextPairToTokenFeaturesRequest& input) {
    trtmc_text_pair_to_token_features_request_v1 result{};
    if (const auto* text = std::get_if<FeatureTextPair>(&input.pair)) {
        result.kind = TRTMC_FEATURE_TEXT_PAIR;
        result.as.text = {c_string(text->first), c_string(text->second)};
    } else {
        const auto& tokens = std::get<FeatureTokenizedPair>(input.pair);
        result.kind = TRTMC_FEATURE_TOKENIZED_PAIR;
        result.as.tokens = {{tokens.token_ids.data(), tokens.token_ids.size()},
                            {tokens.segment_ids.data(), tokens.segment_ids.size()},
                            {tokens.attention_mask.data(), tokens.attention_mask.size()}};
    }
    return result;
}
inline trtmc_text_to_pooled_features_request_v1
feature_request(const TextToPooledFeaturesRequest& input) {
    return {wire_text_source(input.text)};
}
inline trtmc_text_to_embedding_request_v1 feature_request(const TextToEmbeddingRequest& input) {
    return {c_string(input.text), static_cast<uint32_t>(input.role)};
}
inline trtmc_title_body_to_embedding_request_v1
feature_request(const TitleBodyToEmbeddingRequest& input) {
    return {c_string(input.title), c_string(input.body)};
}
inline trtmc_masked_text_to_token_scores_request_v1
feature_request(const MaskedTextToTokenScoresRequest& input) {
    return {wire_text_source(input.text)};
}
inline trtmc_text_pair_to_pretraining_relation_scores_request_v1
feature_request(const TextPairToPretrainingRelationScoresRequest& input) {
    return {c_string(input.first), c_string(input.second)};
}
inline trtmc_text_to_replaced_token_scores_request_v1
feature_request(const TextToReplacedTokenScoresRequest& input) {
    return {wire_text_source(input.text)};
}
inline trtmc_text_prediction_positions_to_token_scores_request_v1
feature_request(const TextPredictionPositionsToTokenScoresRequest& input) {
    return {{input.token_ids.data(), input.token_ids.size()},
            {input.attention_mask.data(), input.attention_mask.size()},
            {input.segment_ids.data(), input.segment_ids.size()},
            {input.blocked_attention.data(), input.blocked_attention.size()},
            {input.prediction_positions.data(), input.prediction_positions.size()}};
}
inline trtmc_image_to_token_features_request_v1
feature_request(const ImageToTokenFeaturesRequest& input) {
    return {input.image.wire};
}
inline trtmc_image_to_spatial_features_request_v1
feature_request(const ImageToSpatialFeaturesRequest& input) {
    return {input.image.wire};
}
inline trtmc_image_to_token_and_pooled_features_request_v1
feature_request(const ImageToTokenAndPooledFeaturesRequest& input) {
    return {input.image.wire};
}
inline trtmc_image_to_pooled_features_request_v1
feature_request(const ImageToPooledFeaturesRequest& input) {
    return {input.image.wire};
}
inline trtmc_image_to_embedding_request_v1 feature_request(const ImageToEmbeddingRequest& input) {
    return {input.image.wire};
}
inline trtmc_image_text_to_embedding_request_v1
feature_request(const ImageTextToEmbeddingRequest& input) {
    return {input.image.wire, c_string(input.text)};
}
inline trtmc_text_pair_to_relevance_request_v1
feature_request(const TextPairToRelevanceRequest& input) {
    return {c_string(input.query), c_string(input.document)};
}
inline trtmc_text_image_to_relevance_request_v1
feature_request(const TextImageToRelevanceRequest& input) {
    return {c_string(input.query), input.document.wire};
}
inline trtmc_text_image_text_to_relevance_request_v1
feature_request(const TextImageTextToRelevanceRequest& input) {
    return {c_string(input.query), input.image.wire, c_string(input.document_text)};
}
inline trtmc_image_to_class_scores_request_v1
feature_request(const ImageToClassScoresRequest& input) {
    return {input.image.wire};
}

// The template only repeats ownership and C-table invocation. Each trait below
// fixes a distinct request, result, interface ID and table; no runtime type guessing.
template <class Traits>
class FeatureTask {
  public:
    static constexpr std::string_view kTask = Traits::kTask;
    static constexpr uint32_t kMajor = 1, kMinor = 0;
    using Request = typename Traits::Request;
    using Result = typename Traits::Result;
    using Table = typename Traits::Table;
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 || table->byte_size < sizeof(Table))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible feature Task table");
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, 1, 0);
    }
    Result run(const Request& input, const Config& config = {}) const {
        const auto request = feature_request(input);
        const auto entries = config.c_entries();
        const auto options = entries.view();
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = api_->run(state_->handle, &request, &options, &raw, &error);
        Result result(state_, raw);
        check(state_->api, status, error);
        error = nullptr;
        status = api_->result_view(raw, &result.wire_view(), &error);
        check(state_->api, status, error);
        return result;
    }

  private:
    friend class ::trtmc::Model;
    FeatureTask(std::shared_ptr<ModelState> state, const trtmc_api_header* table) noexcept
        : state_(std::move(state)), api_(reinterpret_cast<const Table*>(table)) {}
    std::shared_ptr<ModelState> state_;
    const Table* api_;
};

struct TextToTokenFeaturesTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_TO_TOKEN_FEATURES;
    using Request = TextToTokenFeaturesRequest;
    using Result = TokenFeaturesResult;
    using Table = trtmc_text_to_token_features_api_v1;
};
struct TextPairToTokenFeaturesTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_PAIR_TO_TOKEN_FEATURES;
    using Request = TextPairToTokenFeaturesRequest;
    using Result = TokenFeaturesResult;
    using Table = trtmc_text_pair_to_token_features_api_v1;
};
struct TextToPooledFeaturesTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_TO_POOLED_FEATURES;
    using Request = TextToPooledFeaturesRequest;
    using Result = PooledFeaturesResult;
    using Table = trtmc_text_to_pooled_features_api_v1;
};
struct TextToEmbeddingTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_TO_EMBEDDING;
    using Request = TextToEmbeddingRequest;
    using Result = SemanticEmbeddingResult;
    using Table = trtmc_text_to_embedding_api_v1;
};
struct TitleBodyToEmbeddingTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_TITLE_BODY_TO_EMBEDDING;
    using Request = TitleBodyToEmbeddingRequest;
    using Result = SemanticEmbeddingResult;
    using Table = trtmc_title_body_to_embedding_api_v1;
};
struct MaskedTextToTokenScoresTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_MASKED_TEXT_TO_TOKEN_SCORES;
    using Request = MaskedTextToTokenScoresRequest;
    using Result = VocabularyScoresResult;
    using Table = trtmc_masked_text_to_token_scores_api_v1;
};
struct TextPairToPretrainingRelationScoresTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_PAIR_TO_PRETRAINING_RELATION_SCORES;
    using Request = TextPairToPretrainingRelationScoresRequest;
    using Result = LabelScoresResult;
    using Table = trtmc_text_pair_to_pretraining_relation_scores_api_v1;
};
struct TextToReplacedTokenScoresTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_TO_REPLACED_TOKEN_SCORES;
    using Request = TextToReplacedTokenScoresRequest;
    using Result = ReplacedTokenScoresResult;
    using Table = trtmc_text_to_replaced_token_scores_api_v1;
};
struct TextPredictionPositionsToTokenScoresTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_PREDICTION_POSITIONS_TO_TOKEN_SCORES;
    using Request = TextPredictionPositionsToTokenScoresRequest;
    using Result = VocabularyScoresResult;
    using Table = trtmc_text_prediction_positions_to_token_scores_api_v1;
};
struct ImageToTokenFeaturesTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_IMAGE_TO_TOKEN_FEATURES;
    using Request = ImageToTokenFeaturesRequest;
    using Result = ImageTokenFeaturesResult;
    using Table = trtmc_image_to_token_features_api_v1;
};
struct ImageToSpatialFeaturesTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_IMAGE_TO_SPATIAL_FEATURES;
    using Request = ImageToSpatialFeaturesRequest;
    using Result = SpatialFeaturesResult;
    using Table = trtmc_image_to_spatial_features_api_v1;
};
struct ImageToPooledFeaturesTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_IMAGE_TO_POOLED_FEATURES;
    using Request = ImageToPooledFeaturesRequest;
    using Result = PooledFeaturesResult;
    using Table = trtmc_image_to_pooled_features_api_v1;
};
struct ImageToEmbeddingTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_IMAGE_TO_EMBEDDING;
    using Request = ImageToEmbeddingRequest;
    using Result = SemanticEmbeddingResult;
    using Table = trtmc_image_to_embedding_api_v1;
};
struct ImageTextToEmbeddingTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_IMAGE_TEXT_TO_EMBEDDING;
    using Request = ImageTextToEmbeddingRequest;
    using Result = SemanticEmbeddingResult;
    using Table = trtmc_image_text_to_embedding_api_v1;
};
struct TextPairToRelevanceTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_PAIR_TO_RELEVANCE;
    using Request = TextPairToRelevanceRequest;
    using Result = RelevanceResult;
    using Table = trtmc_text_pair_to_relevance_api_v1;
};
struct TextImageToRelevanceTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_IMAGE_TO_RELEVANCE;
    using Request = TextImageToRelevanceRequest;
    using Result = RelevanceResult;
    using Table = trtmc_text_image_to_relevance_api_v1;
};
struct TextImageTextToRelevanceTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_IMAGE_TEXT_TO_RELEVANCE;
    using Request = TextImageTextToRelevanceRequest;
    using Result = RelevanceResult;
    using Table = trtmc_text_image_text_to_relevance_api_v1;
};
struct ImageToTokenAndPooledFeaturesTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_IMAGE_TO_TOKEN_AND_POOLED_FEATURES;
    using Request = ImageToTokenAndPooledFeaturesRequest;
    using Result = ImageTokenAndPooledFeaturesResult;
    using Table = trtmc_image_to_token_and_pooled_features_api_v1;
};
struct ImageToClassScoresTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_IMAGE_TO_CLASS_SCORES;
    using Request = ImageToClassScoresRequest;
    using Result = LabelScoresResult;
    using Table = trtmc_image_to_class_scores_api_v1;
};

} // namespace detail

using TextToTokenFeatures = detail::FeatureTask<detail::TextToTokenFeaturesTraits>;
using TextPairToTokenFeatures = detail::FeatureTask<detail::TextPairToTokenFeaturesTraits>;
using TextToPooledFeatures = detail::FeatureTask<detail::TextToPooledFeaturesTraits>;
using TextToEmbedding = detail::FeatureTask<detail::TextToEmbeddingTraits>;
using TitleBodyToEmbedding = detail::FeatureTask<detail::TitleBodyToEmbeddingTraits>;
using MaskedTextToTokenScores = detail::FeatureTask<detail::MaskedTextToTokenScoresTraits>;
using TextPairToPretrainingRelationScores =
    detail::FeatureTask<detail::TextPairToPretrainingRelationScoresTraits>;
using TextToReplacedTokenScores = detail::FeatureTask<detail::TextToReplacedTokenScoresTraits>;
using TextPredictionPositionsToTokenScores =
    detail::FeatureTask<detail::TextPredictionPositionsToTokenScoresTraits>;
using ImageToTokenFeatures = detail::FeatureTask<detail::ImageToTokenFeaturesTraits>;
using ImageToSpatialFeatures = detail::FeatureTask<detail::ImageToSpatialFeaturesTraits>;
using ImageToPooledFeatures = detail::FeatureTask<detail::ImageToPooledFeaturesTraits>;
using ImageToEmbedding = detail::FeatureTask<detail::ImageToEmbeddingTraits>;
using ImageTextToEmbedding = detail::FeatureTask<detail::ImageTextToEmbeddingTraits>;
using TextPairToRelevance = detail::FeatureTask<detail::TextPairToRelevanceTraits>;
using TextImageToRelevance = detail::FeatureTask<detail::TextImageToRelevanceTraits>;
using TextImageTextToRelevance = detail::FeatureTask<detail::TextImageTextToRelevanceTraits>;
using ImageToClassScores = detail::FeatureTask<detail::ImageToClassScoresTraits>;
using ImageToTokenAndPooledFeatures =
    detail::FeatureTask<detail::ImageToTokenAndPooledFeaturesTraits>;

struct BatchImageToClassScoresItem {
    ImageToClassScoresRequest input;
    Config config{};
};
struct BatchImageToClassScoresRequest {
    std::vector<BatchImageToClassScoresItem> items;
};

namespace detail {
template <class View>
class FeatureBatchResult {
  public:
    using Count = trtmc_status(TRTMC_CALL*)(const trtmc_result*, uint64_t*, trtmc_error**);
    using ItemView = trtmc_status(TRTMC_CALL*)(const trtmc_result*, uint64_t, View*, trtmc_error**);
    FeatureBatchResult(ResultOwner owner, Count count, ItemView item)
        : owner_(std::move(owner)), item_(item) {
        trtmc_error* error = nullptr;
        const auto status = count(owner_.get(), &count_, &error);
        check(owner_.api(), status, error);
    }
    FeatureBatchResult(const FeatureBatchResult&) = delete;
    FeatureBatchResult& operator=(const FeatureBatchResult&) = delete;
    FeatureBatchResult(FeatureBatchResult&& other) noexcept
        : owner_(std::move(other.owner_)), item_(other.item_),
          count_(std::exchange(other.count_, 0)) {}
    FeatureBatchResult& operator=(FeatureBatchResult&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            item_ = other.item_;
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }
    uint64_t size() const noexcept { return count_; }
    // Returned typed views borrow this batch owner, not the original model.
    View operator[](uint64_t index) const {
        View view{};
        trtmc_error* error = nullptr;
        const auto status = item_(owner_.get(), index, &view, &error);
        check(owner_.api(), status, error);
        return view;
    }

  private:
    ResultOwner owner_;
    ItemView item_;
    uint64_t count_{0};
};
template <class Traits>
class FeatureBatchTask {
  public:
    static constexpr std::string_view kTask = Traits::kTask;
    static constexpr uint32_t kMajor = 1, kMinor = 0;
    using Request = typename Traits::Request;
    using Result = FeatureBatchResult<typename Traits::View>;
    static void validate_table(const trtmc_api_header* table) {
        if (!table || table->major != 1 || table->minor != 0 ||
            table->byte_size < sizeof(typename Traits::Table))
            throw Error(TRTMC_VERSION_MISMATCH, "incompatible batch feature Task table");
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(model_, kTask, kMajor, kMinor);
    }
    Result run(const Request& request) const {
        std::vector<Config::CEntries> configs;
        std::vector<typename Traits::WireItem> items;
        configs.reserve(request.items.size());
        items.reserve(request.items.size());
        for (const auto& item : request.items) {
            configs.push_back(item.config.c_entries());
            items.push_back({feature_request(item.input), configs.back().view()});
        }
        const typename Traits::WireRequest input{items.data(), items.size()};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        const auto status = api_->run(model_->handle, &input, &raw, &error);
        ResultOwner owner(model_, raw);
        check(model_->api, status, error);
        return Result(std::move(owner), api_->result_count, api_->result_item_view);
    }

  private:
    friend class ::trtmc::Model;
    FeatureBatchTask(std::shared_ptr<ModelState> model, const trtmc_api_header* table)
        : model_(std::move(model)), api_(reinterpret_cast<const typename Traits::Table*>(table)) {}
    std::shared_ptr<ModelState> model_;
    const typename Traits::Table* api_;
};
struct BatchImageToClassScoresTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_IMAGE_TO_CLASS_SCORES;
    using Request = BatchImageToClassScoresRequest;
    using WireItem = trtmc_batch_image_to_class_scores_item_v1;
    using WireRequest = trtmc_batch_image_to_class_scores_request_v1;
    using Table = trtmc_batch_image_to_class_scores_api_v1;
    using View = trtmc_label_scores_view_v1;
};
} // namespace detail
using BatchImageToClassScores = detail::FeatureBatchTask<detail::BatchImageToClassScoresTraits>;

struct BatchImageToTokenFeaturesItem {
    ImageToTokenFeaturesRequest input;
    Config config{};
};
struct BatchImageToTokenFeaturesRequest {
    std::vector<BatchImageToTokenFeaturesItem> items;
};
namespace detail {
struct BatchImageToTokenFeaturesTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_IMAGE_TO_TOKEN_FEATURES;
    using Request = BatchImageToTokenFeaturesRequest;
    using WireItem = trtmc_batch_image_to_token_features_item_v1;
    using WireRequest = trtmc_batch_image_to_token_features_request_v1;
    using Table = trtmc_batch_image_to_token_features_api_v1;
    using View = trtmc_image_token_features_view_v1;
};
} // namespace detail
using BatchImageToTokenFeatures = detail::FeatureBatchTask<detail::BatchImageToTokenFeaturesTraits>;
struct BatchImageToSpatialFeaturesItem {
    ImageToSpatialFeaturesRequest input;
    Config config{};
};
struct BatchImageToSpatialFeaturesRequest {
    std::vector<BatchImageToSpatialFeaturesItem> items;
};
namespace detail {
struct BatchImageToSpatialFeaturesTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_IMAGE_TO_SPATIAL_FEATURES;
    using Request = BatchImageToSpatialFeaturesRequest;
    using WireItem = trtmc_batch_image_to_spatial_features_item_v1;
    using WireRequest = trtmc_batch_image_to_spatial_features_request_v1;
    using Table = trtmc_batch_image_to_spatial_features_api_v1;
    using View = trtmc_spatial_features_view_v1;
};
} // namespace detail
using BatchImageToSpatialFeatures =
    detail::FeatureBatchTask<detail::BatchImageToSpatialFeaturesTraits>;
struct BatchImageToPooledFeaturesItem {
    ImageToPooledFeaturesRequest input;
    Config config{};
};
struct BatchImageToPooledFeaturesRequest {
    std::vector<BatchImageToPooledFeaturesItem> items;
};
namespace detail {
struct BatchImageToPooledFeaturesTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_IMAGE_TO_POOLED_FEATURES;
    using Request = BatchImageToPooledFeaturesRequest;
    using WireItem = trtmc_batch_image_to_pooled_features_item_v1;
    using WireRequest = trtmc_batch_image_to_pooled_features_request_v1;
    using Table = trtmc_batch_image_to_pooled_features_api_v1;
    using View = trtmc_pooled_features_view_v1;
};
} // namespace detail
using BatchImageToPooledFeatures =
    detail::FeatureBatchTask<detail::BatchImageToPooledFeaturesTraits>;
struct BatchTextToEmbeddingItem {
    TextToEmbeddingRequest input;
    Config config{};
};
struct BatchTextToEmbeddingRequest {
    std::vector<BatchTextToEmbeddingItem> items;
};
namespace detail {
struct BatchTextToEmbeddingTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_TEXT_TO_EMBEDDING;
    using Request = BatchTextToEmbeddingRequest;
    using WireItem = trtmc_batch_text_to_embedding_item_v1;
    using WireRequest = trtmc_batch_text_to_embedding_request_v1;
    using Table = trtmc_batch_text_to_embedding_api_v1;
    using View = trtmc_semantic_embedding_view_v1;
};
} // namespace detail
using BatchTextToEmbedding = detail::FeatureBatchTask<detail::BatchTextToEmbeddingTraits>;
struct BatchTextToTokenFeaturesItem {
    TextToTokenFeaturesRequest input;
    Config config{};
};
struct BatchTextToTokenFeaturesRequest {
    std::vector<BatchTextToTokenFeaturesItem> items;
};
namespace detail {
struct BatchTextToTokenFeaturesTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_TEXT_TO_TOKEN_FEATURES;
    using Request = BatchTextToTokenFeaturesRequest;
    using WireItem = trtmc_batch_text_to_token_features_item_v1;
    using WireRequest = trtmc_batch_text_to_token_features_request_v1;
    using Table = trtmc_batch_text_to_token_features_api_v1;
    using View = trtmc_token_features_view_v1;
};
} // namespace detail
using BatchTextToTokenFeatures = detail::FeatureBatchTask<detail::BatchTextToTokenFeaturesTraits>;
struct BatchImageToTokenAndPooledFeaturesItem {
    ImageToTokenAndPooledFeaturesRequest input;
    Config config{};
};
struct BatchImageToTokenAndPooledFeaturesRequest {
    std::vector<BatchImageToTokenAndPooledFeaturesItem> items;
};
namespace detail {
struct BatchImageToTokenAndPooledFeaturesTraits {
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_IMAGE_TO_TOKEN_AND_POOLED_FEATURES;
    using Request = BatchImageToTokenAndPooledFeaturesRequest;
    using WireItem = trtmc_batch_image_to_token_and_pooled_features_item_v1;
    using WireRequest = trtmc_batch_image_to_token_and_pooled_features_request_v1;
    using Table = trtmc_batch_image_to_token_and_pooled_features_api_v1;
    using View = trtmc_image_token_and_pooled_features_view_v1;
};
} // namespace detail
using BatchImageToTokenAndPooledFeatures =
    detail::FeatureBatchTask<detail::BatchImageToTokenAndPooledFeaturesTraits>;

} // namespace trtmc

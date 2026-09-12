/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/features.h"

#include "api_internal.h"
#include "trtmc/internal/features.h"

#include <cmath>
#include <limits>
#include <type_traits>

namespace trtmc::api {
namespace {

void result_require(bool condition, const char* message) {
    if (!condition)
        throw ApiFailure{TRTMC_INTERNAL_ERROR, message};
}

void require_class_identity(const internal::LabelScoresResult& result) {
    result_require(!result.scores.empty(), "image classification requires class scores");
    bool named = result.labels.size() == result.scores.size();
    for (const auto& label : result.labels)
        named = named && !label.empty();
    result_require(named || !result.vocabulary_id.empty(),
                   "image class scores require labels or an explicit vocabulary identity");
}

trtmc_f32_matrix_view_v1 matrix_view(const internal::FloatMatrix& matrix) {
    result_require(matrix.columns > 0, "feature matrix has zero columns");
    result_require(matrix.rows <=
                       static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
                           matrix.columns / sizeof(float),
                   "feature matrix shape overflows host storage");
    result_require(matrix.values.size() == matrix.rows * matrix.columns,
                   "feature matrix data does not match its shape");
    return {matrix.values.data(), matrix.values.size(), matrix.rows, matrix.columns};
}

std::vector<trtmc_feature_token_v1> token_views(const std::vector<internal::FeatureToken>& tokens) {
    std::vector<trtmc_feature_token_v1> views;
    views.reserve(tokens.size());
    for (const auto& token : tokens) {
        result_require(token.input_index >= internal::kFeaturePaddingInputIndex &&
                           token.input_index <= 1,
                       "invalid feature token input index");
        result_require(token.input_index != internal::kFeaturePaddingInputIndex ||
                           !token.has_byte_offsets,
                       "padding feature tokens cannot have input byte offsets");
        result_require(!token.has_byte_offsets || token.byte_begin <= token.byte_end,
                       "invalid feature token byte interval");
        views.push_back({token.token_id, token.input_index, token.token_index,
                         token.has_byte_offsets ? 1U : 0U, token.byte_begin, token.byte_end});
    }
    return views;
}

struct TokenFeaturesStorage final : ResultStorage {
    explicit TokenFeaturesStorage(internal::TokenFeaturesResult result)
        : value(std::move(result)), tokens(token_views(value.tokens)) {
        const auto matrix = matrix_view(value.features);
        result_require(matrix.rows == tokens.size(), "token feature mapping row count mismatch");
        view = {matrix, tokens.data(), tokens.size()};
    }
    internal::TokenFeaturesResult value;
    std::vector<trtmc_feature_token_v1> tokens;
    trtmc_token_features_view_v1 view{};
};
struct PooledFeaturesStorage final : ResultStorage {
    explicit PooledFeaturesStorage(internal::PooledFeaturesResult result)
        : value(std::move(result)) {
        result_require(!value.values.empty() && !value.pooling.empty() &&
                           !value.normalization.empty(),
                       "pooled features require values and pooling/normalization metadata");
        view = {value.values.data(), value.values.size(), borrowed_string(value.pooling),
                borrowed_string(value.normalization)};
    }
    internal::PooledFeaturesResult value;
    trtmc_pooled_features_view_v1 view{};
};
struct SemanticEmbeddingStorage final : ResultStorage {
    explicit SemanticEmbeddingStorage(internal::SemanticEmbeddingResult result)
        : value(std::move(result)) {
        result_require(!value.values.empty() && !value.pooling.empty() &&
                           !value.normalization.empty(),
                       "semantic embedding requires values and pooling/normalization metadata");
        view = {value.values.data(), value.values.size(), borrowed_string(value.embedding_space),
                borrowed_string(value.pooling), borrowed_string(value.normalization)};
    }
    internal::SemanticEmbeddingResult value;
    trtmc_semantic_embedding_view_v1 view{};
};
struct HeadScoresStorage final : ResultStorage {
    explicit HeadScoresStorage(internal::HeadScoresResult result) : value(std::move(result)) {
        result_require(!value.shape.empty() && !value.pooling.empty() &&
                           !value.normalization.empty(),
                       "head scores require shape and pooling/normalization metadata");
        std::size_t count = 1;
        for (const auto dimension : value.shape) {
            result_require(dimension > 0 &&
                               dimension <= static_cast<std::size_t>(
                                                std::numeric_limits<std::ptrdiff_t>::max()) /
                                                sizeof(float) / count,
                           "head score shape is zero or overflows host storage");
            count *= static_cast<std::size_t>(dimension);
        }
        result_require(count == value.values.size(), "head score data does not match its shape");
        result_require(value.kind == internal::ScoreKind::Logit ||
                           value.kind == internal::ScoreKind::Probability ||
                           value.kind == internal::ScoreKind::Unbounded,
                       "head scores have an unknown score kind");
        view = {value.values.data(),
                value.values.size(),
                value.shape.data(),
                value.shape.size(),
                static_cast<uint32_t>(value.kind),
                borrowed_string(value.pooling),
                borrowed_string(value.normalization)};
    }
    internal::HeadScoresResult value;
    trtmc_head_scores_view_v1 view{};
};
struct VocabularyScoresStorage final : ResultStorage {
    explicit VocabularyScoresStorage(internal::VocabularyScoresResult result)
        : value(std::move(result)), positions(token_views(value.positions)) {
        const auto matrix = matrix_view(value.logits);
        result_require(matrix.rows == positions.size() && !value.vocabulary_id.empty(),
                       "vocabulary score mapping or vocabulary identity is missing");
        view = {matrix, positions.data(), positions.size(), borrowed_string(value.vocabulary_id)};
    }
    internal::VocabularyScoresResult value;
    std::vector<trtmc_feature_token_v1> positions;
    trtmc_vocabulary_scores_view_v1 view{};
};
struct ReplacedTokenScoresStorage final : ResultStorage {
    explicit ReplacedTokenScoresStorage(internal::ReplacedTokenScoresResult result)
        : value(std::move(result)), tokens(token_views(value.tokens)) {
        result_require(value.logits.size() == tokens.size(),
                       "replaced-token mapping count mismatch");
        view = {value.logits.data(), tokens.data(), tokens.size()};
    }
    internal::ReplacedTokenScoresResult value;
    std::vector<trtmc_feature_token_v1> tokens;
    trtmc_replaced_token_scores_view_v1 view{};
};
struct ImageTokenFeaturesStorage final : ResultStorage {
    explicit ImageTokenFeaturesStorage(internal::ImageTokenFeaturesResult result)
        : value(std::move(result)) {
        const auto matrix = matrix_view(value.features);
        result_require(matrix.rows == value.tokens.size(),
                       "image token mapping row count mismatch");
        tokens.reserve(value.tokens.size());
        for (const auto& token : value.tokens) {
            const auto role = static_cast<uint32_t>(token.role);
            result_require(role >= TRTMC_IMAGE_TOKEN_PATCH &&
                               role <= TRTMC_IMAGE_TOKEN_GLOBAL_POOLED,
                           "unknown image feature token role");
            if (token.role == internal::ImageFeatureTokenRole::Patch) {
                result_require(token.grid_row < value.grid_rows &&
                                   token.grid_column < value.grid_columns,
                               "patch token lies outside declared grid");
                result_require(std::isfinite(token.x_min) && std::isfinite(token.y_min) &&
                                   std::isfinite(token.x_max) && std::isfinite(token.y_max) &&
                                   token.x_min >= 0 && token.y_min >= 0 && token.x_max <= 1 &&
                                   token.y_max <= 1 && token.x_min <= token.x_max &&
                                   token.y_min <= token.y_max,
                               "patch token has invalid normalized input footprint");
            }
            tokens.push_back({role, token.grid_row, token.grid_column, token.x_min, token.y_min,
                              token.x_max, token.y_max});
        }
        view = {matrix, tokens.data(), tokens.size(), value.grid_rows, value.grid_columns};
    }
    internal::ImageTokenFeaturesResult value;
    std::vector<trtmc_image_feature_token_v1> tokens;
    trtmc_image_token_features_view_v1 view{};
};
struct ImageTokenAndPooledFeaturesStorage final : ResultStorage {
    explicit ImageTokenAndPooledFeaturesStorage(internal::ImageTokenAndPooledFeaturesResult result)
        : tokens(std::move(result.tokens)), pooled(std::move(result.pooled)) {
        view = {tokens.view, pooled.view};
    }
    ImageTokenFeaturesStorage tokens;
    PooledFeaturesStorage pooled;
    trtmc_image_token_and_pooled_features_view_v1 view{};
};
struct SpatialFeaturesStorage final : ResultStorage {
    explicit SpatialFeaturesStorage(internal::SpatialFeaturesResult result)
        : value(std::move(result)) {
        result_require(value.processed_image_height > 0 && value.processed_image_width > 0,
                       "spatial features require processed-image dimensions");
        maps.reserve(value.maps.size());
        for (const auto& map : value.maps) {
            result_require(map.channels > 0 && map.height > 0 && map.width > 0,
                           "spatial map dimensions must be positive");
            const auto limit =
                static_cast<uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
            result_require(map.height <= limit / map.width &&
                               map.channels <= limit / map.width / map.height,
                           "spatial feature map shape overflows");
            result_require(map.values.size() == map.channels * map.height * map.width &&
                               !map.name.empty() && std::isfinite(map.stride_y) &&
                               map.stride_y > 0 && std::isfinite(map.stride_x) && map.stride_x > 0,
                           "spatial map data, name or stride is invalid");
            maps.push_back({borrowed_string(map.name), map.values.data(), map.values.size(),
                            map.channels, map.height, map.width, map.stride_y, map.stride_x});
        }
        const auto& transform = value.source_to_processed;
        result_require(std::isfinite(transform.scale_x) && transform.scale_x != 0 &&
                           std::isfinite(transform.scale_y) && transform.scale_y != 0 &&
                           std::isfinite(transform.offset_x) && std::isfinite(transform.offset_y),
                       "spatial source-to-processed transform is invalid");
        view = {maps.data(),
                maps.size(),
                value.processed_image_height,
                value.processed_image_width,
                {transform.scale_x, transform.scale_y, transform.offset_x, transform.offset_y}};
    }
    internal::SpatialFeaturesResult value;
    std::vector<trtmc_spatial_feature_map_v1> maps;
    trtmc_spatial_features_view_v1 view{};
};
struct RelevanceStorage final : ResultStorage {
    explicit RelevanceStorage(internal::RelevanceResult result) {
        const auto kind = static_cast<uint32_t>(result.kind);
        result_require(kind >= TRTMC_SCORE_LOGIT && kind <= TRTMC_SCORE_UNBOUNDED,
                       "unknown relevance score kind");
        view = {result.score, kind};
    }
    trtmc_relevance_view_v1 view{};
};
struct DocumentRelevanceStorage final : ResultStorage {
    explicit DocumentRelevanceStorage(internal::DocumentRelevanceResult result)
        : value(std::move(result)) {
        const auto kind = static_cast<uint32_t>(value.kind);
        result_require(kind >= TRTMC_SCORE_LOGIT && kind <= TRTMC_SCORE_UNBOUNDED,
                       "unknown document relevance score kind");
        view = {value.scores.data(), value.scores.size(), kind};
    }
    internal::DocumentRelevanceResult value;
    trtmc_document_relevance_view_v1 view{};
};

template <class Storage, class View>
trtmc_status TRTMC_CALL result_view(const trtmc_result* result, View* out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out != nullptr, "feature result output is null");
        *out = require_result<Storage>(result).view;
    });
}

internal::TextToTokenFeaturesRequest convert(const trtmc_text_to_token_features_request_v1& input) {
    return {text_source(input.text)};
}

internal::TextPairToTokenFeaturesRequest
convert(const trtmc_text_pair_to_token_features_request_v1& input) {
    if (input.kind == TRTMC_FEATURE_TEXT_PAIR)
        return {internal::FeatureTextPair{string_view(input.as.text.first),
                                          string_view(input.as.text.second)}};
    require(input.kind == TRTMC_FEATURE_TOKENIZED_PAIR, "unknown text-pair representation");
    const auto& source = input.as.tokens;
    const auto ids = checked_span(source.token_ids.data, source.token_ids.size);
    const auto segments = checked_span(source.segment_ids.data, source.segment_ids.size);
    const auto mask = checked_span(source.attention_mask.data, source.attention_mask.size);
    require(!ids.empty() && segments.size() == ids.size(),
            "tokenized pair requires segment IDs matching its token count");
    require(mask.empty() || mask.size() == ids.size(),
            "tokenized pair attention mask length differs from token count");
    for (const auto value : mask)
        require(value <= 1, "tokenized pair attention mask must contain zero or one");
    return {internal::FeatureTokenizedPair{ids, segments, mask}};
}

internal::TextToPooledFeaturesRequest
convert(const trtmc_text_to_pooled_features_request_v1& input) {
    return {text_source(input.text)};
}
internal::TextToHeadScoresRequest convert(const trtmc_text_to_head_scores_request_v1& input) {
    return {text_source(input.text)};
}

internal::TextToEmbeddingRequest convert(const trtmc_text_to_embedding_request_v1& input) {
    require(input.role <= TRTMC_EMBEDDING_DOCUMENT, "unknown embedding input role");
    return {string_view(input.text), static_cast<internal::EmbeddingRole>(input.role)};
}

internal::TitleBodyToEmbeddingRequest
convert(const trtmc_title_body_to_embedding_request_v1& input) {
    return {string_view(input.title), string_view(input.body)};
}

internal::MaskedTextToTokenScoresRequest
convert(const trtmc_masked_text_to_token_scores_request_v1& input) {
    return {text_source(input.text)};
}

internal::TextPairToPretrainingRelationScoresRequest
convert(const trtmc_text_pair_to_pretraining_relation_scores_request_v1& input) {
    return {string_view(input.first), string_view(input.second)};
}

internal::TextToReplacedTokenScoresRequest
convert(const trtmc_text_to_replaced_token_scores_request_v1& input) {
    return {text_source(input.text)};
}

internal::TextPredictionPositionsToTokenScoresRequest
convert(const trtmc_text_prediction_positions_to_token_scores_request_v1& input) {
    const auto ids = checked_span(input.token_ids.data, input.token_ids.size);
    const auto mask = checked_span(input.attention_mask.data, input.attention_mask.size);
    const auto segments = checked_span(input.segment_ids.data, input.segment_ids.size);
    const auto blocked = checked_span(input.blocked_attention.data, input.blocked_attention.size);
    const auto positions =
        checked_span(input.prediction_positions.data, input.prediction_positions.size);
    require(!ids.empty() && !positions.empty(),
            "prediction-position input must contain tokens and positions");
    const auto length = ids.size();
    require(mask.empty() || mask.size() == length,
            "attention mask length differs from token count");
    require(segments.empty() || segments.size() == length,
            "segment IDs length differs from token count");
    checked_size(length, length);
    require(blocked.size() == length * length, "blocked attention must be a square token matrix");
    for (const auto value : mask)
        require(value <= 1, "attention mask must contain zero or one");
    for (const auto value : blocked)
        require(value <= 1, "blocked attention must contain zero or one");
    for (const auto position : positions)
        require(position < length, "prediction position is out of range");
    return {ids, mask, segments, blocked, positions};
}

internal::ImageToTokenFeaturesRequest
convert(const trtmc_image_to_token_features_request_v1& input) {
    return {image_input(input.image)};
}

internal::ImageToSpatialFeaturesRequest
convert(const trtmc_image_to_spatial_features_request_v1& input) {
    return {image_input(input.image)};
}

internal::ImageToPooledFeaturesRequest
convert(const trtmc_image_to_pooled_features_request_v1& input) {
    return {image_input(input.image)};
}

internal::ImageToEmbeddingRequest convert(const trtmc_image_to_embedding_request_v1& input) {
    return {image_input(input.image)};
}

internal::ImageTextToEmbeddingRequest
convert(const trtmc_image_text_to_embedding_request_v1& input) {
    return {image_input(input.image), string_view(input.text)};
}

internal::TextPairToRelevanceRequest convert(const trtmc_text_pair_to_relevance_request_v1& input) {
    return {string_view(input.query), string_view(input.document)};
}

internal::TextImageToRelevanceRequest
convert(const trtmc_text_image_to_relevance_request_v1& input) {
    return {string_view(input.query), image_input(input.document)};
}

internal::TextImageTextToRelevanceRequest
convert(const trtmc_text_image_text_to_relevance_request_v1& input) {
    return {string_view(input.query), image_input(input.image), string_view(input.document_text)};
}

internal::ImageToTokenAndPooledFeaturesRequest
convert(const trtmc_image_to_token_and_pooled_features_request_v1& input) {
    return {image_input(input.image)};
}
internal::ImageToClassScoresRequest convert(const trtmc_image_to_class_scores_request_v1& input) {
    return {image_input(input.image)};
}

template <class Interface, class Request, class Storage>
trtmc_status TRTMC_CALL run(trtmc_model* model, const Request* input,
                            const trtmc_config_view_v1* config, trtmc_result** out,
                            trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(input != nullptr && out != nullptr, "feature input or result output is null");
        const auto request = convert(*input);
        const ConvertedConfig options(config);

        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<Interface>(model, Interface::kTask);
        validate_task_config(model_owner(model), internal::contract_key<Interface>(),
                             options.view());
        auto result = family.run(request, options.view());
        if constexpr (std::is_same_v<Interface, internal::ITextPairToPretrainingRelationScores>)
            result_require(!result.scores.empty() && result.labels.size() == result.scores.size(),
                           "pretraining relation scores require explicit relation labels");
        if constexpr (std::is_same_v<Interface, internal::IImageToClassScores>)
            require_class_identity(result);
        *out = make_result<Storage>(std::move(result));
    });
}

const trtmc_text_to_token_features_api_v1 text_to_token_features_api = {
    {1, 0, sizeof(trtmc_text_to_token_features_api_v1)},
    run<internal::ITextToTokenFeatures, trtmc_text_to_token_features_request_v1,
        TokenFeaturesStorage>,
    result_view<TokenFeaturesStorage, trtmc_token_features_view_v1>};
static_assert(offsetof(trtmc_text_to_token_features_api_v1, header) == 0);

const trtmc_text_pair_to_token_features_api_v1 text_pair_to_token_features_api = {
    {1, 0, sizeof(trtmc_text_pair_to_token_features_api_v1)},
    run<internal::ITextPairToTokenFeatures, trtmc_text_pair_to_token_features_request_v1,
        TokenFeaturesStorage>,
    result_view<TokenFeaturesStorage, trtmc_token_features_view_v1>};
static_assert(offsetof(trtmc_text_pair_to_token_features_api_v1, header) == 0);

const trtmc_text_to_pooled_features_api_v1 text_to_pooled_features_api = {
    {1, 0, sizeof(trtmc_text_to_pooled_features_api_v1)},
    run<internal::ITextToPooledFeatures, trtmc_text_to_pooled_features_request_v1,
        PooledFeaturesStorage>,
    result_view<PooledFeaturesStorage, trtmc_pooled_features_view_v1>};
static_assert(offsetof(trtmc_text_to_pooled_features_api_v1, header) == 0);
const trtmc_text_to_head_scores_api_v1 text_to_head_scores_api = {
    {1, 0, sizeof(trtmc_text_to_head_scores_api_v1)},
    run<internal::ITextToHeadScores, trtmc_text_to_head_scores_request_v1, HeadScoresStorage>,
    result_view<HeadScoresStorage, trtmc_head_scores_view_v1>};
static_assert(offsetof(trtmc_text_to_head_scores_api_v1, header) == 0);

const trtmc_text_to_embedding_api_v1 text_to_embedding_api = {
    {1, 0, sizeof(trtmc_text_to_embedding_api_v1)},
    run<internal::ITextToEmbedding, trtmc_text_to_embedding_request_v1, SemanticEmbeddingStorage>,
    result_view<SemanticEmbeddingStorage, trtmc_semantic_embedding_view_v1>};
static_assert(offsetof(trtmc_text_to_embedding_api_v1, header) == 0);

const trtmc_title_body_to_embedding_api_v1 title_body_to_embedding_api = {
    {1, 0, sizeof(trtmc_title_body_to_embedding_api_v1)},
    run<internal::ITitleBodyToEmbedding, trtmc_title_body_to_embedding_request_v1,
        SemanticEmbeddingStorage>,
    result_view<SemanticEmbeddingStorage, trtmc_semantic_embedding_view_v1>};
static_assert(offsetof(trtmc_title_body_to_embedding_api_v1, header) == 0);

const trtmc_masked_text_to_token_scores_api_v1 masked_text_to_token_scores_api = {
    {1, 0, sizeof(trtmc_masked_text_to_token_scores_api_v1)},
    run<internal::IMaskedTextToTokenScores, trtmc_masked_text_to_token_scores_request_v1,
        VocabularyScoresStorage>,
    result_view<VocabularyScoresStorage, trtmc_vocabulary_scores_view_v1>};
static_assert(offsetof(trtmc_masked_text_to_token_scores_api_v1, header) == 0);

const trtmc_text_pair_to_pretraining_relation_scores_api_v1
    text_pair_to_pretraining_relation_scores_api = {
        {1, 0, sizeof(trtmc_text_pair_to_pretraining_relation_scores_api_v1)},
        run<internal::ITextPairToPretrainingRelationScores,
            trtmc_text_pair_to_pretraining_relation_scores_request_v1, LabelScoresStorage>,
        label_scores_result_view};
static_assert(offsetof(trtmc_text_pair_to_pretraining_relation_scores_api_v1, header) == 0);

const trtmc_text_to_replaced_token_scores_api_v1 text_to_replaced_token_scores_api = {
    {1, 0, sizeof(trtmc_text_to_replaced_token_scores_api_v1)},
    run<internal::ITextToReplacedTokenScores, trtmc_text_to_replaced_token_scores_request_v1,
        ReplacedTokenScoresStorage>,
    result_view<ReplacedTokenScoresStorage, trtmc_replaced_token_scores_view_v1>};
static_assert(offsetof(trtmc_text_to_replaced_token_scores_api_v1, header) == 0);

const trtmc_text_prediction_positions_to_token_scores_api_v1
    text_prediction_positions_to_token_scores_api = {
        {1, 0, sizeof(trtmc_text_prediction_positions_to_token_scores_api_v1)},
        run<internal::ITextPredictionPositionsToTokenScores,
            trtmc_text_prediction_positions_to_token_scores_request_v1, VocabularyScoresStorage>,
        result_view<VocabularyScoresStorage, trtmc_vocabulary_scores_view_v1>};
static_assert(offsetof(trtmc_text_prediction_positions_to_token_scores_api_v1, header) == 0);

const trtmc_image_to_token_features_api_v1 image_to_token_features_api = {
    {1, 0, sizeof(trtmc_image_to_token_features_api_v1)},
    run<internal::IImageToTokenFeatures, trtmc_image_to_token_features_request_v1,
        ImageTokenFeaturesStorage>,
    result_view<ImageTokenFeaturesStorage, trtmc_image_token_features_view_v1>};
static_assert(offsetof(trtmc_image_to_token_features_api_v1, header) == 0);

const trtmc_image_to_spatial_features_api_v1 image_to_spatial_features_api = {
    {1, 0, sizeof(trtmc_image_to_spatial_features_api_v1)},
    run<internal::IImageToSpatialFeatures, trtmc_image_to_spatial_features_request_v1,
        SpatialFeaturesStorage>,
    result_view<SpatialFeaturesStorage, trtmc_spatial_features_view_v1>};
static_assert(offsetof(trtmc_image_to_spatial_features_api_v1, header) == 0);

const trtmc_image_to_pooled_features_api_v1 image_to_pooled_features_api = {
    {1, 0, sizeof(trtmc_image_to_pooled_features_api_v1)},
    run<internal::IImageToPooledFeatures, trtmc_image_to_pooled_features_request_v1,
        PooledFeaturesStorage>,
    result_view<PooledFeaturesStorage, trtmc_pooled_features_view_v1>};
static_assert(offsetof(trtmc_image_to_pooled_features_api_v1, header) == 0);

const trtmc_image_to_embedding_api_v1 image_to_embedding_api = {
    {1, 0, sizeof(trtmc_image_to_embedding_api_v1)},
    run<internal::IImageToEmbedding, trtmc_image_to_embedding_request_v1, SemanticEmbeddingStorage>,
    result_view<SemanticEmbeddingStorage, trtmc_semantic_embedding_view_v1>};
static_assert(offsetof(trtmc_image_to_embedding_api_v1, header) == 0);

const trtmc_image_text_to_embedding_api_v1 image_text_to_embedding_api = {
    {1, 0, sizeof(trtmc_image_text_to_embedding_api_v1)},
    run<internal::IImageTextToEmbedding, trtmc_image_text_to_embedding_request_v1,
        SemanticEmbeddingStorage>,
    result_view<SemanticEmbeddingStorage, trtmc_semantic_embedding_view_v1>};
static_assert(offsetof(trtmc_image_text_to_embedding_api_v1, header) == 0);

const trtmc_text_pair_to_relevance_api_v1 text_pair_to_relevance_api = {
    {1, 0, sizeof(trtmc_text_pair_to_relevance_api_v1)},
    run<internal::ITextPairToRelevance, trtmc_text_pair_to_relevance_request_v1, RelevanceStorage>,
    result_view<RelevanceStorage, trtmc_relevance_view_v1>};
static_assert(offsetof(trtmc_text_pair_to_relevance_api_v1, header) == 0);

const trtmc_text_image_to_relevance_api_v1 text_image_to_relevance_api = {
    {1, 0, sizeof(trtmc_text_image_to_relevance_api_v1)},
    run<internal::ITextImageToRelevance, trtmc_text_image_to_relevance_request_v1,
        RelevanceStorage>,
    result_view<RelevanceStorage, trtmc_relevance_view_v1>};
static_assert(offsetof(trtmc_text_image_to_relevance_api_v1, header) == 0);

const trtmc_text_image_text_to_relevance_api_v1 text_image_text_to_relevance_api = {
    {1, 0, sizeof(trtmc_text_image_text_to_relevance_api_v1)},
    run<internal::ITextImageTextToRelevance, trtmc_text_image_text_to_relevance_request_v1,
        RelevanceStorage>,
    result_view<RelevanceStorage, trtmc_relevance_view_v1>};
static_assert(offsetof(trtmc_text_image_text_to_relevance_api_v1, header) == 0);

const trtmc_image_to_class_scores_api_v1 image_to_class_scores_api = {
    {1, 0, sizeof(trtmc_image_to_class_scores_api_v1)},
    run<internal::IImageToClassScores, trtmc_image_to_class_scores_request_v1, LabelScoresStorage>,
    label_scores_result_view};
static_assert(offsetof(trtmc_image_to_class_scores_api_v1, header) == 0);

trtmc_status TRTMC_CALL query_documents_run(
    trtmc_model* model, const trtmc_text_query_documents_to_relevance_request_v1* input,
    const trtmc_config_view_v1* config, trtmc_result** out, trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(input != nullptr && out != nullptr, "document relevance input or output is null");
        const auto source = checked_span(input->documents.data, input->documents.size);
        std::vector<std::string_view> documents;
        documents.reserve(source.size());
        for (const auto document : source)
            documents.push_back(string_view(document));
        const internal::TextQueryDocumentsToRelevanceRequest request{
            string_view(input->query), {documents.data(), documents.size()}};
        const ConvertedConfig options(config);

        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<internal::ITextQueryDocumentsToRelevance>(
            model, internal::ITextQueryDocumentsToRelevance::kTask);
        validate_task_config(model_owner(model),
                             internal::contract_key<internal::ITextQueryDocumentsToRelevance>(),
                             options.view());
        auto result = family.run(request, options.view());
        result_require(result.scores.size() == documents.size(),
                       "document relevance result count differs from document count");
        *out = make_result<DocumentRelevanceStorage>(std::move(result));
    });
}

const trtmc_text_query_documents_to_relevance_api_v1 query_documents_api = {
    {1, 0, sizeof(trtmc_text_query_documents_to_relevance_api_v1)},
    query_documents_run,
    result_view<DocumentRelevanceStorage, trtmc_document_relevance_view_v1>};
static_assert(offsetof(trtmc_text_query_documents_to_relevance_api_v1, header) == 0);

const trtmc_image_to_token_and_pooled_features_api_v1 image_to_token_and_pooled_features_api = {
    {1, 0, sizeof(trtmc_image_to_token_and_pooled_features_api_v1)},
    run<internal::IImageToTokenAndPooledFeatures,
        trtmc_image_to_token_and_pooled_features_request_v1, ImageTokenAndPooledFeaturesStorage>,
    result_view<ImageTokenAndPooledFeaturesStorage, trtmc_image_token_and_pooled_features_view_v1>};
static_assert(offsetof(trtmc_image_to_token_and_pooled_features_api_v1, header) == 0);

// This loop transports/owns typed items; it never executes scalar model Tasks.
template <class Function>
auto feature_item(size_t index, Function function) {
    try {
        return function();
    } catch (const internal::ConfigError& failure) {
        throw OwnedApiFailure{TRTMC_INVALID_CONFIG,
                              "batch item[" + std::to_string(index) + "]: " + failure.what()};
    } catch (const ApiFailure& failure) {
        throw OwnedApiFailure{failure.status,
                              "batch item[" + std::to_string(index) + "]: " + failure.message};
    } catch (const OwnedApiFailure& failure) {
        throw OwnedApiFailure{failure.status,
                              "batch item[" + std::to_string(index) + "]: " + failure.message};
    }
    // Allocation failure propagates directly to guarded; do not allocate an
    // alternate diagnostic in an attempt to recover from out-of-memory.
}
template <class Storage, class View>
void fill_feature_view(const Storage& value, View* out) {
    *out = value.view;
}
void fill_feature_view(const LabelScoresStorage& value, trtmc_label_scores_view_v1* out) {
    fill_label_scores_view(value, out);
}
template <class Storage>
struct FeatureBatchStorage final : ResultStorage {
    template <class Results>
    explicit FeatureBatchStorage(Results values) {
        items.reserve(values.size());
        for (size_t index = 0; index < values.size(); ++index)
            items.push_back(feature_item(
                index, [&] { return std::make_unique<Storage>(std::move(values[index])); }));
    }
    std::vector<std::unique_ptr<Storage>> items;
};
template <class Storage>
trtmc_status TRTMC_CALL feature_batch_count(const trtmc_result* result, uint64_t* out,
                                            trtmc_error** error) noexcept {
    if (out)
        *out = 0;
    return guarded(error, [&] {
        require(out != nullptr, "batch result count output is null");
        *out = require_result<FeatureBatchStorage<Storage>>(result).items.size();
    });
}
template <class Storage, class View>
trtmc_status TRTMC_CALL feature_batch_item_view(const trtmc_result* result, uint64_t index,
                                                View* out, trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out != nullptr, "batch item view output is null");
        const auto& storage = require_result<FeatureBatchStorage<Storage>>(result);
        require(index < storage.items.size(), "batch result item index is out of range");
        fill_feature_view(*storage.items[index], out);
    });
}
template <class Interface, class WireRequest, class Storage>
trtmc_status TRTMC_CALL run_feature_batch(trtmc_model* model, const WireRequest* input,
                                          trtmc_result** out, trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(input && out, "feature batch request and output are required");
        const auto source = checked_span(input->items, input->count);
        require(!source.empty(), "feature batch requires at least one item");
        using Request = typename Interface::Request;
        std::vector<typename Request::Item> items;
        std::vector<ConvertedConfig> configs;
        items.reserve(source.size());
        configs.reserve(source.size());
        for (size_t index = 0; index < source.size(); ++index) {
            feature_item(index, [&] {
                configs.emplace_back(&source[index].config);

                items.push_back({convert(source[index].input), configs.back().view()});
            });
        }
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<Interface>(model, Interface::kTask);
        validate_batch_configs(model_owner(model), internal::contract_key<Interface>(), configs);
        auto results = family.run_batch(Request{{items.data(), items.size()}});
        result_require(results.size() == items.size(),
                       "family batch result count differs from input count");
        if constexpr (std::is_same_v<Interface, internal::IBatchImageToClassScores>)
            for (size_t index = 0; index < results.size(); ++index)
                feature_item(index, [&] { require_class_identity(results[index]); });
        *out = make_result<FeatureBatchStorage<Storage>>(std::move(results));
    });
}
const trtmc_batch_image_to_class_scores_api_v1 batch_image_to_class_scores_api = {
    {1, 0, sizeof(trtmc_batch_image_to_class_scores_api_v1)},
    run_feature_batch<internal::IBatchImageToClassScores,
                      trtmc_batch_image_to_class_scores_request_v1, LabelScoresStorage>,
    feature_batch_count<LabelScoresStorage>,
    feature_batch_item_view<LabelScoresStorage, trtmc_label_scores_view_v1>};

const trtmc_batch_image_to_token_features_api_v1 batch_image_to_token_features_api = {
    {1, 0, sizeof(trtmc_batch_image_to_token_features_api_v1)},
    run_feature_batch<internal::IBatchImageToTokenFeatures,
                      trtmc_batch_image_to_token_features_request_v1, ImageTokenFeaturesStorage>,
    feature_batch_count<ImageTokenFeaturesStorage>,
    feature_batch_item_view<ImageTokenFeaturesStorage, trtmc_image_token_features_view_v1>};
static_assert(offsetof(trtmc_batch_image_to_token_features_api_v1, header) == 0);
const trtmc_batch_image_to_spatial_features_api_v1 batch_image_to_spatial_features_api = {
    {1, 0, sizeof(trtmc_batch_image_to_spatial_features_api_v1)},
    run_feature_batch<internal::IBatchImageToSpatialFeatures,
                      trtmc_batch_image_to_spatial_features_request_v1, SpatialFeaturesStorage>,
    feature_batch_count<SpatialFeaturesStorage>,
    feature_batch_item_view<SpatialFeaturesStorage, trtmc_spatial_features_view_v1>};
static_assert(offsetof(trtmc_batch_image_to_spatial_features_api_v1, header) == 0);
const trtmc_batch_image_to_pooled_features_api_v1 batch_image_to_pooled_features_api = {
    {1, 0, sizeof(trtmc_batch_image_to_pooled_features_api_v1)},
    run_feature_batch<internal::IBatchImageToPooledFeatures,
                      trtmc_batch_image_to_pooled_features_request_v1, PooledFeaturesStorage>,
    feature_batch_count<PooledFeaturesStorage>,
    feature_batch_item_view<PooledFeaturesStorage, trtmc_pooled_features_view_v1>};
static_assert(offsetof(trtmc_batch_image_to_pooled_features_api_v1, header) == 0);
const trtmc_batch_text_to_embedding_api_v1 batch_text_to_embedding_api = {
    {1, 0, sizeof(trtmc_batch_text_to_embedding_api_v1)},
    run_feature_batch<internal::IBatchTextToEmbedding, trtmc_batch_text_to_embedding_request_v1,
                      SemanticEmbeddingStorage>,
    feature_batch_count<SemanticEmbeddingStorage>,
    feature_batch_item_view<SemanticEmbeddingStorage, trtmc_semantic_embedding_view_v1>};
static_assert(offsetof(trtmc_batch_text_to_embedding_api_v1, header) == 0);
const trtmc_batch_text_to_token_features_api_v1 batch_text_to_token_features_api = {
    {1, 0, sizeof(trtmc_batch_text_to_token_features_api_v1)},
    run_feature_batch<internal::IBatchTextToTokenFeatures,
                      trtmc_batch_text_to_token_features_request_v1, TokenFeaturesStorage>,
    feature_batch_count<TokenFeaturesStorage>,
    feature_batch_item_view<TokenFeaturesStorage, trtmc_token_features_view_v1>};
static_assert(offsetof(trtmc_batch_text_to_token_features_api_v1, header) == 0);
const trtmc_batch_image_to_token_and_pooled_features_api_v1
    batch_image_to_token_and_pooled_features_api = {
        {1, 0, sizeof(trtmc_batch_image_to_token_and_pooled_features_api_v1)},
        run_feature_batch<internal::IBatchImageToTokenAndPooledFeatures,
                          trtmc_batch_image_to_token_and_pooled_features_request_v1,
                          ImageTokenAndPooledFeaturesStorage>,
        feature_batch_count<ImageTokenAndPooledFeaturesStorage>,
        feature_batch_item_view<ImageTokenAndPooledFeaturesStorage,
                                trtmc_image_token_and_pooled_features_view_v1>};
static_assert(offsetof(trtmc_batch_image_to_token_and_pooled_features_api_v1, header) == 0);

const TaskBinding bindings[] = {
    {internal::IBatchImageToTokenFeatures::kTask, 1, 0, &batch_image_to_token_features_api.header},
    {internal::IBatchImageToSpatialFeatures::kTask, 1, 0,
     &batch_image_to_spatial_features_api.header},
    {internal::IBatchImageToPooledFeatures::kTask, 1, 0,
     &batch_image_to_pooled_features_api.header},
    {internal::IBatchTextToEmbedding::kTask, 1, 0, &batch_text_to_embedding_api.header},
    {internal::IBatchTextToTokenFeatures::kTask, 1, 0, &batch_text_to_token_features_api.header},
    {internal::IBatchImageToTokenAndPooledFeatures::kTask, 1, 0,
     &batch_image_to_token_and_pooled_features_api.header},
    {internal::IBatchImageToClassScores::kTask, 1, 0, &batch_image_to_class_scores_api.header},
    {internal::IImageToTokenAndPooledFeatures::kTask, 1, 0,
     &image_to_token_and_pooled_features_api.header},
    {internal::ITextQueryDocumentsToRelevance::kTask, 1, 0, &query_documents_api.header},
    {internal::ITextToTokenFeatures::kTask, 1, 0, &text_to_token_features_api.header},
    {internal::ITextPairToTokenFeatures::kTask, 1, 0, &text_pair_to_token_features_api.header},
    {internal::ITextToPooledFeatures::kTask, 1, 0, &text_to_pooled_features_api.header},
    {internal::ITextToHeadScores::kTask, 1, 0, &text_to_head_scores_api.header},
    {internal::ITextToEmbedding::kTask, 1, 0, &text_to_embedding_api.header},
    {internal::ITitleBodyToEmbedding::kTask, 1, 0, &title_body_to_embedding_api.header},
    {internal::IMaskedTextToTokenScores::kTask, 1, 0, &masked_text_to_token_scores_api.header},
    {internal::ITextPairToPretrainingRelationScores::kTask, 1, 0,
     &text_pair_to_pretraining_relation_scores_api.header},
    {internal::ITextToReplacedTokenScores::kTask, 1, 0, &text_to_replaced_token_scores_api.header},
    {internal::ITextPredictionPositionsToTokenScores::kTask, 1, 0,
     &text_prediction_positions_to_token_scores_api.header},
    {internal::IImageToTokenFeatures::kTask, 1, 0, &image_to_token_features_api.header},
    {internal::IImageToSpatialFeatures::kTask, 1, 0, &image_to_spatial_features_api.header},
    {internal::IImageToPooledFeatures::kTask, 1, 0, &image_to_pooled_features_api.header},
    {internal::IImageToEmbedding::kTask, 1, 0, &image_to_embedding_api.header},
    {internal::IImageTextToEmbedding::kTask, 1, 0, &image_text_to_embedding_api.header},
    {internal::ITextPairToRelevance::kTask, 1, 0, &text_pair_to_relevance_api.header},
    {internal::ITextImageToRelevance::kTask, 1, 0, &text_image_to_relevance_api.header},
    {internal::ITextImageTextToRelevance::kTask, 1, 0, &text_image_text_to_relevance_api.header},
    {internal::IImageToClassScores::kTask, 1, 0, &image_to_class_scores_api.header},
};

} // namespace

Span<const TaskBinding> features_task_bindings() noexcept {
    return bindings;
}

} // namespace trtmc::api

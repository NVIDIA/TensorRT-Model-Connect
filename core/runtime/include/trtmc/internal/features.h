/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/internal/image.h"
#include "trtmc/internal/matrix.h"
#include "trtmc/internal/scores.h"
#include "trtmc/internal/text.h"

namespace trtmc::internal {

inline constexpr int32_t kFeaturePaddingInputIndex = -2;
// Mapping is in the family-tokenized combined sequence. input_index is 0 for
// single/first input, 1 for second input, -1 for inserted special tokens, and
// kFeaturePaddingInputIndex for padding. Padding retains its token ID, sequence
// index and raw feature values, but has no original-input byte offsets.
// Byte offsets, when present, refer to original UTF-8 input, end-exclusive.
struct FeatureToken {
    int64_t token_id{0};
    int32_t input_index{0};
    uint64_t token_index{0};
    bool has_byte_offsets{false};
    uint64_t byte_begin{0};
    uint64_t byte_end{0};
};
struct TokenFeaturesResult {
    // [returned sequence row, hidden feature]. Family may return valid-only or
    // full-profile rows; any included padding is explicitly mapped in tokens.
    FloatMatrix features;
    std::vector<FeatureToken> tokens;
};
struct PooledFeaturesResult {
    std::vector<float> values;
    std::string pooling;       // e.g. "cls", "mean", "first_token"; family semantics.
    std::string normalization; // e.g. "none" or "l2".
};
struct SemanticEmbeddingResult {
    std::vector<float> values;
    // Identifies the checkpoint space when known, not a content hash. Empty
    // means unknown; it never establishes compatibility with another model.
    std::string embedding_space;
    std::string pooling;
    std::string normalization;
};
enum class EmbeddingRole : uint32_t { Default = 0, Query = 1, Document = 2 };
struct VocabularyScoresResult {
    FloatMatrix logits; // [selected position, vocabulary ID]; no softmax.
    std::vector<FeatureToken> positions;
    std::string vocabulary_id;
};
struct ReplacedTokenScoresResult {
    std::vector<float> logits; // Positive logits favor "replaced", not "original".
    std::vector<FeatureToken> tokens;
};
enum class ImageFeatureTokenRole : uint32_t { Patch = 1, Class = 2, Register = 3 };
struct ImageFeatureToken {
    ImageFeatureTokenRole role{ImageFeatureTokenRole::Patch};
    uint64_t grid_row{0};
    uint64_t grid_column{0};
    // For patch tokens only: footprint in normalized original-input coordinates.
    float x_min{0}, y_min{0}, x_max{0}, y_max{0};
};
struct ImageTokenFeaturesResult {
    FloatMatrix features; // [token, feature], retaining class/register rows.
    std::vector<ImageFeatureToken> tokens;
    uint64_t grid_rows{0}, grid_columns{0};
};
struct ImageTokenAndPooledFeaturesResult {
    ImageTokenFeaturesResult tokens;
    PooledFeaturesResult pooled;
};
struct SpatialFeatureMap {
    std::string name;
    std::vector<float> values; // Contiguous CHW.
    uint64_t channels{0}, height{0}, width{0};
    // Strides are in processed-image pixels, not original-image coordinates.
    double stride_y{0}, stride_x{0};
};
struct FeatureImageTransform {
    // Edge coordinates: image corner is (0,0), first pixel center is (0.5,0.5).
    // processed_x = scale_x * source_x + offset_x, likewise for y.
    double scale_x{1}, scale_y{1}, offset_x{0}, offset_y{0};
};
struct SpatialFeaturesResult {
    std::vector<SpatialFeatureMap> maps;
    uint64_t processed_image_height{0}, processed_image_width{0};
    FeatureImageTransform source_to_processed;
};
struct RelevanceResult {
    float score{0};
    ScoreKind kind{ScoreKind::Unbounded};
};
struct DocumentRelevanceResult {
    std::vector<float> scores; // One per input document, retaining its order.
    ScoreKind kind{ScoreKind::Unbounded};
};
struct TextQueryDocumentsToRelevanceRequest {
    std::string_view query;
    Span<const std::string_view> documents;
};

class ITextQueryDocumentsToRelevance {
  public:
    using TaskInterface = ITextQueryDocumentsToRelevance;
    static constexpr std::string_view kTask = "text_query_documents_to_relevance";
    virtual ~ITextQueryDocumentsToRelevance() = default;
    // One family-owned list-scoring operation with a shared query. Native
    // batching, profile-sized chunks, or serial orchestration belong to the
    // family; the shared runtime never synthesizes a loop over scalar rerank.
    // Empty documents produce an empty result. Results preserve input order.
    // This functional contract makes no provider-wide native-batching claim.
    virtual DocumentRelevanceResult run(const TextQueryDocumentsToRelevanceRequest&,
                                        ConfigView) = 0;
};

struct TextToTokenFeaturesRequest {
    TextSource text;
};
struct FeatureTextPair {
    std::string_view first;
    std::string_view second;
};
struct FeatureTokenizedPair {
    Span<const int32_t> token_ids;
    Span<const int32_t> segment_ids;    // Required per-token model segment IDs.
    Span<const uint8_t> attention_mask; // Empty means every supplied token is valid.
};
struct TextPairToTokenFeaturesRequest {
    std::variant<FeatureTextPair, FeatureTokenizedPair> pair;
};
struct TextToPooledFeaturesRequest {
    TextSource text;
};
struct TextToEmbeddingRequest {
    std::string_view text;
    EmbeddingRole role{EmbeddingRole::Default};
};
struct TitleBodyToEmbeddingRequest {
    std::string_view title;
    std::string_view body;
};
struct MaskedTextToTokenScoresRequest {
    TextSource text;
};
struct TextPairToPretrainingRelationScoresRequest {
    std::string_view first;
    std::string_view second;
};
struct TextToReplacedTokenScoresRequest {
    TextSource text;
};
struct TextPredictionPositionsToTokenScoresRequest {
    Span<const int64_t> token_ids;
    Span<const uint8_t> attention_mask;
    Span<const int64_t> segment_ids;
    Span<const uint8_t> blocked_attention;
    Span<const uint64_t> prediction_positions;
};
struct ImageToTokenFeaturesRequest {
    ImageView image;
};
struct ImageToTokenAndPooledFeaturesRequest {
    ImageView image;
};
struct ImageToSpatialFeaturesRequest {
    ImageView image;
};
struct ImageToPooledFeaturesRequest {
    ImageView image;
};
struct ImageToEmbeddingRequest {
    ImageView image;
};
struct ImageTextToEmbeddingRequest {
    ImageView image;
    std::string_view text;
};
struct TextPairToRelevanceRequest {
    std::string_view query;
    std::string_view document;
};
struct TextImageToRelevanceRequest {
    std::string_view query;
    ImageView document;
};
struct TextImageTextToRelevanceRequest {
    std::string_view query;
    ImageView image;
    std::string_view document_text;
};
struct ImageToClassScoresRequest {
    ImageView image;
};

// Each request type is distinct so one family can implement multiple run overloads.
class ITextToTokenFeatures {
  public:
    using TaskInterface = ITextToTokenFeatures;
    static constexpr std::string_view kTask = "text_to_token_features";
    virtual ~ITextToTokenFeatures() = default;
    virtual TokenFeaturesResult run(const TextToTokenFeaturesRequest&, ConfigView) = 0;
};

class ITextPairToTokenFeatures {
  public:
    using TaskInterface = ITextPairToTokenFeatures;
    static constexpr std::string_view kTask = "text_pair_to_token_features";
    virtual ~ITextPairToTokenFeatures() = default;
    virtual TokenFeaturesResult run(const TextPairToTokenFeaturesRequest&, ConfigView) = 0;
};

class ITextToPooledFeatures {
  public:
    using TaskInterface = ITextToPooledFeatures;
    static constexpr std::string_view kTask = "text_to_pooled_features";
    virtual ~ITextToPooledFeatures() = default;
    virtual PooledFeaturesResult run(const TextToPooledFeaturesRequest&, ConfigView) = 0;
};

class ITextToEmbedding {
  public:
    using TaskInterface = ITextToEmbedding;
    static constexpr std::string_view kTask = "text_to_embedding";
    virtual ~ITextToEmbedding() = default;
    virtual SemanticEmbeddingResult run(const TextToEmbeddingRequest&, ConfigView) = 0;
};

class ITitleBodyToEmbedding {
  public:
    using TaskInterface = ITitleBodyToEmbedding;
    static constexpr std::string_view kTask = "title_body_to_embedding";
    virtual ~ITitleBodyToEmbedding() = default;
    virtual SemanticEmbeddingResult run(const TitleBodyToEmbeddingRequest&, ConfigView) = 0;
};

class IMaskedTextToTokenScores {
  public:
    using TaskInterface = IMaskedTextToTokenScores;
    static constexpr std::string_view kTask = "masked_text_to_token_scores";
    virtual ~IMaskedTextToTokenScores() = default;
    virtual VocabularyScoresResult run(const MaskedTextToTokenScoresRequest&, ConfigView) = 0;
};

class ITextPairToPretrainingRelationScores {
  public:
    using TaskInterface = ITextPairToPretrainingRelationScores;
    static constexpr std::string_view kTask = "text_pair_to_pretraining_relation_scores";
    virtual ~ITextPairToPretrainingRelationScores() = default;
    virtual LabelScoresResult run(const TextPairToPretrainingRelationScoresRequest&,
                                  ConfigView) = 0;
};

class ITextToReplacedTokenScores {
  public:
    using TaskInterface = ITextToReplacedTokenScores;
    static constexpr std::string_view kTask = "text_to_replaced_token_scores";
    virtual ~ITextToReplacedTokenScores() = default;
    virtual ReplacedTokenScoresResult run(const TextToReplacedTokenScoresRequest&, ConfigView) = 0;
};

class ITextPredictionPositionsToTokenScores {
  public:
    using TaskInterface = ITextPredictionPositionsToTokenScores;
    static constexpr std::string_view kTask = "text_prediction_positions_to_token_scores";
    virtual ~ITextPredictionPositionsToTokenScores() = default;
    virtual VocabularyScoresResult run(const TextPredictionPositionsToTokenScoresRequest&,
                                       ConfigView) = 0;
};

class IImageToTokenFeatures {
  public:
    using TaskInterface = IImageToTokenFeatures;
    static constexpr std::string_view kTask = "image_to_token_features";
    virtual ~IImageToTokenFeatures() = default;
    virtual ImageTokenFeaturesResult run(const ImageToTokenFeaturesRequest&, ConfigView) = 0;
};

class IImageToSpatialFeatures {
  public:
    using TaskInterface = IImageToSpatialFeatures;
    static constexpr std::string_view kTask = "image_to_spatial_features";
    virtual ~IImageToSpatialFeatures() = default;
    virtual SpatialFeaturesResult run(const ImageToSpatialFeaturesRequest&, ConfigView) = 0;
};

class IImageToPooledFeatures {
  public:
    using TaskInterface = IImageToPooledFeatures;
    static constexpr std::string_view kTask = "image_to_pooled_features";
    virtual ~IImageToPooledFeatures() = default;
    virtual PooledFeaturesResult run(const ImageToPooledFeaturesRequest&, ConfigView) = 0;
};

class IImageToEmbedding {
  public:
    using TaskInterface = IImageToEmbedding;
    static constexpr std::string_view kTask = "image_to_embedding";
    virtual ~IImageToEmbedding() = default;
    virtual SemanticEmbeddingResult run(const ImageToEmbeddingRequest&, ConfigView) = 0;
};

class IImageTextToEmbedding {
  public:
    using TaskInterface = IImageTextToEmbedding;
    static constexpr std::string_view kTask = "image_text_to_embedding";
    virtual ~IImageTextToEmbedding() = default;
    virtual SemanticEmbeddingResult run(const ImageTextToEmbeddingRequest&, ConfigView) = 0;
};

class ITextPairToRelevance {
  public:
    using TaskInterface = ITextPairToRelevance;
    static constexpr std::string_view kTask = "text_pair_to_relevance";
    virtual ~ITextPairToRelevance() = default;
    virtual RelevanceResult run(const TextPairToRelevanceRequest&, ConfigView) = 0;
};

class ITextImageToRelevance {
  public:
    using TaskInterface = ITextImageToRelevance;
    static constexpr std::string_view kTask = "text_image_to_relevance";
    virtual ~ITextImageToRelevance() = default;
    virtual RelevanceResult run(const TextImageToRelevanceRequest&, ConfigView) = 0;
};

class ITextImageTextToRelevance {
  public:
    using TaskInterface = ITextImageTextToRelevance;
    static constexpr std::string_view kTask = "text_image_text_to_relevance";
    virtual ~ITextImageTextToRelevance() = default;
    virtual RelevanceResult run(const TextImageTextToRelevanceRequest&, ConfigView) = 0;
};

class IImageToTokenAndPooledFeatures {
  public:
    using TaskInterface = IImageToTokenAndPooledFeatures;
    static constexpr std::string_view kTask = "image_to_token_and_pooled_features";
    virtual ~IImageToTokenAndPooledFeatures() = default;
    // One family-owned extraction returns both views of the same evaluation.
    // Shared code must not synthesize two independent inference calls.
    virtual ImageTokenAndPooledFeaturesResult run(const ImageToTokenAndPooledFeaturesRequest&,
                                                  ConfigView) = 0;
};

class IImageToClassScores {
  public:
    using TaskInterface = IImageToClassScores;
    static constexpr std::string_view kTask = "image_to_class_scores";
    virtual ~IImageToClassScores() = default;
    // Nonempty scores in vocabulary-ordinal order. Supply either a nonempty
    // label for each class or an explicit vocabulary_id; never anonymous scores.
    // ScoreKind declares their interpretation without shared normalization.
    virtual LabelScoresResult run(const ImageToClassScoresRequest&, ConfigView) = 0;
};

// Native batch is a separate execution contract. Validate every item before
// inference. Return one result per item in the original order; shared code
// must not implement this by invoking scalar interfaces.
// Failure returns no partial batch result, but does not roll back GPU/RNG/cache
// effects already performed by the family. Identify the item in errors when known.
struct BatchImageToClassScoresItem {
    ImageToClassScoresRequest input;
    ConfigView config;
};
struct BatchImageToClassScoresRequest {
    using Item = BatchImageToClassScoresItem;
    Span<const Item> items;
};
class IBatchImageToClassScores {
  public:
    using TaskInterface = IBatchImageToClassScores;
    static constexpr std::string_view kTask = "batch_image_to_class_scores";
    using Request = BatchImageToClassScoresRequest;
    virtual ~IBatchImageToClassScores() = default;
    // Every item obeys the same class-identity contract as IImageToClassScores.
    virtual std::vector<LabelScoresResult> run_batch(const BatchImageToClassScoresRequest&) = 0;
};

struct BatchImageToTokenFeaturesItem {
    ImageToTokenFeaturesRequest input;
    ConfigView config;
};
struct BatchImageToTokenFeaturesRequest {
    using Item = BatchImageToTokenFeaturesItem;
    Span<const Item> items;
};
class IBatchImageToTokenFeatures {
  public:
    using TaskInterface = IBatchImageToTokenFeatures;
    static constexpr std::string_view kTask = "batch_image_to_token_features";
    using Request = BatchImageToTokenFeaturesRequest;
    virtual ~IBatchImageToTokenFeatures() = default;
    virtual std::vector<ImageTokenFeaturesResult>
    run_batch(const BatchImageToTokenFeaturesRequest&) = 0;
};
struct BatchImageToSpatialFeaturesItem {
    ImageToSpatialFeaturesRequest input;
    ConfigView config;
};
struct BatchImageToSpatialFeaturesRequest {
    using Item = BatchImageToSpatialFeaturesItem;
    Span<const Item> items;
};
class IBatchImageToSpatialFeatures {
  public:
    using TaskInterface = IBatchImageToSpatialFeatures;
    static constexpr std::string_view kTask = "batch_image_to_spatial_features";
    using Request = BatchImageToSpatialFeaturesRequest;
    virtual ~IBatchImageToSpatialFeatures() = default;
    virtual std::vector<SpatialFeaturesResult>
    run_batch(const BatchImageToSpatialFeaturesRequest&) = 0;
};
struct BatchImageToPooledFeaturesItem {
    ImageToPooledFeaturesRequest input;
    ConfigView config;
};
struct BatchImageToPooledFeaturesRequest {
    using Item = BatchImageToPooledFeaturesItem;
    Span<const Item> items;
};
class IBatchImageToPooledFeatures {
  public:
    using TaskInterface = IBatchImageToPooledFeatures;
    static constexpr std::string_view kTask = "batch_image_to_pooled_features";
    using Request = BatchImageToPooledFeaturesRequest;
    virtual ~IBatchImageToPooledFeatures() = default;
    virtual std::vector<PooledFeaturesResult>
    run_batch(const BatchImageToPooledFeaturesRequest&) = 0;
};
struct BatchTextToEmbeddingItem {
    TextToEmbeddingRequest input;
    ConfigView config;
};
struct BatchTextToEmbeddingRequest {
    using Item = BatchTextToEmbeddingItem;
    Span<const Item> items;
};
class IBatchTextToEmbedding {
  public:
    using TaskInterface = IBatchTextToEmbedding;
    static constexpr std::string_view kTask = "batch_text_to_embedding";
    using Request = BatchTextToEmbeddingRequest;
    virtual ~IBatchTextToEmbedding() = default;
    virtual std::vector<SemanticEmbeddingResult> run_batch(const BatchTextToEmbeddingRequest&) = 0;
};
struct BatchTextToTokenFeaturesItem {
    TextToTokenFeaturesRequest input;
    ConfigView config;
};
struct BatchTextToTokenFeaturesRequest {
    using Item = BatchTextToTokenFeaturesItem;
    Span<const Item> items;
};
class IBatchTextToTokenFeatures {
  public:
    using TaskInterface = IBatchTextToTokenFeatures;
    static constexpr std::string_view kTask = "batch_text_to_token_features";
    using Request = BatchTextToTokenFeaturesRequest;
    virtual ~IBatchTextToTokenFeatures() = default;
    virtual std::vector<TokenFeaturesResult> run_batch(const BatchTextToTokenFeaturesRequest&) = 0;
};
struct BatchImageToTokenAndPooledFeaturesItem {
    ImageToTokenAndPooledFeaturesRequest input;
    ConfigView config;
};
struct BatchImageToTokenAndPooledFeaturesRequest {
    using Item = BatchImageToTokenAndPooledFeaturesItem;
    Span<const Item> items;
};
class IBatchImageToTokenAndPooledFeatures {
  public:
    using TaskInterface = IBatchImageToTokenAndPooledFeatures;
    static constexpr std::string_view kTask = "batch_image_to_token_and_pooled_features";
    using Request = BatchImageToTokenAndPooledFeaturesRequest;
    virtual ~IBatchImageToTokenAndPooledFeatures() = default;
    virtual std::vector<ImageTokenAndPooledFeaturesResult>
    run_batch(const BatchImageToTokenAndPooledFeaturesRequest&) = 0;
};

} // namespace trtmc::internal

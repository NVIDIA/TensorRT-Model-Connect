/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TRTMC_FEATURES_H
#define TRTMC_FEATURES_H

#include "trtmc/image.h"
#include "trtmc/matrix.h"
#include "trtmc/scores.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const uint8_t* data;
    uint64_t size;
} trtmc_feature_u8_view;
typedef struct {
    const uint64_t* data;
    uint64_t size;
} trtmc_feature_u64_view;

enum { TRTMC_FEATURE_INPUT_PADDING = -2, TRTMC_FEATURE_INPUT_SPECIAL = -1 };
/* token_index is in the complete tokenized sequence, including special/padding
 * rows. input_index is 0/1 for the corresponding text, SPECIAL for an inserted
 * special token, or PADDING for padding. Included padding retains token_id,
 * token_index and raw features; has_byte_offsets must be zero for padding.
 * Other byte offsets are optional UTF-8 byte offsets, end-exclusive. */
typedef struct {
    int64_t token_id;
    int32_t input_index;
    uint64_t token_index;
    uint32_t has_byte_offsets;
    uint64_t byte_begin;
    uint64_t byte_end;
} trtmc_feature_token_v1;
typedef struct {
    /* Family-owned valid-only or full-profile rows, never implicitly trimmed. */
    trtmc_f32_matrix_view_v1 features;
    const trtmc_feature_token_v1* tokens;
    uint64_t token_count;
} trtmc_token_features_view_v1;
typedef struct {
    const float* values;
    uint64_t count;
    trtmc_string_view pooling;
    trtmc_string_view normalization;
} trtmc_pooled_features_view_v1;
typedef struct {
    const float* values;
    uint64_t count;
    /* Empty means the checkpoint space is unknown, not a shared space ID.
     * Do not infer cross-model compatibility from two empty identifiers. */
    trtmc_string_view embedding_space;
    trtmc_string_view pooling;
    trtmc_string_view normalization;
} trtmc_semantic_embedding_view_v1;
enum { TRTMC_EMBEDDING_DEFAULT = 0, TRTMC_EMBEDDING_QUERY = 1, TRTMC_EMBEDDING_DOCUMENT = 2 };
typedef struct {
    trtmc_f32_matrix_view_v1 logits; /* [selected position, vocabulary ID]. */
    const trtmc_feature_token_v1* positions;
    uint64_t position_count;
    trtmc_string_view vocabulary_id;
} trtmc_vocabulary_scores_view_v1;
typedef struct {
    const float* logits; /* Positive values favor replaced rather than original. */
    const trtmc_feature_token_v1* tokens;
    uint64_t token_count;
} trtmc_replaced_token_scores_view_v1;
enum { TRTMC_IMAGE_TOKEN_PATCH = 1, TRTMC_IMAGE_TOKEN_CLASS = 2, TRTMC_IMAGE_TOKEN_REGISTER = 3 };
typedef struct {
    uint32_t role;
    uint64_t grid_row;
    uint64_t grid_column;
    float x_min, y_min, x_max, y_max; /* Patch footprint in normalized original image. */
} trtmc_image_feature_token_v1;
typedef struct {
    trtmc_f32_matrix_view_v1 features;
    const trtmc_image_feature_token_v1* tokens;
    uint64_t token_count;
    uint64_t grid_rows;
    uint64_t grid_columns;
} trtmc_image_token_features_view_v1;
typedef struct {
    trtmc_image_token_features_view_v1 tokens;
    trtmc_pooled_features_view_v1 pooled;
} trtmc_image_token_and_pooled_features_view_v1;
typedef struct {
    trtmc_string_view name;
    const float* values; /* Contiguous CHW. */
    uint64_t count;
    uint64_t channels;
    uint64_t height;
    uint64_t width;
    double stride_y; /* Processed-image pixel units. */
    double stride_x;
} trtmc_spatial_feature_map_v1;
typedef struct {
    /* Edge coordinates: corner=(0,0), first pixel center=(0.5,0.5).
     * processed_x = scale_x * source_x + offset_x; likewise for y.
     * This transports resize/crop geometry, not feature receptive fields. */
    double scale_x;
    double scale_y;
    double offset_x;
    double offset_y;
} trtmc_feature_image_transform_v1;
typedef struct {
    const trtmc_spatial_feature_map_v1* maps;
    uint64_t map_count;
    uint64_t processed_image_height;
    uint64_t processed_image_width;
    trtmc_feature_image_transform_v1 source_to_processed;
} trtmc_spatial_features_view_v1;
typedef struct {
    float score;
    uint32_t kind;
} trtmc_relevance_view_v1;
typedef struct {
    const float* scores;
    uint64_t count;
    uint32_t kind;
} trtmc_document_relevance_view_v1;
typedef struct {
    trtmc_string_view query;
    trtmc_strings_view documents;
} trtmc_text_query_documents_to_relevance_request_v1;

/* Scores retain document order, including an empty list. The shared runtime
 * invokes the family once. Native batching, profile chunks or serial execution
 * are family-owned and require separate performance evidence. */
#define TRTMC_TASK_TEXT_QUERY_DOCUMENTS_TO_RELEVANCE "text_query_documents_to_relevance"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_text_query_documents_to_relevance_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_document_relevance_view_v1*,
                                          trtmc_error**);
} trtmc_text_query_documents_to_relevance_api_v1;

typedef struct {
    trtmc_text_source_v1 text;
} trtmc_text_to_token_features_request_v1;
enum { TRTMC_FEATURE_TEXT_PAIR = 1, TRTMC_FEATURE_TOKENIZED_PAIR = 2 };
typedef struct {
    trtmc_string_view first;
    trtmc_string_view second;
} trtmc_feature_text_pair_v1;
typedef struct {
    trtmc_i32_view token_ids;
    /* Required model-specific segment IDs, one per token. The family interprets
     * their meanings; the shared C layer does not drop or remap them. */
    trtmc_i32_view segment_ids;
    /* Empty means every token is valid; otherwise exactly one 0/1 per token. */
    trtmc_feature_u8_view attention_mask;
} trtmc_feature_tokenized_pair_v1;
typedef struct {
    uint32_t kind;
    union {
        trtmc_feature_text_pair_v1 text;
        trtmc_feature_tokenized_pair_v1 tokens;
    } as;
} trtmc_text_pair_to_token_features_request_v1;
typedef struct {
    trtmc_text_source_v1 text;
} trtmc_text_to_pooled_features_request_v1;
typedef struct {
    trtmc_string_view text;
    uint32_t role;
} trtmc_text_to_embedding_request_v1;
typedef struct {
    trtmc_string_view title;
    trtmc_string_view body;
} trtmc_title_body_to_embedding_request_v1;
typedef struct {
    trtmc_text_source_v1 text;
} trtmc_masked_text_to_token_scores_request_v1;
typedef struct {
    trtmc_string_view first;
    trtmc_string_view second;
} trtmc_text_pair_to_pretraining_relation_scores_request_v1;
typedef struct {
    trtmc_text_source_v1 text;
} trtmc_text_to_replaced_token_scores_request_v1;
typedef struct {
    trtmc_i64_view token_ids;
    trtmc_feature_u8_view attention_mask;
    trtmc_i64_view segment_ids;
    trtmc_feature_u8_view blocked_attention;
    trtmc_feature_u64_view prediction_positions;
} trtmc_text_prediction_positions_to_token_scores_request_v1;
typedef struct {
    trtmc_image_input_v1 image;
} trtmc_image_to_token_features_request_v1;
typedef struct {
    trtmc_image_input_v1 image;
} trtmc_image_to_token_and_pooled_features_request_v1;
typedef struct {
    trtmc_image_input_v1 image;
} trtmc_image_to_spatial_features_request_v1;
typedef struct {
    trtmc_image_input_v1 image;
} trtmc_image_to_pooled_features_request_v1;
typedef struct {
    trtmc_image_input_v1 image;
} trtmc_image_to_embedding_request_v1;
typedef struct {
    trtmc_image_input_v1 image;
    trtmc_string_view text;
} trtmc_image_text_to_embedding_request_v1;
typedef struct {
    trtmc_string_view query;
    trtmc_string_view document;
} trtmc_text_pair_to_relevance_request_v1;
typedef struct {
    trtmc_string_view query;
    trtmc_image_input_v1 document;
} trtmc_text_image_to_relevance_request_v1;
typedef struct {
    trtmc_string_view query;
    trtmc_image_input_v1 image;
    trtmc_string_view document_text;
} trtmc_text_image_text_to_relevance_request_v1;
typedef struct {
    trtmc_image_input_v1 image;
} trtmc_image_to_class_scores_request_v1;

/* XLNet-style input: blocked_attention is [L,L], row=query and column=key,
 * with 1 blocking attention. prediction_positions retains order and duplicates.
 * No persistent memory state is accepted by this synchronous Task.
 * Input buffers are borrowed only through run; every result view is owned by
 * its trtmc_result handle and survives model_release. */

#define TRTMC_TASK_TEXT_TO_TOKEN_FEATURES "text_to_token_features"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_to_token_features_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_token_features_view_v1*,
                                          trtmc_error**);
} trtmc_text_to_token_features_api_v1;

#define TRTMC_TASK_TEXT_PAIR_TO_TOKEN_FEATURES "text_pair_to_token_features"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_pair_to_token_features_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_token_features_view_v1*,
                                          trtmc_error**);
} trtmc_text_pair_to_token_features_api_v1;

#define TRTMC_TASK_TEXT_TO_POOLED_FEATURES "text_to_pooled_features"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_to_pooled_features_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_pooled_features_view_v1*,
                                          trtmc_error**);
} trtmc_text_to_pooled_features_api_v1;

#define TRTMC_TASK_TEXT_TO_EMBEDDING "text_to_embedding"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_to_embedding_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_semantic_embedding_view_v1*,
                                          trtmc_error**);
} trtmc_text_to_embedding_api_v1;

#define TRTMC_TASK_TITLE_BODY_TO_EMBEDDING "title_body_to_embedding"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_title_body_to_embedding_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_semantic_embedding_view_v1*,
                                          trtmc_error**);
} trtmc_title_body_to_embedding_api_v1;

#define TRTMC_TASK_MASKED_TEXT_TO_TOKEN_SCORES "masked_text_to_token_scores"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_masked_text_to_token_scores_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_vocabulary_scores_view_v1*,
                                          trtmc_error**);
} trtmc_masked_text_to_token_scores_api_v1;

#define TRTMC_TASK_TEXT_PAIR_TO_PRETRAINING_RELATION_SCORES                                        \
    "text_pair_to_pretraining_relation_scores"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_text_pair_to_pretraining_relation_scores_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_label_scores_view_v1*,
                                          trtmc_error**);
} trtmc_text_pair_to_pretraining_relation_scores_api_v1;

#define TRTMC_TASK_TEXT_TO_REPLACED_TOKEN_SCORES "text_to_replaced_token_scores"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_text_to_replaced_token_scores_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_replaced_token_scores_view_v1*,
                                          trtmc_error**);
} trtmc_text_to_replaced_token_scores_api_v1;

#define TRTMC_TASK_TEXT_PREDICTION_POSITIONS_TO_TOKEN_SCORES                                       \
    "text_prediction_positions_to_token_scores"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_text_prediction_positions_to_token_scores_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_vocabulary_scores_view_v1*,
                                          trtmc_error**);
} trtmc_text_prediction_positions_to_token_scores_api_v1;

#define TRTMC_TASK_IMAGE_TO_TOKEN_FEATURES "image_to_token_features"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_image_to_token_features_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_image_token_features_view_v1*,
                                          trtmc_error**);
} trtmc_image_to_token_features_api_v1;

#define TRTMC_TASK_IMAGE_TO_SPATIAL_FEATURES "image_to_spatial_features"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_image_to_spatial_features_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_spatial_features_view_v1*,
                                          trtmc_error**);
} trtmc_image_to_spatial_features_api_v1;

#define TRTMC_TASK_IMAGE_TO_POOLED_FEATURES "image_to_pooled_features"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_image_to_pooled_features_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_pooled_features_view_v1*,
                                          trtmc_error**);
} trtmc_image_to_pooled_features_api_v1;

#define TRTMC_TASK_IMAGE_TO_EMBEDDING "image_to_embedding"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_image_to_embedding_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_semantic_embedding_view_v1*,
                                          trtmc_error**);
} trtmc_image_to_embedding_api_v1;

#define TRTMC_TASK_IMAGE_TEXT_TO_EMBEDDING "image_text_to_embedding"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_image_text_to_embedding_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_semantic_embedding_view_v1*,
                                          trtmc_error**);
} trtmc_image_text_to_embedding_api_v1;

#define TRTMC_TASK_TEXT_PAIR_TO_RELEVANCE "text_pair_to_relevance"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_pair_to_relevance_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_relevance_view_v1*,
                                          trtmc_error**);
} trtmc_text_pair_to_relevance_api_v1;

#define TRTMC_TASK_TEXT_IMAGE_TO_RELEVANCE "text_image_to_relevance"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_image_to_relevance_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_relevance_view_v1*,
                                          trtmc_error**);
} trtmc_text_image_to_relevance_api_v1;

#define TRTMC_TASK_TEXT_IMAGE_TEXT_TO_RELEVANCE "text_image_text_to_relevance"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_text_image_text_to_relevance_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_relevance_view_v1*,
                                          trtmc_error**);
} trtmc_text_image_text_to_relevance_api_v1;

#define TRTMC_TASK_IMAGE_TO_CLASS_SCORES "image_to_class_scores"
/* Nonempty scores, indexed by vocabulary ordinal. A result provides either a
 * nonempty label per class or an explicit vocabulary_id. score_kind declares
 * interpretation; shared code does not normalize or relabel the scores.
 * The same contract applies to each native batch item below. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_image_to_class_scores_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_label_scores_view_v1*,
                                          trtmc_error**);
} trtmc_image_to_class_scores_api_v1;

#define TRTMC_TASK_IMAGE_TO_TOKEN_AND_POOLED_FEATURES "image_to_token_and_pooled_features"
typedef struct {
    trtmc_api_header header;
    /* One family extraction returns both outputs; no implicit second Task call. */
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_image_to_token_and_pooled_features_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*,
                                          trtmc_image_token_and_pooled_features_view_v1*,
                                          trtmc_error**);
} trtmc_image_to_token_and_pooled_features_api_v1;

/* Native feature batches contain independent complete requests and per-item
 * config. At least one item is required. Results retain input order and expose
 * the exact scalar view, including its axes and metadata. Item views borrow
 * the batch result and remain valid until its result_release.
 * An error returns no partial batch handle; this is not a transaction and
 * does not roll back native side effects. No implicit scalar execution. */
typedef struct {
    trtmc_image_to_class_scores_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_image_to_class_scores_item_v1;
typedef struct {
    const trtmc_batch_image_to_class_scores_item_v1* items;
    uint64_t count;
} trtmc_batch_image_to_class_scores_request_v1;
#define TRTMC_TASK_BATCH_IMAGE_TO_CLASS_SCORES "batch_image_to_class_scores"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_batch_image_to_class_scores_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_label_scores_view_v1*, trtmc_error**);
} trtmc_batch_image_to_class_scores_api_v1;

typedef struct {
    trtmc_image_to_token_features_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_image_to_token_features_item_v1;
typedef struct {
    const trtmc_batch_image_to_token_features_item_v1* items;
    uint64_t count;
} trtmc_batch_image_to_token_features_request_v1;
#define TRTMC_TASK_BATCH_IMAGE_TO_TOKEN_FEATURES "batch_image_to_token_features"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_image_to_token_features_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_image_token_features_view_v1*, trtmc_error**);
} trtmc_batch_image_to_token_features_api_v1;
typedef struct {
    trtmc_image_to_spatial_features_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_image_to_spatial_features_item_v1;
typedef struct {
    const trtmc_batch_image_to_spatial_features_item_v1* items;
    uint64_t count;
} trtmc_batch_image_to_spatial_features_request_v1;
#define TRTMC_TASK_BATCH_IMAGE_TO_SPATIAL_FEATURES "batch_image_to_spatial_features"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_image_to_spatial_features_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_spatial_features_view_v1*, trtmc_error**);
} trtmc_batch_image_to_spatial_features_api_v1;
typedef struct {
    trtmc_image_to_pooled_features_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_image_to_pooled_features_item_v1;
typedef struct {
    const trtmc_batch_image_to_pooled_features_item_v1* items;
    uint64_t count;
} trtmc_batch_image_to_pooled_features_request_v1;
#define TRTMC_TASK_BATCH_IMAGE_TO_POOLED_FEATURES "batch_image_to_pooled_features"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_image_to_pooled_features_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_pooled_features_view_v1*, trtmc_error**);
} trtmc_batch_image_to_pooled_features_api_v1;
typedef struct {
    trtmc_text_to_embedding_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_text_to_embedding_item_v1;
typedef struct {
    const trtmc_batch_text_to_embedding_item_v1* items;
    uint64_t count;
} trtmc_batch_text_to_embedding_request_v1;
#define TRTMC_TASK_BATCH_TEXT_TO_EMBEDDING "batch_text_to_embedding"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_batch_text_to_embedding_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_semantic_embedding_view_v1*, trtmc_error**);
} trtmc_batch_text_to_embedding_api_v1;
typedef struct {
    trtmc_text_to_token_features_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_text_to_token_features_item_v1;
typedef struct {
    const trtmc_batch_text_to_token_features_item_v1* items;
    uint64_t count;
} trtmc_batch_text_to_token_features_request_v1;
#define TRTMC_TASK_BATCH_TEXT_TO_TOKEN_FEATURES "batch_text_to_token_features"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_text_to_token_features_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_token_features_view_v1*, trtmc_error**);
} trtmc_batch_text_to_token_features_api_v1;
typedef struct {
    trtmc_image_to_token_and_pooled_features_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_image_to_token_and_pooled_features_item_v1;
typedef struct {
    const trtmc_batch_image_to_token_and_pooled_features_item_v1* items;
    uint64_t count;
} trtmc_batch_image_to_token_and_pooled_features_request_v1;
#define TRTMC_TASK_BATCH_IMAGE_TO_TOKEN_AND_POOLED_FEATURES                                        \
    "batch_image_to_token_and_pooled_features"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_image_to_token_and_pooled_features_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_image_token_and_pooled_features_view_v1*,
                                               trtmc_error**);
} trtmc_batch_image_to_token_and_pooled_features_api_v1;

#ifdef __cplusplus
}
#endif

#endif /* TRTMC_FEATURES_H */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TRTMC_TEXT_C_API_H
#define TRTMC_TEXT_C_API_H

#include "trtmc/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TRTMC_TASK_CONDITIONAL_TEXT_GENERATION "conditional_text_generation"
#define TRTMC_TASK_CORRUPTED_TEXT_RECONSTRUCTION "corrupted_text_reconstruction"
#define TRTMC_TASK_UNCONDITIONAL_TEXT_GENERATION "unconditional_text_generation"
#define TRTMC_TASK_TEXT_TRANSLATION "text_translation"
#define TRTMC_TASK_TEXT_SUMMARIZATION "text_summarization"
#define TRTMC_TASK_TEXT_PREFIX_SUFFIX_INFILLING "text_prefix_suffix_infilling"
#define TRTMC_TASK_CONTEXT_QUESTION_ANSWERING "context_question_answering"
#define TRTMC_TASK_BATCH_TEXT_CONTINUATION "batch_text_continuation"

typedef struct {
    trtmc_text_source_v1 source;
} trtmc_conditional_text_generation_request_v1;

typedef struct {
    trtmc_string_view corrupted_text;
} trtmc_corrupted_text_reconstruction_request_v1;

typedef struct {
    trtmc_string_view source_text;
    /* Absent languages select only defaults declared by the loaded family.
     * A family without an applicable default rejects the missing language. */
    uint32_t has_target_language;      /* Exactly zero or one. */
    trtmc_string_view target_language; /* Read only when present; then nonempty. */
    uint32_t has_source_language;      /* Exactly zero or one. */
    trtmc_string_view source_language; /* Read only when present; then nonempty. */
} trtmc_text_translation_request_v1;

typedef struct {
    trtmc_string_view document;
} trtmc_text_summarization_request_v1;

typedef struct {
    trtmc_string_view prefix;
    trtmc_string_view suffix;
} trtmc_text_prefix_suffix_infilling_request_v1;

typedef struct {
    trtmc_string_view question;
    trtmc_string_view context;
} trtmc_context_question_answering_request_v1;

/* A source conditions newly generated target text; it is not a decoder prefix. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_conditional_text_generation_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_conditional_text_generation_api_v1;

/* Returns the complete reconstructed text, not only the replacement spans. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_corrupted_text_reconstruction_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_corrupted_text_reconstruction_api_v1;

/* This Task has no conditioning request. Config contains optional generation
 * controls only; a fake empty input struct is unnecessary. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_config_view_v1*, trtmc_result**,
                                  trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_unconditional_text_generation_api_v1;

typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_translation_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_text_translation_api_v1;

typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_summarization_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_text_summarization_api_v1;

/* Returns only the generated missing middle, without repeating either boundary
 * or exposing family-specific sentinel formatting. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_text_prefix_suffix_infilling_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_text_prefix_suffix_infilling_api_v1;

/* Returns generated answer text, not extractive offsets or calibrated scores. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_context_question_answering_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_context_question_answering_api_v1;

typedef struct {
    trtmc_text_continuation_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_text_continuation_item_v1;

typedef struct {
    const trtmc_batch_text_continuation_item_v1* items;
    uint64_t count;
} trtmc_batch_text_continuation_request_v1;

/* All items are validated before execution. Success returns one result per
 * input in order. Failure returns no partial result. Item views borrow the
 * batch result and remain valid until core result_release. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run_batch)(trtmc_model*,
                                        const trtmc_batch_text_continuation_request_v1*,
                                        trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_text_result_view_v1*, trtmc_error**);
} trtmc_batch_text_continuation_api_v1;

#ifdef __cplusplus
}
#endif

#endif /* TRTMC_TEXT_C_API_H */

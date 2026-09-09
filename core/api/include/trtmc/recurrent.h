/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TRTMC_RECURRENT_H
#define TRTMC_RECURRENT_H
#include "trtmc/types.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct trtmc_recurrent_state trtmc_recurrent_state;
/* All arrays are contiguous row-major. Host buffers use device_ordinal = -1;
 * CUDA buffers use their nonnegative device ordinal and never require CUDA
 * headers in the SDK. Inputs are borrowed and ready before the synchronous
 * call; the family finishes reading them before returning. Result arrays are
 * immutable, ready on return, and valid until core.result_release. No implicit
 * dtype cast, host/device transfer, padding, tokenization, or sampling occurs. */
enum { TRTMC_RECURRENT_FLOAT32 = 1, TRTMC_RECURRENT_FLOAT16 = 2, TRTMC_RECURRENT_BFLOAT16 = 3 };
enum { TRTMC_RECURRENT_HOST = 1, TRTMC_RECURRENT_CUDA = 2 };
typedef struct {
    const void* data;
    uint64_t byte_size;
    uint32_t scalar, memory;
    int32_t device_ordinal;
} trtmc_recurrent_buffer_v1;
typedef struct {
    trtmc_recurrent_buffer_v1 buffer;
    uint64_t rows, columns;
} trtmc_recurrent_array_v1;
typedef struct {
    trtmc_i32_view token_ids;
    const uint8_t* input_mask;
    uint64_t mask_count;
} trtmc_recurrent_tokens_input_v1;
typedef struct {
    trtmc_recurrent_array_v1 embeddings; /* [T,H] for exactly one sequence. */
    const uint8_t* input_mask;
    uint64_t mask_count;
} trtmc_recurrent_embeddings_input_v1;
enum {
    TRTMC_RECURRENT_ROWS_ALL = 0,
    TRTMC_RECURRENT_ROWS_LAST = 1,
    TRTMC_RECURRENT_ROWS_INDICES = 2
};
typedef struct {
    uint32_t kind;
    uint64_t last_count;
    const uint64_t* indices;
    uint64_t index_count;
} trtmc_recurrent_logit_rows_v1;
typedef struct {
    trtmc_recurrent_tokens_input_v1 input;
    trtmc_recurrent_logit_rows_v1 rows;
    uint32_t output_memory;
} trtmc_recurrent_tokens_logits_request_v1;
typedef struct {
    trtmc_recurrent_embeddings_input_v1 input;
    trtmc_recurrent_logit_rows_v1 rows;
    uint32_t output_memory;
} trtmc_recurrent_embeddings_logits_request_v1;
typedef struct {
    trtmc_recurrent_tokens_input_v1 input;
    uint32_t output_memory;
} trtmc_recurrent_tokens_hidden_request_v1;
typedef struct {
    trtmc_recurrent_embeddings_input_v1 input;
    uint32_t output_memory;
} trtmc_recurrent_embeddings_hidden_request_v1;
enum {
    TRTMC_RECURRENT_POST_BLOCK = 1,
    TRTMC_RECURRENT_FINAL_NORMALIZATION = 2,
    TRTMC_RECURRENT_MIXER_CONTRIBUTION = 3
};
typedef struct {
    uint32_t stage;
    int64_t block_index;
    trtmc_recurrent_array_v1 values;
} trtmc_recurrent_trace_item_v1;
typedef struct {
    uint32_t has_trace;
    const trtmc_recurrent_trace_item_v1* items;
    uint64_t count;
} trtmc_recurrent_trace_v1;
typedef struct {
    trtmc_recurrent_array_v1 logits; /* [selected input rows,V], unnormalized. */
    const uint64_t* token_positions;
    uint64_t position_count;
    trtmc_string_view vocabulary_id;
    trtmc_recurrent_trace_v1 trace;
} trtmc_recurrent_logits_view_v1;
typedef struct {
    trtmc_recurrent_array_v1 hidden; /* [T,H], final normalized hidden values. */
    trtmc_recurrent_trace_v1 trace;
} trtmc_recurrent_hidden_view_v1;
typedef struct {
    uint32_t poisoned, context_valid, initialized, has_tokens_seen;
    uint64_t tokens_seen, device_memory_bytes;
} trtmc_recurrent_state_info_v1;
typedef struct {
    trtmc_recurrent_array_v1 convolution, ssm; /* [D,K] oldest-first and [D,N]. */
    uint32_t has_previous_state;
} trtmc_mamba1_layer_state_v1;
typedef struct {
    const trtmc_mamba1_layer_state_v1* layers;
    uint64_t layer_count;
    uint32_t has_tokens_seen;
    uint64_t tokens_seen;
} trtmc_mamba1_state_view_v1;
typedef struct {
    trtmc_recurrent_array_v1 ffn_previous, attention_previous, wkv_numerator, wkv_denominator,
        wkv_running_max;
    /* Each component is [channel,layer], never a hidden batch axis. */
    uint32_t has_tokens_seen;
    uint64_t tokens_seen;
} trtmc_rwkv4_state_view_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* assign)(trtmc_recurrent_state*, const trtmc_mamba1_state_view_v1*,
                                     trtmc_error**);
    trtmc_status(TRTMC_CALL* snapshot)(trtmc_recurrent_state*, uint32_t memory, trtmc_result**,
                                       trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_mamba1_state_view_v1*,
                                          trtmc_error**);
} trtmc_mamba1_state_exchange_api_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* assign)(trtmc_recurrent_state*, const trtmc_rwkv4_state_view_v1*,
                                     trtmc_error**);
    trtmc_status(TRTMC_CALL* snapshot)(trtmc_recurrent_state*, uint32_t memory, trtmc_result**,
                                       trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_rwkv4_state_view_v1*,
                                          trtmc_error**);
} trtmc_rwkv4_state_exchange_api_v1;
/* Idle states retain model/DSO without reserving execution. Forward mutates one
 * sequence; clone explicitly forks independent state. A state is never a batch
 * row group or a reversible token history. Concurrent execution returns BUSY.
 * Family preflight INVALID_ARGUMENT/INVALID_CONFIG/UNSUPPORTED rejection preserves existing
 * state, including existing poison. Execution or result-packing failure poisons
 * all submitted states; successful reset or typed assign is explicit recovery.
 * info.context_valid independently reports family-owned effective-weight/config
 * validity. No weight hash is computed. Unknown tokens_seen is never inferred.
 * Typed assign copies/consumes borrowed arrays before return; it must not retain
 * caller storage. Snapshot and clone must not alias mutable source storage.
 * Callers must not release state concurrently with any use of that handle. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* clone)(const trtmc_recurrent_state*, trtmc_recurrent_state**,
                                    trtmc_error**);
    trtmc_status(TRTMC_CALL* reset)(trtmc_recurrent_state*, trtmc_error**);
    trtmc_status(TRTMC_CALL* info)(const trtmc_recurrent_state*, trtmc_recurrent_state_info_v1*,
                                   trtmc_error**);
    trtmc_status(TRTMC_CALL* get_mamba1_exchange)(trtmc_recurrent_state*, uint32_t, uint32_t,
                                                  const trtmc_mamba1_state_exchange_api_v1**,
                                                  trtmc_error**);
    trtmc_status(TRTMC_CALL* get_rwkv4_exchange)(trtmc_recurrent_state*, uint32_t, uint32_t,
                                                 const trtmc_rwkv4_state_exchange_api_v1**,
                                                 trtmc_error**);
    void(TRTMC_CALL* release)(trtmc_recurrent_state*);
} trtmc_recurrent_state_api_v1;
#define TRTMC_TASK_RECURRENT_TOKENS_TO_LOGITS "recurrent_tokens_to_logits"
#define TRTMC_TASK_RECURRENT_EMBEDDINGS_TO_LOGITS "recurrent_embeddings_to_logits"
#define TRTMC_TASK_RECURRENT_TOKENS_TO_HIDDEN_STATES "recurrent_tokens_to_hidden_states"
#define TRTMC_TASK_RECURRENT_EMBEDDINGS_TO_HIDDEN_STATES "recurrent_embeddings_to_hidden_states"
#define TRTMC_TASK_BATCH_RECURRENT_TOKENS_TO_LOGITS "batch_recurrent_tokens_to_logits"
#define TRTMC_TASK_BATCH_RECURRENT_EMBEDDINGS_TO_LOGITS "batch_recurrent_embeddings_to_logits"
#define TRTMC_TASK_BATCH_RECURRENT_TOKENS_TO_HIDDEN_STATES "batch_recurrent_tokens_to_hidden_states"
#define TRTMC_TASK_BATCH_RECURRENT_EMBEDDINGS_TO_HIDDEN_STATES                                     \
    "batch_recurrent_embeddings_to_hidden_states"
#define TRTMC_RECURRENT_API(Name, Request, View)                                                   \
    typedef struct {                                                                               \
        trtmc_api_header header;                                                                   \
        trtmc_status(TRTMC_CALL* create_state)(trtmc_model*, trtmc_recurrent_state**,              \
                                               trtmc_error**);                                     \
        trtmc_status(TRTMC_CALL* forward)(trtmc_recurrent_state*, const Request*,                  \
                                          const trtmc_config_view_v1*, trtmc_result**,             \
                                          trtmc_error**);                                          \
        trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, View*, trtmc_error**);          \
        const trtmc_recurrent_state_api_v1* state_api;                                             \
    } trtmc_##Name##_api_v1;
TRTMC_RECURRENT_API(recurrent_tokens_to_logits, trtmc_recurrent_tokens_logits_request_v1,
                    trtmc_recurrent_logits_view_v1)
TRTMC_RECURRENT_API(recurrent_embeddings_to_logits, trtmc_recurrent_embeddings_logits_request_v1,
                    trtmc_recurrent_logits_view_v1)
TRTMC_RECURRENT_API(recurrent_tokens_to_hidden_states, trtmc_recurrent_tokens_hidden_request_v1,
                    trtmc_recurrent_hidden_view_v1)
TRTMC_RECURRENT_API(recurrent_embeddings_to_hidden_states,
                    trtmc_recurrent_embeddings_hidden_request_v1, trtmc_recurrent_hidden_view_v1)
#undef TRTMC_RECURRENT_API
#define TRTMC_RECURRENT_BATCH_API(Name, Request, View)                                             \
    typedef struct {                                                                               \
        trtmc_recurrent_state* state;                                                              \
        Request request;                                                                           \
        trtmc_config_view_v1 config;                                                               \
    } trtmc_##Name##_item_v1;                                                                      \
    typedef struct {                                                                               \
        trtmc_api_header header;                                                                   \
        trtmc_status(TRTMC_CALL* create_state)(trtmc_model*, trtmc_recurrent_state**,              \
                                               trtmc_error**);                                     \
        trtmc_status(TRTMC_CALL* forward)(const trtmc_##Name##_item_v1*, uint64_t, trtmc_result**, \
                                          trtmc_error**);                                          \
        trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);     \
        trtmc_status(TRTMC_CALL* result_item)(const trtmc_result*, uint64_t, View*,                \
                                              trtmc_error**);                                      \
        const trtmc_recurrent_state_api_v1* state_api;                                             \
    } trtmc_##Name##_api_v1;
TRTMC_RECURRENT_BATCH_API(batch_recurrent_tokens_to_logits,
                          trtmc_recurrent_tokens_logits_request_v1, trtmc_recurrent_logits_view_v1)
TRTMC_RECURRENT_BATCH_API(batch_recurrent_embeddings_to_logits,
                          trtmc_recurrent_embeddings_logits_request_v1,
                          trtmc_recurrent_logits_view_v1)
TRTMC_RECURRENT_BATCH_API(batch_recurrent_tokens_to_hidden_states,
                          trtmc_recurrent_tokens_hidden_request_v1, trtmc_recurrent_hidden_view_v1)
TRTMC_RECURRENT_BATCH_API(batch_recurrent_embeddings_to_hidden_states,
                          trtmc_recurrent_embeddings_hidden_request_v1,
                          trtmc_recurrent_hidden_view_v1)
#undef TRTMC_RECURRENT_BATCH_API
#ifdef __cplusplus
}
#endif
#endif

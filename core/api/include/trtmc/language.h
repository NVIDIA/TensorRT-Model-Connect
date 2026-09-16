/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TRTMC_LANGUAGE_H
#define TRTMC_LANGUAGE_H

#include "trtmc/audio.h"
#include "trtmc/matrix.h"
#include "trtmc/tools.h"
#include "trtmc/video.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    TRTMC_ROLE_SYSTEM = 1,
    TRTMC_ROLE_DEVELOPER = 2,
    TRTMC_ROLE_USER = 3,
    TRTMC_ROLE_ASSISTANT = 4,
    TRTMC_ROLE_TOOL = 5
};
enum {
    TRTMC_MEDIA_TEXT = 1,
    TRTMC_MEDIA_IMAGE = 2,
    TRTMC_MEDIA_VIDEO = 3,
    TRTMC_MEDIA_AUDIO = 4,
    TRTMC_MEDIA_ALIGNED_AUDIO_VIDEO = 5,
    TRTMC_MEDIA_REASONING = 6,
    TRTMC_MEDIA_TOOL_CALL = 7,
    TRTMC_MEDIA_TOOL_RESULT = 8
};
typedef struct {
    trtmc_video_view_v1 video;
    trtmc_audio_view_v1 audio;
    uint32_t has_audio_start;
    double audio_start_seconds; /* Same clock as video; absent only with family-declared default. */
} trtmc_aligned_audio_video_input_v1;

/* Media messages preserve roles and ordered/interleaved parts. Their closed
 * part unions differ by Task. Image/Video conversation inputs also retain
 * assistant reasoning/tool calls and correlated tool-result messages.
 * Family-specific limits and default timing/rates remain family-owned. */
typedef struct {
    uint32_t kind;
    union {
        trtmc_string_view text;
        trtmc_image_input_v1 image;
        trtmc_tool_call_v1 tool_call;
        trtmc_tool_result_v1 tool_result;
    } content;
} trtmc_images_text_part_v1;
typedef struct {
    uint32_t role;
    const trtmc_images_text_part_v1* parts;
    uint64_t part_count;
} trtmc_images_text_message_v1;

typedef struct {
    uint32_t kind;
    union {
        trtmc_string_view text;
        trtmc_video_view_v1 video;
        trtmc_tool_call_v1 tool_call;
        trtmc_tool_result_v1 tool_result;
    } content;
} trtmc_video_text_part_v1;
typedef struct {
    uint32_t role;
    const trtmc_video_text_part_v1* parts;
    uint64_t part_count;
} trtmc_video_text_message_v1;

typedef struct {
    uint32_t kind;
    union {
        trtmc_string_view text;
        trtmc_image_input_v1 image;
        trtmc_video_view_v1 video;
    } content;
} trtmc_image_video_text_part_v1;
typedef struct {
    uint32_t role;
    const trtmc_image_video_text_part_v1* parts;
    uint64_t part_count;
} trtmc_image_video_text_message_v1;

typedef struct {
    uint32_t kind;
    union {
        trtmc_string_view text;
        trtmc_audio_view_v1 audio;
    } content;
} trtmc_audio_text_part_v1;
typedef struct {
    uint32_t role;
    const trtmc_audio_text_part_v1* parts;
    uint64_t part_count;
} trtmc_audio_text_message_v1;

typedef struct {
    uint32_t kind;
    union {
        trtmc_string_view text;
        trtmc_image_input_v1 image;
        trtmc_audio_view_v1 audio;
    } content;
} trtmc_image_audio_text_part_v1;
typedef struct {
    uint32_t role;
    const trtmc_image_audio_text_part_v1* parts;
    uint64_t part_count;
} trtmc_image_audio_text_message_v1;

typedef struct {
    uint32_t kind;
    union {
        trtmc_string_view text;
        trtmc_aligned_audio_video_input_v1 aligned;
    } content;
} trtmc_audio_video_text_part_v1;
typedef struct {
    uint32_t role;
    const trtmc_audio_video_text_part_v1* parts;
    uint64_t part_count;
} trtmc_audio_video_text_message_v1;

typedef struct {
    const trtmc_images_text_message_v1* messages;
    uint64_t message_count;
    const trtmc_tool_definition_v1* tools;
    uint64_t tool_count;
} trtmc_images_text_to_text_request_v1;
typedef struct {
    const trtmc_video_text_message_v1* messages;
    uint64_t message_count;
    const trtmc_tool_definition_v1* tools;
    uint64_t tool_count;
} trtmc_video_text_to_text_request_v1;
typedef struct {
    const trtmc_image_video_text_message_v1* messages;
    uint64_t message_count;
} trtmc_image_video_text_to_text_request_v1;
typedef struct {
    const trtmc_audio_text_message_v1* messages;
    uint64_t message_count;
} trtmc_audio_text_to_text_request_v1;
typedef struct {
    const trtmc_image_audio_text_message_v1* messages;
    uint64_t message_count;
} trtmc_image_audio_to_text_request_v1;
typedef struct {
    const trtmc_audio_video_text_message_v1* messages;
    uint64_t message_count;
} trtmc_audio_video_text_to_text_request_v1;
typedef struct {
    const trtmc_image_audio_text_message_v1* messages;
    uint64_t message_count;
} trtmc_image_audio_text_to_text_request_v1;
typedef struct {
    const trtmc_image_audio_text_message_v1* messages;
    uint64_t message_count;
} trtmc_image_audio_text_to_text_speech_response_request_v1;

enum {
    TRTMC_CONVERSATION_TEXT = 1,
    TRTMC_CONVERSATION_REASONING = 2,
    TRTMC_CONVERSATION_TOOL_CALL = 3,
    TRTMC_CONVERSATION_TOOL_RESULT = 4
};
typedef struct {
    uint32_t kind;
    union {
        trtmc_string_view text;
        trtmc_tool_call_v1 tool_call;
        trtmc_tool_result_v1 tool_result;
    } content;
} trtmc_conversation_part_v1;
typedef struct {
    uint32_t role;
    const trtmc_conversation_part_v1* parts;
    uint64_t part_count;
} trtmc_conversation_message_v1;
typedef struct {
    const trtmc_conversation_message_v1* messages;
    uint64_t message_count;
    const trtmc_tool_definition_v1* tools;
    uint64_t tool_count;
} trtmc_text_conversation_request_v1;
enum {
    TRTMC_FINISH_UNKNOWN = 0,
    TRTMC_FINISH_STOP = 1,
    TRTMC_FINISH_LENGTH = 2,
    TRTMC_FINISH_TOOL_CALLS = 3,
    TRTMC_FINISH_CONTENT_FILTER = 4,
    TRTMC_FINISH_OTHER = 5
};
typedef struct {
    uint32_t has_input_tokens, has_output_tokens, has_total_tokens;
    uint64_t input_tokens, output_tokens, total_tokens;
} trtmc_token_usage_v1;
typedef struct {
    /* Assistant output only: text, reasoning or tool call, never tool result. */
    const trtmc_conversation_part_v1* parts;
    uint64_t part_count;
    trtmc_i32_view token_ids;
    double setup_ms, prefill_ms, decode_ms;
    uint32_t finish_reason;
    trtmc_string_view other_finish_reason;
    trtmc_token_usage_v1 usage;
} trtmc_conversation_result_view_v1;
typedef struct {
    trtmc_text_result_view_v1 text;
    trtmc_audio_result_view_v1 speech;
} trtmc_text_speech_result_view_v1; /* Paired response; no word alignment is implied. */
typedef struct {
    trtmc_string_view text;
} trtmc_text_label_classification_request_v1;
typedef struct {
    trtmc_string_view first;
    trtmc_string_view second;
} trtmc_text_pair_label_classification_request_v1;
typedef struct {
    trtmc_string_view raw_label;
    int64_t label_index; /* -1 means unmapped; raw model output is always preserved. */
    trtmc_strings_view vocabulary;
    trtmc_i32_view token_ids;
} trtmc_generated_label_result_view_v1;
typedef struct {
    const uint8_t* data;
    uint64_t count;
} trtmc_language_mask_view_v1;
typedef struct {
    trtmc_i32_view token_ids;
    trtmc_language_mask_view_v1 attention_mask; /* Empty means all valid. */
} trtmc_language_token_sequence_v1;
typedef struct {
    trtmc_language_token_sequence_v1 source;
    trtmc_language_token_sequence_v1 decoder;
} trtmc_text_encoder_decoder_hidden_states_request_v1;
typedef struct {
    trtmc_f32_matrix_view_v1 last_hidden_state;
    trtmc_i32_view token_ids;
    trtmc_language_mask_view_v1 attention_mask; /* Effective mask, one per row. */
} trtmc_token_axis_states_view_v1;
typedef struct {
    trtmc_token_axis_states_view_v1 encoder;
    trtmc_token_axis_states_view_v1 decoder;
} trtmc_encoder_decoder_states_result_view_v1;

/* Full caller-supplied token sequences are preserved, including padding rows.
 * Encoder and decoder axes are separate; this is final-layer state, not a
 * normalized embedding and not a promise to expose every intermediate layer. */

#define TRTMC_TASK_IMAGES_TEXT_TO_TEXT "images_text_to_text"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_images_text_to_text_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_images_text_to_text_api_v1;

#define TRTMC_TASK_VIDEO_TEXT_TO_TEXT "video_text_to_text"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_video_text_to_text_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_video_text_to_text_api_v1;

#define TRTMC_TASK_IMAGE_VIDEO_TEXT_TO_TEXT "image_video_text_to_text"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_image_video_text_to_text_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_image_video_text_to_text_api_v1;

#define TRTMC_TASK_AUDIO_TEXT_TO_TEXT "audio_text_to_text"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_audio_text_to_text_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_audio_text_to_text_api_v1;

#define TRTMC_TASK_IMAGE_AUDIO_TO_TEXT "image_audio_to_text"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_image_audio_to_text_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_image_audio_to_text_api_v1;

#define TRTMC_TASK_AUDIO_VIDEO_TEXT_TO_TEXT "audio_video_text_to_text"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_audio_video_text_to_text_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_audio_video_text_to_text_api_v1;

#define TRTMC_TASK_IMAGE_AUDIO_TEXT_TO_TEXT "image_audio_text_to_text"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_image_audio_text_to_text_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_image_audio_text_to_text_api_v1;

#define TRTMC_TASK_IMAGE_AUDIO_TEXT_TO_TEXT_SPEECH_RESPONSE                                        \
    "image_audio_text_to_text_speech_response"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_image_audio_text_to_text_speech_response_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_speech_result_view_v1*,
                                          trtmc_error**);
} trtmc_image_audio_text_to_text_speech_response_api_v1;

#define TRTMC_TASK_TEXT_CONVERSATION "text_conversation"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_conversation_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_conversation_result_view_v1*,
                                          trtmc_error**);
} trtmc_text_conversation_api_v1;
#define TRTMC_TASK_IMAGES_TEXT_CONVERSATION "images_text_conversation"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_images_text_to_text_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_conversation_result_view_v1*,
                                          trtmc_error**);
} trtmc_images_text_conversation_api_v1;
#define TRTMC_TASK_VIDEO_TEXT_CONVERSATION "video_text_conversation"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_video_text_to_text_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_conversation_result_view_v1*,
                                          trtmc_error**);
} trtmc_video_text_conversation_api_v1;

typedef struct {
    uint32_t kind;
    union {
        trtmc_string_view text;
        trtmc_audio_view_v1 audio;
        trtmc_tool_call_v1 tool_call;
        trtmc_tool_result_v1 tool_result;
    } content;
} trtmc_audio_text_conversation_part_v1;
typedef struct {
    uint32_t role;
    const trtmc_audio_text_conversation_part_v1* parts;
    uint64_t part_count;
} trtmc_audio_text_conversation_message_v1;
typedef struct {
    const trtmc_audio_text_conversation_message_v1* messages;
    uint64_t message_count;
    const trtmc_tool_definition_v1* tools;
    uint64_t tool_count;
} trtmc_audio_text_conversation_request_v1;
typedef struct {
    uint32_t kind;
    union {
        trtmc_string_view text;
        trtmc_image_input_v1 image;
        trtmc_audio_view_v1 audio;
        trtmc_tool_call_v1 tool_call;
        trtmc_tool_result_v1 tool_result;
    } content;
} trtmc_image_audio_text_conversation_part_v1;
typedef struct {
    uint32_t role;
    const trtmc_image_audio_text_conversation_part_v1* parts;
    uint64_t part_count;
} trtmc_image_audio_text_conversation_message_v1;
typedef struct {
    const trtmc_image_audio_text_conversation_message_v1* messages;
    uint64_t message_count;
    const trtmc_tool_definition_v1* tools;
    uint64_t tool_count;
} trtmc_image_audio_text_conversation_request_v1;
enum {
    TRTMC_BATCH_CONVERSATION_TEXT = 1,
    TRTMC_BATCH_CONVERSATION_IMAGES_TEXT = 2,
    TRTMC_BATCH_CONVERSATION_VIDEO_TEXT = 3,
    TRTMC_BATCH_CONVERSATION_AUDIO_TEXT = 4,
    TRTMC_BATCH_CONVERSATION_IMAGE_AUDIO_TEXT = 5
};
typedef struct {
    uint32_t kind;
    union {
        trtmc_text_conversation_request_v1 text;
        trtmc_images_text_to_text_request_v1 images_text;
        trtmc_video_text_to_text_request_v1 video_text;
    } input;
} trtmc_text_images_video_conversation_request_v1;
typedef struct {
    uint32_t kind;
    union {
        trtmc_text_conversation_request_v1 text;
        trtmc_images_text_to_text_request_v1 images_text;
        trtmc_audio_text_conversation_request_v1 audio_text;
        trtmc_image_audio_text_conversation_request_v1 image_audio_text;
    } input;
} trtmc_text_images_audio_conversation_request_v1;

#define TRTMC_TASK_BATCH_TEXT_CONVERSATION "batch_text_conversation"
typedef struct {
    trtmc_text_conversation_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_text_conversation_item_v1;
typedef struct {
    const trtmc_batch_text_conversation_item_v1* items;
    uint64_t count;
} trtmc_batch_text_conversation_request_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_batch_text_conversation_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_conversation_result_view_v1*, trtmc_error**);
} trtmc_batch_text_conversation_api_v1;

#define TRTMC_TASK_BATCH_VIDEO_TEXT_CONVERSATION "batch_video_text_conversation"
typedef struct {
    trtmc_video_text_to_text_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_video_text_conversation_item_v1;
typedef struct {
    const trtmc_batch_video_text_conversation_item_v1* items;
    uint64_t count;
} trtmc_batch_video_text_conversation_request_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_video_text_conversation_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_conversation_result_view_v1*, trtmc_error**);
} trtmc_batch_video_text_conversation_api_v1;

#define TRTMC_TASK_BATCH_AUDIO_TEXT_CONVERSATION "batch_audio_text_conversation"
typedef struct {
    trtmc_audio_text_conversation_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_audio_text_conversation_item_v1;
typedef struct {
    const trtmc_batch_audio_text_conversation_item_v1* items;
    uint64_t count;
} trtmc_batch_audio_text_conversation_request_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_audio_text_conversation_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_conversation_result_view_v1*, trtmc_error**);
} trtmc_batch_audio_text_conversation_api_v1;

#define TRTMC_TASK_BATCH_IMAGE_AUDIO_TEXT_CONVERSATION "batch_image_audio_text_conversation"
typedef struct {
    trtmc_image_audio_text_conversation_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_image_audio_text_conversation_item_v1;
typedef struct {
    const trtmc_batch_image_audio_text_conversation_item_v1* items;
    uint64_t count;
} trtmc_batch_image_audio_text_conversation_request_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_image_audio_text_conversation_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_conversation_result_view_v1*, trtmc_error**);
} trtmc_batch_image_audio_text_conversation_api_v1;

#define TRTMC_TASK_BATCH_TEXT_IMAGES_VIDEO_CONVERSATIONS "batch_text_images_video_conversations"
typedef struct {
    trtmc_text_images_video_conversation_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_text_images_video_conversations_item_v1;
typedef struct {
    const trtmc_batch_text_images_video_conversations_item_v1* items;
    uint64_t count;
} trtmc_batch_text_images_video_conversations_request_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_text_images_video_conversations_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_conversation_result_view_v1*, trtmc_error**);
} trtmc_batch_text_images_video_conversations_api_v1;

#define TRTMC_TASK_BATCH_TEXT_IMAGES_AUDIO_CONVERSATIONS "batch_text_images_audio_conversations"
typedef struct {
    trtmc_text_images_audio_conversation_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_text_images_audio_conversations_item_v1;
typedef struct {
    const trtmc_batch_text_images_audio_conversations_item_v1* items;
    uint64_t count;
} trtmc_batch_text_images_audio_conversations_request_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_text_images_audio_conversations_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_conversation_result_view_v1*, trtmc_error**);
} trtmc_batch_text_images_audio_conversations_api_v1;
/* Each item has its own complete history/tools/config. One family batch call
 * returns all ordered results or fails; no rollback of provider side effects
 * is promised. Views borrow their batch result until result_release. */
#define TRTMC_TASK_BATCH_IMAGES_TEXT_CONVERSATION "batch_images_text_conversation"
typedef struct {
    trtmc_images_text_to_text_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_images_text_conversation_item_v1;
typedef struct {
    const trtmc_batch_images_text_conversation_item_v1* items;
    uint64_t count;
} trtmc_batch_images_text_conversation_request_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_images_text_conversation_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_conversation_result_view_v1*, trtmc_error**);
} trtmc_batch_images_text_conversation_api_v1;

#define TRTMC_TASK_TEXT_LABEL_CLASSIFICATION "text_label_classification"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_label_classification_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*,
                                          trtmc_generated_label_result_view_v1*, trtmc_error**);
} trtmc_text_label_classification_api_v1;

#define TRTMC_TASK_TEXT_PAIR_LABEL_CLASSIFICATION "text_pair_label_classification"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_text_pair_label_classification_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*,
                                          trtmc_generated_label_result_view_v1*, trtmc_error**);
} trtmc_text_pair_label_classification_api_v1;

#define TRTMC_TASK_TEXT_ENCODER_DECODER_HIDDEN_STATES "text_encoder_decoder_hidden_states"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_text_encoder_decoder_hidden_states_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*,
                                          trtmc_encoder_decoder_states_result_view_v1*,
                                          trtmc_error**);
} trtmc_text_encoder_decoder_hidden_states_api_v1;

#ifdef __cplusplus
}
#endif
#endif /* TRTMC_LANGUAGE_H */

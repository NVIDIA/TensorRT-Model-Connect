/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TRTMC_STREAM_API_H
#define TRTMC_STREAM_API_H

#include "trtmc/language.h"
#include "trtmc/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct trtmc_text_stream trtmc_text_stream;
enum { TRTMC_STREAM_DELTA = 1, TRTMC_STREAM_COMPLETE = 2, TRTMC_STREAM_CANCELLED = 3 };

typedef struct {
    uint32_t kind;
    trtmc_string_view text_delta;
    trtmc_i32_view token_ids;
    /* Read only for COMPLETE. Other event kinds leave this zero-initialized. */
    trtmc_text_result_view_v1 final_result;
} trtmc_text_stream_event_view_v1;

typedef struct {
    trtmc_api_header header;
    /* OK returns an owned event; AGAIN is a timeout; END follows a terminal
     * event. AGAIN/END return no event or error. -1 blocks, 0 polls.
     * A competing reader returns BUSY without waiting; cancel may run
     * concurrently with next. Fatal family/packing errors stop the stream,
     * release its execution state, and make subsequent reads return END.
     * Pre-call argument errors are retryable. */
    trtmc_status(TRTMC_CALL* next)(trtmc_text_stream*, int64_t timeout_ms, trtmc_result**,
                                   trtmc_error**);
    trtmc_status(TRTMC_CALL* event_view)(const trtmc_result*, trtmc_text_stream_event_view_v1*,
                                         trtmc_error**);
    trtmc_status(TRTMC_CALL* cancel)(trtmc_text_stream*, trtmc_error**);
    /* Null is accepted. Release cancels unfinished work, then joins it before
     * releasing the model. It must not race other calls on the same handle. */
    void(TRTMC_CALL* release)(trtmc_text_stream*);
} trtmc_text_stream_api_v1;

#define TRTMC_TASK_STREAMING_TEXT_CONTINUATION "streaming_text_continuation"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* start)(trtmc_model*, const trtmc_text_continuation_request_v1*,
                                    const trtmc_config_view_v1*, trtmc_text_stream**,
                                    trtmc_error**);
    const trtmc_text_stream_api_v1* stream_api;
} trtmc_streaming_text_continuation_api_v1;
#define TRTMC_TASK_STREAMING_IMAGES_TEXT_TO_TEXT "streaming_images_text_to_text"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* start)(trtmc_model*, const trtmc_images_text_to_text_request_v1*,
                                    const trtmc_config_view_v1*, trtmc_text_stream**,
                                    trtmc_error**);
    const trtmc_text_stream_api_v1* stream_api;
} trtmc_streaming_images_text_to_text_api_v1;

typedef struct trtmc_conversation_stream trtmc_conversation_stream;
enum {
    TRTMC_CONVERSATION_STREAM_TEXT = 1,
    TRTMC_CONVERSATION_STREAM_REASONING = 2,
    TRTMC_CONVERSATION_STREAM_TOOL_DELTA = 3,
    TRTMC_CONVERSATION_STREAM_TOOL_END = 4,
    TRTMC_CONVERSATION_STREAM_TOKENS = 5,
    TRTMC_CONVERSATION_STREAM_COMPLETE = 6,
    TRTMC_CONVERSATION_STREAM_CANCELLED = 7
};
typedef struct {
    uint64_t part_index;
    trtmc_string_view text_delta;
} trtmc_conversation_text_delta_v1;
typedef struct {
    uint64_t part_index;
    uint32_t has_call_id;
    trtmc_string_view call_id, name_delta, arguments_delta;
} trtmc_conversation_tool_delta_v1;
typedef struct {
    uint64_t part_index;
    uint32_t state;
} trtmc_conversation_tool_end_v1;
typedef struct {
    uint32_t kind;
    union {
        trtmc_conversation_text_delta_v1 text;
        trtmc_conversation_text_delta_v1 reasoning;
        trtmc_conversation_tool_delta_v1 tool_delta;
        trtmc_conversation_tool_end_v1 tool_end;
        trtmc_i32_view token_ids;
        trtmc_conversation_result_view_v1 complete;
    } content;
} trtmc_conversation_stream_event_view_v1;
/* One assistant response. Indices identify final part positions, not input
 * messages. Tool fragments are append-only UTF-8, not complete JSON values.
 * Complete/Cancelled are terminal; poll/cancel/lifetime follow text streams. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* next)(trtmc_conversation_stream*, int64_t, trtmc_result**,
                                   trtmc_error**);
    trtmc_status(TRTMC_CALL* event_view)(const trtmc_result*,
                                         trtmc_conversation_stream_event_view_v1*, trtmc_error**);
    trtmc_status(TRTMC_CALL* cancel)(trtmc_conversation_stream*, trtmc_error**);
    void(TRTMC_CALL* release)(trtmc_conversation_stream*);
} trtmc_conversation_stream_api_v1;
#define TRTMC_TASK_STREAMING_TEXT_CONVERSATION "streaming_text_conversation"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* start)(trtmc_model*, const trtmc_text_conversation_request_v1*,
                                    const trtmc_config_view_v1*, trtmc_conversation_stream**,
                                    trtmc_error**);
    const trtmc_conversation_stream_api_v1* stream_api;
} trtmc_streaming_text_conversation_api_v1;
#define TRTMC_TASK_STREAMING_VIDEO_TEXT_TO_TEXT "streaming_video_text_to_text"
#define TRTMC_TASK_STREAMING_IMAGE_VIDEO_TEXT_TO_TEXT "streaming_image_video_text_to_text"
#define TRTMC_TASK_STREAMING_IMAGES_TEXT_CONVERSATION "streaming_images_text_conversation"
#define TRTMC_TASK_STREAMING_VIDEO_TEXT_CONVERSATION "streaming_video_text_conversation"
#define TRTMC_MEDIA_STREAM_API(Name, Request, Handle, Reader)                                      \
    typedef struct {                                                                               \
        trtmc_api_header header;                                                                   \
        trtmc_status(TRTMC_CALL* start)(trtmc_model*, const Request*, const trtmc_config_view_v1*, \
                                        Handle**, trtmc_error**);                                  \
        const Reader* stream_api;                                                                  \
    } trtmc_##Name##_api_v1;
TRTMC_MEDIA_STREAM_API(streaming_video_text_to_text, trtmc_video_text_to_text_request_v1,
                       trtmc_text_stream, trtmc_text_stream_api_v1)
TRTMC_MEDIA_STREAM_API(streaming_image_video_text_to_text,
                       trtmc_image_video_text_to_text_request_v1, trtmc_text_stream,
                       trtmc_text_stream_api_v1)
TRTMC_MEDIA_STREAM_API(streaming_images_text_conversation, trtmc_images_text_to_text_request_v1,
                       trtmc_conversation_stream, trtmc_conversation_stream_api_v1)
TRTMC_MEDIA_STREAM_API(streaming_video_text_conversation, trtmc_video_text_to_text_request_v1,
                       trtmc_conversation_stream, trtmc_conversation_stream_api_v1)
#undef TRTMC_MEDIA_STREAM_API

#ifdef __cplusplus
}
#endif
#endif /* TRTMC_STREAM_API_H */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TRTMC_SPEECH_H
#define TRTMC_SPEECH_H

#include "trtmc/audio.h"
#include "trtmc/tools.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct trtmc_asr_stream trtmc_asr_stream;
typedef struct trtmc_speech_session trtmc_speech_session;

typedef struct {
    uint32_t has_sample_rate, sample_rate, channels;
} trtmc_speech_input_format_v1;
typedef struct {
    uint32_t input_sample_rate, input_channels;
    uint32_t has_output_format, output_sample_rate, output_channels;
    uint32_t has_source_language;
    trtmc_string_view source_language;
    uint32_t has_system_prompt;
    trtmc_string_view system_prompt;
} trtmc_speech_session_info_v1;
typedef struct {
    trtmc_speech_input_format_v1 input;
    uint32_t has_source_language;
    trtmc_string_view source_language;
} trtmc_streaming_speech_transcription_request_v1;
typedef struct {
    trtmc_text_result_view_v1 transcript;
    uint32_t is_final;
    uint64_t chunk_index, accepted_samples;
    uint32_t sample_rate, channels;
} trtmc_speech_transcript_update_v1;

/* Input methods are serialized; finish/cancel do not destroy resettable
 * handles. Release must not race another call. Config/info results own data. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* create)(trtmc_model*,
                                     const trtmc_streaming_speech_transcription_request_v1*,
                                     const trtmc_config_view_v1*, trtmc_asr_stream**,
                                     trtmc_error**);
    trtmc_status(TRTMC_CALL* accept_audio)(trtmc_asr_stream*, const float*, uint64_t sample_count,
                                           uint32_t is_final, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* finish)(trtmc_asr_stream*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* update_view)(const trtmc_result*, trtmc_speech_transcript_update_v1*,
                                          trtmc_error**);
    trtmc_status(TRTMC_CALL* reset)(trtmc_asr_stream*, trtmc_error**);
    trtmc_status(TRTMC_CALL* info)(trtmc_asr_stream*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* info_view)(const trtmc_result*, trtmc_speech_session_info_v1*,
                                        trtmc_error**);
    trtmc_status(TRTMC_CALL* config)(trtmc_asr_stream*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* config_view)(const trtmc_result*, trtmc_config_view_v1*,
                                          trtmc_error**);
    void(TRTMC_CALL* release)(trtmc_asr_stream*);
} trtmc_streaming_speech_transcription_api_v1;

enum { TRTMC_AUDIO_DELIVERY_COMPLETE = 1, TRTMC_AUDIO_DELIVERY_STOPPED = 2 };
typedef struct {
    uint64_t emitted_sample_count, emitted_frame_count;
    uint32_t sample_rate, channels, outcome;
    double setup_ms, inference_ms;
} trtmc_streaming_audio_summary_v1;
typedef struct {
    trtmc_string_view error_message;
} trtmc_audio_chunk_reply_v1;
/* Called synchronously/serially only during run. Audio is borrowed for this
 * callback. Return OK to continue, END for a normal stop, or an error status.
 * No exception crosses C. Error-message storage must survive callback return
 * until the bridge copies it; do not point into destroyed callback locals. */
typedef trtmc_status(TRTMC_CALL* trtmc_audio_chunk_callback_v1)(void*, const trtmc_audio_view_v1*,
                                                                trtmc_audio_chunk_reply_v1*);
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_to_speech_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_audio_chunk_callback_v1,
                                  void* context, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* summary_view)(const trtmc_result*, trtmc_streaming_audio_summary_v1*,
                                           trtmc_error**);
} trtmc_streaming_text_to_speech_api_v1;

typedef struct {
    trtmc_speech_input_format_v1 input;
    uint32_t has_system_prompt;
    trtmc_string_view system_prompt;
} trtmc_speech_dialogue_request_v1;
typedef struct {
    trtmc_string_view tool_name;
    trtmc_strings_view messages;
} trtmc_tool_acknowledgement_v1;
typedef struct {
    trtmc_speech_dialogue_request_v1 dialogue;
    const trtmc_tool_definition_v1* tools;
    uint64_t tool_count;
    const trtmc_tool_acknowledgement_v1* acknowledgements;
    uint64_t acknowledgement_count;
    uint32_t has_default_acknowledgements;
    trtmc_strings_view default_acknowledgements;
} trtmc_tool_speech_dialogue_request_v1;

enum {
    TRTMC_SPEECH_AGENT_AUDIO = 1,
    TRTMC_SPEECH_AGENT_TEXT,
    TRTMC_SPEECH_USER_TRANSCRIPT,
    TRTMC_SPEECH_TURN_STARTED,
    TRTMC_SPEECH_TURN_FINISHED,
    TRTMC_SPEECH_YIELDED,
    TRTMC_SPEECH_CANCELLED,
    TRTMC_SPEECH_RESET,
    TRTMC_SPEECH_ERROR,
    TRTMC_SPEECH_INPUT_FINISHED,
    TRTMC_SPEECH_USER_STARTED,
    TRTMC_SPEECH_USER_STOPPED,
    TRTMC_SPEECH_FUNCTION_CALL,
    TRTMC_SPEECH_FUNCTION_CALL_STARTED,
    TRTMC_SPEECH_FUNCTION_RESPONSE_FINISHED,
    TRTMC_SPEECH_INPUT_CLEARED
};
enum { TRTMC_SPEECH_ACTIVE = 1, TRTMC_SPEECH_EPOCH_ENDED = 2, TRTMC_SPEECH_FAILED = 3 };
typedef struct {
    uint32_t kind;
    uint64_t epoch, sequence;
    trtmc_audio_view_v1 audio;
    /* -1 means unavailable. Media positions are per-channel sample frames at
     * audio.sample_rate. frame_index is the family's logical model timeline. */
    int64_t media_start_sample, media_end_sample, frame_index;
    trtmc_string_view text;
    uint32_t is_final, has_tool_call;
    trtmc_tool_call_v1 tool_call;
} trtmc_speech_event_view_v1;
typedef struct {
    const trtmc_speech_event_view_v1* events;
    uint64_t count;
    uint32_t state;
} trtmc_speech_event_batch_view_v1;

typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* commit_input_turn)(trtmc_speech_session*, uint32_t create_response,
                                                trtmc_error**);
    trtmc_status(TRTMC_CALL* create_response)(trtmc_speech_session*, trtmc_error**);
    trtmc_status(TRTMC_CALL* clear_pending_input)(trtmc_speech_session*, trtmc_error**);
    trtmc_status(TRTMC_CALL* cancel_response)(trtmc_speech_session*, trtmc_error**);
    trtmc_status(TRTMC_CALL* truncate_response)(trtmc_speech_session*, uint64_t epoch,
                                                int64_t played_output_samples, trtmc_error**);
} trtmc_speech_realtime_api_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* submit_tool_result)(trtmc_speech_session*, uint64_t epoch,
                                                 const trtmc_tool_result_v1*, trtmc_error**);
} trtmc_speech_tools_api_v1;

/* One event reader at a time; competing reads return BUSY. Append/controls may
 * run while read_events waits. AGAIN means timeout (read) or no input accepted
 * due to backpressure (append). END is the current epoch drained; reset remains
 * valid. Error/turn/is_final markers alone do not close the session. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* append_audio)(trtmc_speech_session*, const float*,
                                           uint64_t sample_count, trtmc_error**);
    trtmc_status(TRTMC_CALL* finish_input)(trtmc_speech_session*, trtmc_error**);
    trtmc_status(TRTMC_CALL* read_events)(trtmc_speech_session*, int64_t timeout_ms, trtmc_result**,
                                          trtmc_error**);
    trtmc_status(TRTMC_CALL* events_view)(const trtmc_result*, trtmc_speech_event_batch_view_v1*,
                                          trtmc_error**);
    trtmc_status(TRTMC_CALL* cancel)(trtmc_speech_session*, trtmc_error**);
    trtmc_status(TRTMC_CALL* reset)(trtmc_speech_session*, trtmc_error**);
    trtmc_status(TRTMC_CALL* info)(trtmc_speech_session*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* info_view)(const trtmc_result*, trtmc_speech_session_info_v1*,
                                        trtmc_error**);
    trtmc_status(TRTMC_CALL* config)(trtmc_speech_session*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* config_view)(const trtmc_result*, trtmc_config_view_v1*,
                                          trtmc_error**);
    trtmc_status(TRTMC_CALL* get_realtime_api)(trtmc_speech_session*, uint32_t major,
                                               uint32_t minor, const trtmc_speech_realtime_api_v1**,
                                               trtmc_error**);
    trtmc_status(TRTMC_CALL* get_tools_api)(trtmc_speech_session*, uint32_t major, uint32_t minor,
                                            const trtmc_speech_tools_api_v1**, trtmc_error**);
    /* No concurrent call may use a handle being released. Family destruction
     * stops/joins work before the model's logical execution owner is released. */
    void(TRTMC_CALL* release)(trtmc_speech_session*);
} trtmc_speech_session_api_v1;

#define TRTMC_TASK_STREAMING_SPEECH_TRANSCRIPTION "streaming_speech_transcription"
#define TRTMC_TASK_STREAMING_TEXT_TO_SPEECH "streaming_text_to_speech"
#define TRTMC_TASK_DUPLEX_SPEECH_DIALOGUE "duplex_speech_dialogue"
#define TRTMC_TASK_OFFLINE_SPEECH_DIALOGUE "offline_speech_dialogue"
#define TRTMC_TASK_TOOL_SPEECH_DIALOGUE "tool_speech_dialogue"

typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* create)(trtmc_model*, const trtmc_speech_dialogue_request_v1*,
                                     const trtmc_config_view_v1*, trtmc_speech_session**,
                                     trtmc_error**);
    const trtmc_speech_session_api_v1* session_api;
} trtmc_duplex_speech_dialogue_api_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* create)(trtmc_model*, const trtmc_speech_dialogue_request_v1*,
                                     const trtmc_config_view_v1*, trtmc_speech_session**,
                                     trtmc_error**);
    const trtmc_speech_session_api_v1* session_api;
} trtmc_offline_speech_dialogue_api_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* create)(trtmc_model*, const trtmc_tool_speech_dialogue_request_v1*,
                                     const trtmc_config_view_v1*, trtmc_speech_session**,
                                     trtmc_error**);
    const trtmc_speech_session_api_v1* session_api;
} trtmc_tool_speech_dialogue_api_v1;

#ifdef __cplusplus
}
#endif
#endif /* TRTMC_SPEECH_H */

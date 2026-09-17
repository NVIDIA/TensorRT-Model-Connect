/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TRTMC_AUDIO_H
#define TRTMC_AUDIO_H

#include "trtmc/scores.h"
#include "trtmc/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Interleaved host PCM float32; the C boundary never resamples or downmixes.
 * sample_count counts scalar values. Audio frames = sample_count / channels.
 * Channels must be positive and frames complete. An omitted input sample rate
 * uses only a default declared by the loaded family; otherwise the family
 * rejects omission. Present sample rates must be positive. Outputs always
 * contain a present, actual sample rate. */
typedef struct {
    const float* samples;
    uint64_t sample_count;
    uint32_t has_sample_rate; /* Exactly zero or one. */
    uint32_t sample_rate;
    uint32_t channels;
} trtmc_audio_view_v1;

typedef struct {
    trtmc_audio_view_v1 audio;
    double setup_ms;
    double inference_ms;
} trtmc_audio_result_view_v1;

#define TRTMC_TASK_TEXT_TO_AUDIO "text_to_audio"
#define TRTMC_TASK_TEXT_AUDIO_TOKEN_HISTORY_TO_AUDIO "text_audio_token_history_to_audio"
#define TRTMC_TASK_BATCH_TEXT_AUDIO_TOKEN_HISTORY_TO_AUDIO "batch_text_audio_token_history_to_audio"
#define TRTMC_TASK_TEXT_TO_SPEECH "text_to_speech"
#define TRTMC_TASK_SPEECH_TRANSCRIPTION "speech_transcription"
#define TRTMC_TASK_SPEECH_TRANSLATION "speech_translation"
#define TRTMC_TASK_AUDIO_LANGUAGE_IDENTIFICATION "audio_language_identification"
#define TRTMC_TASK_SPEECH_TO_SPEECH_RESPONSE "speech_to_speech_response"
#define TRTMC_TASK_BATCH_SPEECH_TRANSCRIPTION "batch_speech_transcription"
#define TRTMC_TASK_BATCH_SPEECH_TRANSLATION "batch_speech_translation"
#define TRTMC_TASK_MIXED_BATCH_SPEECH_TO_TEXT "mixed_batch_speech_to_text"
#define TRTMC_TASK_BATCH_TEXT_TO_AUDIO "batch_text_to_audio"
#define TRTMC_TASK_BATCH_TEXT_TO_SPEECH "batch_text_to_speech"

typedef struct {
    trtmc_string_view prompt;
} trtmc_text_to_audio_request_v1;
/* Borrowed integer history, not PCM/embeddings. Acoustic tokens are contiguous
 * [codebook, frame], with independent coarse/fine frame counts. No shared
 * codebook constants, vocabulary conversion, alignment or preset-file I/O. */
typedef struct {
    trtmc_i32_view tokens;
    uint64_t codebooks, frames;
} trtmc_audio_codebook_tokens_view_v1;
typedef struct {
    trtmc_i32_view semantic_tokens;
    trtmc_audio_codebook_tokens_view_v1 coarse_tokens, fine_tokens;
} trtmc_semantic_acoustic_history_view_v1;
typedef struct {
    trtmc_string_view prompt;
    trtmc_semantic_acoustic_history_view_v1 history;
} trtmc_text_audio_token_history_to_audio_request_v1;
typedef struct {
    trtmc_string_view text;
    uint32_t has_language;
    trtmc_string_view language;
} trtmc_text_to_speech_request_v1;
typedef struct {
    trtmc_audio_view_v1 audio;
    uint32_t has_source_language;
    trtmc_string_view source_language;
} trtmc_speech_transcription_request_v1;
typedef struct {
    trtmc_audio_view_v1 audio;
    uint32_t has_target_language;
    trtmc_string_view target_language;
    uint32_t has_source_language;
    trtmc_string_view source_language;
} trtmc_speech_translation_request_v1;
typedef struct {
    trtmc_audio_view_v1 audio;
} trtmc_audio_language_identification_request_v1;
typedef struct {
    trtmc_audio_view_v1 audio;
} trtmc_speech_to_speech_response_request_v1;

/* Language presence fields are exactly zero or one. Present identifiers must
 * be nonempty. Absence uses only a default declared by the loaded family;
 * a family without an applicable default rejects absence. */

typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_to_audio_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_audio_result_view_v1*,
                                          trtmc_error**);
} trtmc_text_to_audio_api_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_text_audio_token_history_to_audio_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_audio_result_view_v1*,
                                          trtmc_error**);
} trtmc_text_audio_token_history_to_audio_api_v1;

/* Unlike TextToAudio, this Task guarantees speech realizing the transcript. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_to_speech_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_audio_result_view_v1*,
                                          trtmc_error**);
} trtmc_text_to_speech_api_v1;

/* Same-language transcription, preserving available tokens/segments/timings.
 * Chunk/segment timestamps do not imply word-level alignment. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_speech_transcription_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_speech_transcription_api_v1;

typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_speech_translation_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_text_result_view_v1*,
                                          trtmc_error**);
} trtmc_speech_translation_api_v1;

/* One actual language label per score; score kind is explicit. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_audio_language_identification_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_label_scores_view_v1*,
                                          trtmc_error**);
} trtmc_audio_language_identification_api_v1;

/* Response speech to the complete input utterance, not generic audio editing. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_speech_to_speech_response_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_audio_result_view_v1*,
                                          trtmc_error**);
} trtmc_speech_to_speech_response_api_v1;

typedef struct {
    trtmc_speech_transcription_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_speech_transcription_item_v1;
typedef struct {
    const trtmc_batch_speech_transcription_item_v1* items;
    uint64_t count;
} trtmc_batch_speech_transcription_request_v1;
typedef struct {
    trtmc_speech_translation_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_speech_translation_item_v1;
typedef struct {
    const trtmc_batch_speech_translation_item_v1* items;
    uint64_t count;
} trtmc_batch_speech_translation_request_v1;

/* Native batches prevalidate all items. Success preserves order/cardinality;
 * failure returns no partial result. Each item owns its independent config. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run_batch)(trtmc_model*,
                                        const trtmc_batch_speech_transcription_request_v1*,
                                        trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_text_result_view_v1*, trtmc_error**);
} trtmc_batch_speech_transcription_api_v1;

typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run_batch)(trtmc_model*,
                                        const trtmc_batch_speech_translation_request_v1*,
                                        trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_text_result_view_v1*, trtmc_error**);
} trtmc_batch_speech_translation_api_v1;

enum { TRTMC_SPEECH_TEXT_TRANSCRIPTION = 1, TRTMC_SPEECH_TEXT_TRANSLATION = 2 };
typedef struct {
    uint32_t kind;
    union {
        trtmc_speech_transcription_request_v1 transcription;
        trtmc_speech_translation_request_v1 translation;
    } input;
    trtmc_config_view_v1 config;
} trtmc_mixed_batch_speech_to_text_item_v1;
typedef struct {
    const trtmc_mixed_batch_speech_to_text_item_v1* items;
    uint64_t count;
} trtmc_mixed_batch_speech_to_text_request_v1;

/* One native execution contract for a closed mixture of transcribe/translate
 * inputs. The C layer forwards the complete typed batch to one family call.
 * Success preserves item order/cardinality; failure returns no partial result. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run_batch)(trtmc_model*,
                                        const trtmc_mixed_batch_speech_to_text_request_v1*,
                                        trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_text_result_view_v1*, trtmc_error**);
} trtmc_mixed_batch_speech_to_text_api_v1;

typedef struct {
    trtmc_text_to_audio_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_text_to_audio_item_v1;
typedef struct {
    trtmc_text_to_speech_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_text_to_speech_item_v1;
typedef struct {
    const trtmc_batch_text_to_audio_item_v1* items;
    uint64_t count;
} trtmc_batch_text_to_audio_request_v1;
typedef struct {
    trtmc_semantic_acoustic_history_view_v1 history;
    const trtmc_batch_text_to_audio_item_v1* items;
    uint64_t count;
} trtmc_batch_text_audio_token_history_to_audio_request_v1;
typedef struct {
    const trtmc_batch_text_to_speech_item_v1* items;
    uint64_t count;
} trtmc_batch_text_to_speech_request_v1;

/* Nonempty native batches; one family invocation, never shared grouping or
 * serial synthesis. The family prevalidates all items and batch restrictions.
 * Success returns exactly count ordered results; failure returns no partial
 * result. Each item view borrows the result owner, not the model/input, and
 * exposes actual PCM length/rate/channels without batch padding. Independent
 * Config values do not imply every heterogeneous combination is supported. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run_batch)(trtmc_model*, const trtmc_batch_text_to_audio_request_v1*,
                                        trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_audio_result_view_v1*, trtmc_error**);
} trtmc_batch_text_to_audio_api_v1;
/* One shared history plus nonempty independent prompt/config items. Same
 * result ordering/ownership as BatchTextToAudio; no per-item history inference,
 * preset loading, shared scheduling, or partial success. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run_batch)(
        trtmc_model*, const trtmc_batch_text_audio_token_history_to_audio_request_v1*,
        trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_audio_result_view_v1*, trtmc_error**);
} trtmc_batch_text_audio_token_history_to_audio_api_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run_batch)(trtmc_model*, const trtmc_batch_text_to_speech_request_v1*,
                                        trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_audio_result_view_v1*, trtmc_error**);
} trtmc_batch_text_to_speech_api_v1;

#ifdef __cplusplus
}
#endif

#endif /* TRTMC_AUDIO_H */

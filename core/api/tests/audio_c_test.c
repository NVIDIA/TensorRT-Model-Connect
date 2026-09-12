/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/audio.h"
#include "trtmc/trtmc.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

_Static_assert(offsetof(trtmc_speech_transcription_api_v1, header) == 0, "first member header");
_Static_assert(offsetof(trtmc_batch_speech_translation_api_v1, header) == 0, "first member header");

static int failures;
static const trtmc_core_api_v1* core;
static trtmc_string_view str(const char* text) {
    trtmc_string_view result = {text, strlen(text)};
    return result;
}
static void check(int passed, const char* label) {
    if (!passed) {
        fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}
static void status_is(trtmc_status status, trtmc_status expected, trtmc_error* error,
                      const char* label) {
    check(status == expected, label);
    core->error_release(error);
}
static int write_bundle(const char* path, const char* mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    char header[512];
    const int length =
        snprintf(header, sizeof(header),
                 "{\"format\":1,\"family\":\"audio_fixture\",\"task\":\"%s\",\"backend\":\"fake\","
                 "\"sections\":{\"engine.plan\":{\"offset\":0,\"length\":4}}}",
                 mode);
    if (length < 0 || (size_t)length >= sizeof(header))
        return 0;
    FILE* file = fopen(path, "wb");
    if (!file)
        return 0;
    fwrite(magic, 1, sizeof(magic), file);
    for (unsigned shift = 0; shift < 64; shift += 8)
        fputc((int)(((uint64_t)length >> shift) & 255U), file);
    fwrite(header, 1, (size_t)length, file);
    fwrite("PLAN", 1, 4, file);
    const int good = !ferror(file);
    return fclose(file) == 0 && good;
}
static const trtmc_api_header* table(trtmc_model* model, const char* task) {
    const trtmc_api_header* result = NULL;
    trtmc_error* error = NULL;
    trtmc_status status = core->model_get_task_api(model, str(task), 1, 0, &result, &error);
    status_is(status, TRTMC_OK, error, "C audio Task lookup succeeds");
    return result;
}

static void test_generated_batches(trtmc_model* model) {
    const trtmc_batch_text_to_audio_api_v1* audio =
        (const trtmc_batch_text_to_audio_api_v1*)table(model, TRTMC_TASK_BATCH_TEXT_TO_AUDIO);
    const trtmc_batch_text_to_speech_api_v1* speech =
        (const trtmc_batch_text_to_speech_api_v1*)table(model, TRTMC_TASK_BATCH_TEXT_TO_SPEECH);
    if (!audio || !speech)
        return;
    trtmc_error* error = NULL;
    trtmc_result* result = NULL;
    trtmc_audio_result_view_v1 first = {0}, second = {0};
    uint64_t count = 0;
    trtmc_config_entry_v1 gain = {{"gain", 4}, {TRTMC_CONFIG_F64, {.f64 = 0.5}}};
    trtmc_batch_text_to_audio_item_v1 audio_items[] = {{{str("wind")}, {NULL, 0}},
                                                       {{str("rain now")}, {&gain, 1}}};
    const trtmc_batch_text_to_audio_request_v1 audio_request = {audio_items, 2};
    trtmc_status status = audio->run_batch(model, &audio_request, &result, &error);
    status_is(status, TRTMC_OK, error, "C native audio generation batch executes");
    status = audio->result_count(result, &count, &error);
    status_is(status, TRTMC_OK, error, "C generated audio batch count is readable");
    status = audio->result_item_view(result, 0, &first, &error);
    status_is(status, TRTMC_OK, error, "C first audio item is readable");
    status = audio->result_item_view(result, 1, &second, &error);
    status_is(status, TRTMC_OK, error, "C second audio item is readable");
    check(count == 2 && first.audio.sample_count == 4 && second.audio.sample_count == 10 &&
              first.audio.channels == 1 && second.audio.channels == 2 &&
              first.audio.sample_rate == 24000 && second.audio.sample_rate == 16000 &&
              first.audio.samples[0] == 0.04F && second.audio.samples[0] == 0.04F &&
              first.setup_ms == 1 && second.inference_ms == 0,
          "C audio batch preserves unpadded PCM/rate/channels/config with zero single calls");
    core->result_release(result);
    result = NULL;
    trtmc_config_entry_v1 speech_options[] = {
        {{"speaker", 7}, {TRTMC_CONFIG_I64, {.i64 = 1}}},
        {{"normalize", 9}, {TRTMC_CONFIG_BOOL, {.boolean = 0}}}};
    trtmc_batch_text_to_speech_item_v1 speech_items[] = {
        {{str("Hello"), 0, {NULL, 0}}, {NULL, 0}},
        {{str("Bonjour"), 1, str("fr")}, {speech_options, 2}}};
    const trtmc_batch_text_to_speech_request_v1 speech_request = {speech_items, 2};
    status = speech->run_batch(model, &speech_request, &result, &error);
    status_is(status, TRTMC_OK, error, "C native speech generation batch executes");
    status = speech->result_item_view(result, 0, &first, &error);
    status_is(status, TRTMC_OK, error, "C default-language speech item is readable");
    status = speech->result_item_view(result, 1, &second, &error);
    status_is(status, TRTMC_OK, error, "C explicit-language speech item is readable");
    check(first.audio.sample_count == 4 && second.audio.sample_count == 10 &&
              first.audio.samples[2] == 0.1F && second.audio.samples[1] == 0.1F &&
              second.audio.samples[2] == 0.2F && second.audio.samples[3] == 0 &&
              second.setup_ms == 1 && second.inference_ms == 0,
          "C batch speech keeps item language/speaker/false and family defaults");
    core->result_release(result);
}

static void test_generated_batch_errors(trtmc_model* model, trtmc_model* restricted) {
    const trtmc_batch_text_to_audio_api_v1* audio =
        (const trtmc_batch_text_to_audio_api_v1*)table(model, TRTMC_TASK_BATCH_TEXT_TO_AUDIO);
    const trtmc_batch_text_to_speech_api_v1* speech =
        (const trtmc_batch_text_to_speech_api_v1*)table(model, TRTMC_TASK_BATCH_TEXT_TO_SPEECH);
    const trtmc_text_to_audio_api_v1* single =
        (const trtmc_text_to_audio_api_v1*)table(model, TRTMC_TASK_TEXT_TO_AUDIO);
    if (!audio || !speech || !single)
        return;
    trtmc_batch_text_to_audio_item_v1 audio_items[] = {{{str("a")}, {NULL, 0}},
                                                       {{str("b")}, {NULL, 0}}};
    trtmc_batch_text_to_audio_request_v1 input = {audio_items, 2};
    trtmc_batch_text_to_speech_item_v1 speech_items[] = {{{str("a"), 0, {NULL, 0}}, {NULL, 0}},
                                                         {{str("b"), 0, {NULL, 0}}, {NULL, 0}}};
    trtmc_batch_text_to_speech_request_v1 speech_input = {speech_items, 2};
    trtmc_error* error = NULL;
    trtmc_result* result = NULL;
    trtmc_status status;
    trtmc_audio_result_view_v1 view = {0};
    uint64_t count = 99;
#define SYNTH_STATUS(call, expected, label)                                                        \
    do {                                                                                           \
        status = (call);                                                                           \
        status_is(status, (expected), error, (label));                                             \
        error = NULL;                                                                              \
    } while (0)
    SYNTH_STATUS(audio->run_batch(model, &input, &result, &error), TRTMC_OK,
                 "audio batch available for view validation");
    SYNTH_STATUS(audio->result_item_view(result, 2, &view, &error), TRTMC_INVALID_ARGUMENT,
                 "out-of-range batch item rejected");
    check(view.audio.samples == NULL && view.audio.sample_count == 0,
          "failed item view is cleared");
    SYNTH_STATUS(single->result_view(result, &view, &error), TRTMC_INVALID_ARGUMENT,
                 "batch audio does not masquerade as one scalar result");
    SYNTH_STATUS(audio->result_item_view(result, 0, NULL, &error), TRTMC_INVALID_ARGUMENT,
                 "NULL item-view output rejected");
    core->result_release(result);
    result = NULL;
    trtmc_text_to_audio_request_v1 prompt = {str("one")};
    SYNTH_STATUS(single->run(model, &prompt, NULL, &result, &error), TRTMC_OK,
                 "single result available for wrong-owner validation");
    SYNTH_STATUS(audio->result_count(result, &count, &error), TRTMC_INVALID_ARGUMENT,
                 "single audio is not a batch result");
    check(count == 0, "failed batch count is cleared");
    core->result_release(result);
    result = (trtmc_result*)(uintptr_t)1;
    SYNTH_STATUS(audio->run_batch(restricted, &input, &result, &error), TRTMC_UNSUPPORTED,
                 "model A audio batch table cannot bypass model B support");
    check(result == NULL, "unsupported model clears output instead of preserving stale data");
    SYNTH_STATUS(speech->run_batch(restricted, &speech_input, &result, &error), TRTMC_UNSUPPORTED,
                 "speech batch support is checked on every submitted model");
    SYNTH_STATUS(audio->run_batch(model, NULL, &result, &error), TRTMC_INVALID_ARGUMENT,
                 "NULL synthesis request rejected");
    SYNTH_STATUS(audio->run_batch(model, &input, NULL, &error), TRTMC_INVALID_ARGUMENT,
                 "NULL synthesis result output rejected");
    input.items = NULL;
    SYNTH_STATUS(audio->run_batch(model, &input, &result, &error), TRTMC_INVALID_ARGUMENT,
                 "NULL positive-count synthesis items rejected");
    input.items = audio_items;
    input.count = UINT64_MAX;
    SYNTH_STATUS(audio->run_batch(model, &input, &result, &error), TRTMC_INVALID_ARGUMENT,
                 "overflowed synthesis count rejected before dereference");
    input.count = 0;
    SYNTH_STATUS(audio->run_batch(model, &input, &result, &error), TRTMC_INVALID_ARGUMENT,
                 "empty synthesis submission rejected");
    input.count = 2;
    audio_items[1].input.prompt = (trtmc_string_view){NULL, 1};
    SYNTH_STATUS(audio->run_batch(model, &input, &result, &error), TRTMC_INVALID_ARGUMENT,
                 "NULL positive-length item text rejected");
    audio_items[1].input.prompt = str("b");
    speech_items[1].input.has_language = 2;
    SYNTH_STATUS(speech->run_batch(model, &speech_input, &result, &error), TRTMC_INVALID_ARGUMENT,
                 "invalid language presence encoding rejected");
    speech_items[1].input.has_language = 1;
    speech_items[1].input.language = str("");
    SYNTH_STATUS(speech->run_batch(model, &speech_input, &result, &error), TRTMC_INVALID_ARGUMENT,
                 "explicit empty synthesis language rejected");
    speech_items[1].input.has_language = 0;
    trtmc_config_entry_v1 bad = {{"normalize", 9}, {TRTMC_CONFIG_BOOL, {.boolean = 2}}};
    speech_items[1].config = (trtmc_config_view_v1){&bad, 1};
    SYNTH_STATUS(speech->run_batch(model, &speech_input, &result, &error), TRTMC_INVALID_ARGUMENT,
                 "invalid item bool encoding rejected before family execution");
    bad.value.kind = 999;
    SYNTH_STATUS(speech->run_batch(model, &speech_input, &result, &error), TRTMC_INVALID_ARGUMENT,
                 "unknown item wire kind rejected");
    bad.name = str("unknown");
    bad.value.kind = TRTMC_CONFIG_I64;
    bad.value.as.i64 = 0;
    SYNTH_STATUS(speech->run_batch(model, &speech_input, &result, &error), TRTMC_INVALID_CONFIG,
                 "unknown item key is a family config failure");
    check(result == NULL, "invalid later item returns no partial audio result");
#undef SYNTH_STATUS
}

static void test_audio(trtmc_model* model, trtmc_model* restricted) {
    const trtmc_text_to_audio_api_v1* generation =
        (const trtmc_text_to_audio_api_v1*)table(model, TRTMC_TASK_TEXT_TO_AUDIO);
    const trtmc_speech_transcription_api_v1* asr =
        (const trtmc_speech_transcription_api_v1*)table(model, TRTMC_TASK_SPEECH_TRANSCRIPTION);
    const trtmc_speech_translation_api_v1* translation =
        (const trtmc_speech_translation_api_v1*)table(model, TRTMC_TASK_SPEECH_TRANSLATION);
    const trtmc_audio_language_identification_api_v1* language =
        (const trtmc_audio_language_identification_api_v1*)table(
            model, TRTMC_TASK_AUDIO_LANGUAGE_IDENTIFICATION);
    const trtmc_batch_speech_transcription_api_v1* batch =
        (const trtmc_batch_speech_transcription_api_v1*)table(
            model, TRTMC_TASK_BATCH_SPEECH_TRANSCRIPTION);
    const trtmc_mixed_batch_speech_to_text_api_v1* mixed =
        (const trtmc_mixed_batch_speech_to_text_api_v1*)table(
            model, TRTMC_TASK_MIXED_BATCH_SPEECH_TO_TEXT);
    if (!generation || !asr || !translation || !language || !batch || !mixed)
        return;
    trtmc_error* error = NULL;
    trtmc_result* result = NULL;
    trtmc_text_to_audio_request_v1 prompt = {str("wind")};
    trtmc_status status = generation->run(model, &prompt, NULL, &result, &error);
    status_is(status, TRTMC_OK, error, "C generation succeeds with default config");
    trtmc_audio_result_view_v1 output = {0};
    status = generation->result_view(result, &output, &error);
    status_is(status, TRTMC_OK, error, "C PCM result view succeeds");
    check(output.audio.has_sample_rate == 1 && output.audio.sample_rate == 24000 &&
              output.audio.channels == 2 && output.audio.sample_count == 4,
          "PCM output explicitly preserves stereo layout");
    trtmc_text_result_view_v1 wrong = {0};
    status = asr->result_view(result, &wrong, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error,
              "audio result cannot masquerade as transcript");
    core->result_release(result);

    const float pcm[] = {0.1F, 0.2F, 0.3F, 0.4F};
    trtmc_speech_transcription_request_v1 request = {0};
    request.audio.samples = pcm;
    request.audio.sample_count = 4;
    request.audio.channels = 2;
    status = asr->run(model, &request, NULL, &result, &error);
    status_is(status, TRTMC_OK, error, "omitted C sample rate reaches family default");
    core->result_release(result);
    request.audio.has_sample_rate = 1;
    status = asr->run(model, &request, NULL, &result, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "zero PCM sample rate rejected");
    check(result == NULL, "failed audio call clears result");
    request.audio.has_sample_rate = 2;
    status = asr->run(model, &request, NULL, &result, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "invalid PCM rate presence rejected");
    request.audio.has_sample_rate = 1;
    request.audio.sample_rate = 16000;
    request.audio.channels = 0;
    status = asr->run(model, &request, NULL, &result, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "zero PCM channels rejected");
    request.audio.channels = 3;
    status = asr->run(model, &request, NULL, &result, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "partial interleaved frame rejected");
    request.audio.channels = 2;
    request.audio.sample_count = UINT64_MAX;
    status = asr->run(model, &request, NULL, &result, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "audio sample count overflow rejected");
    request.audio.sample_count = 4;
    request.audio.samples = NULL;
    status = asr->run(model, &request, NULL, &result, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "nonempty null PCM buffer rejected");
    request.audio.samples = pcm;
    request.has_source_language = 1;
    status = asr->run(model, &request, NULL, &result, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "empty present source language rejected");
    request.has_source_language = 0;
    status = asr->run(model, &request, NULL, &result, &error);
    status_is(status, TRTMC_OK, error, "valid stereo ASR request succeeds");
    trtmc_text_result_view_v1 transcript = {0};
    status = asr->result_view(result, &transcript, &error);
    status_is(status, TRTMC_OK, error, "ASR result retains typed text view");
    check(transcript.segment_count == 1 && transcript.token_ids.data[1] == 16000 &&
              transcript.token_ids.data[2] == 2 &&
              transcript.segments[0].end_seconds == 2.0 / 16000,
          "sample frame timeline and stereo metadata reach family");
    core->result_release(result);

    trtmc_speech_translation_request_v1 translated_input = {0};
    translated_input.audio = request.audio;
    status = translation->run(restricted, &translated_input, NULL, &result, &error);
    status_is(status, TRTMC_UNSUPPORTED, error,
              "full-model translation table cannot bypass restricted model");
    translated_input.has_target_language = 1;
    status = translation->run(model, &translated_input, NULL, &result, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "empty explicit speech target rejected");

    trtmc_audio_language_identification_request_v1 language_input = {request.audio};
    status = language->run(model, &language_input, NULL, &result, &error);
    status_is(status, TRTMC_OK, error, "C language identification succeeds");
    trtmc_label_scores_view_v1 scores = {0};
    status = language->result_view(result, &scores, &error);
    status_is(status, TRTMC_OK, error, "C label scores result view succeeds");
    check(scores.count == 2 && scores.labels.size == 2 && scores.kind == TRTMC_SCORE_PROBABILITY &&
              scores.labels.data[0].size == 2 && memcmp(scores.labels.data[0].data, "en", 2) == 0,
          "language labels and score meaning are explicit");
    core->result_release(result);

    trtmc_batch_speech_transcription_item_v1 item = {0};
    item.input = request;
    trtmc_batch_speech_transcription_request_v1 batch_input = {&item, 1};
    status = batch->run_batch(model, &batch_input, &result, &error);
    status_is(status, TRTMC_OK, error, "C native ASR batch succeeds");
    uint64_t count = 0;
    status = batch->result_count(result, &count, &error);
    status_is(status, TRTMC_OK, error, "C batch count readable");
    check(count == 1, "native batch result cardinality preserved");
    core->result_release(result);

    trtmc_mixed_batch_speech_to_text_item_v1 mixed_items[2] = {0};
    mixed_items[0].kind = TRTMC_SPEECH_TEXT_TRANSCRIPTION;
    mixed_items[0].input.transcription = request;
    mixed_items[1].kind = TRTMC_SPEECH_TEXT_TRANSLATION;
    translated_input.target_language = str("de");
    translated_input.has_source_language = 1;
    translated_input.source_language = str("fr");
    mixed_items[1].input.translation = translated_input;
    trtmc_mixed_batch_speech_to_text_request_v1 mixed_input = {mixed_items, 2};
    status = mixed->run_batch(model, &mixed_input, &result, &error);
    status_is(status, TRTMC_OK, error, "C closed-variant mixed native batch succeeds");
    status = mixed->result_count(result, &count, &error);
    status_is(status, TRTMC_OK, error, "mixed result count readable");
    check(count == 2, "mixed batch preserves cardinality");
    trtmc_text_result_view_v1 mixed_view = {0};
    status = mixed->result_item_view(result, 1, &mixed_view, &error);
    status_is(status, TRTMC_OK, error, "mixed translation item view readable");
    const char expected_mixed[] = "mixed_translate:fr->de!";
    check(mixed_view.text.size == sizeof(expected_mixed) - 1 &&
              memcmp(mixed_view.text.data, expected_mixed, sizeof(expected_mixed) - 1) == 0,
          "mixed C item retains its translation role and languages");
    core->result_release(result);
    mixed_items[1].kind = 999;
    status = mixed->run_batch(model, &mixed_input, &result, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "unknown closed-variant speech item rejected");
    check(result == NULL, "invalid mixed input clears result");
    mixed_items[1].kind = TRTMC_SPEECH_TEXT_TRANSLATION;
    status = mixed->run_batch(restricted, &mixed_input, &result, &error);
    status_is(status, TRTMC_UNSUPPORTED, error,
              "single ASR model cannot borrow mixed native batch table");
}

static trtmc_result*
test_history(trtmc_model* model, trtmc_model* restricted,
             const trtmc_text_audio_token_history_to_audio_api_v1** retained_api) {
    const trtmc_text_audio_token_history_to_audio_api_v1* api =
        (const trtmc_text_audio_token_history_to_audio_api_v1*)table(
            model, TRTMC_TASK_TEXT_AUDIO_TOKEN_HISTORY_TO_AUDIO);
    if (!api)
        return NULL;
    *retained_api = api;
    int32_t semantic[] = {11, 12, 13}, coarse[] = {21, 22, 23, 24}, fine[24];
    for (int32_t i = 0; i < 24; ++i)
        fine[i] = 100 + i;
    const trtmc_semantic_acoustic_history_view_v1 history = {
        {semantic, 3}, {{coarse, 4}, 2, 2}, {{fine, 24}, 8, 3}};
    trtmc_text_audio_token_history_to_audio_request_v1 input = {str("hello"), history};
    trtmc_error* error = NULL;
    trtmc_result *result = NULL, *rejected = NULL;
    trtmc_status status = api->run(model, &input, NULL, &result, &error);
    status_is(status, TRTMC_OK, error, "C explicit token-history generation succeeds");
    trtmc_audio_result_view_v1 view = {0};
    status = api->result_view(result, &view, &error);
    status_is(status, TRTMC_OK, error, "C history result uses existing owned PCM storage");
    const float expected[] = {11, 13, 21, 23, 24, 100, 121, 123, 5, 1};
    check(view.audio.sample_count == 10 && view.audio.channels == 1 &&
              view.audio.sample_rate == 24000 &&
              memcmp(view.audio.samples, expected, sizeof(expected)) == 0,
          "C keeps semantic/coarse/fine roles and independent codebook/frame axes");
    const trtmc_speech_transcription_api_v1* asr =
        (const trtmc_speech_transcription_api_v1*)table(model, TRTMC_TASK_SPEECH_TRANSCRIPTION);
    const float pcm[] = {0.25F};
    const trtmc_speech_transcription_request_v1 asr_input = {{pcm, 1, 1, 16000, 1}, 0, {NULL, 0}};
    trtmc_result* text_result = NULL;
    status = asr->run(model, &asr_input, NULL, &text_result, &error);
    status_is(status, TRTMC_OK, error, "produce a non-audio result for typed-view validation");
    status = api->result_view(text_result, &view, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error,
              "history audio view rejects a transcript handle");
    check(view.audio.sample_count == 0, "wrong-kind audio view is cleared");
    core->result_release(text_result);
    input.history.coarse_tokens.frames = 3;
    status = api->run(model, &input, NULL, &rejected, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "C rejects inconsistent token-history shape");
    check(rejected == NULL, "failed history calls clear their output handle");
    input.history = history;
    input.history.coarse_tokens = (trtmc_audio_codebook_tokens_view_v1){{NULL, 0}, UINT64_MAX, 2};
    status = api->run(model, &input, NULL, &rejected, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "C rejects codebook shape product overflow");
    input.history = history;
    input.history.semantic_tokens.data = NULL;
    status = api->run(model, &input, NULL, &rejected, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "C rejects nonempty history with null data");
    input.history = history;
    input.history.fine_tokens = (trtmc_audio_codebook_tokens_view_v1){{NULL, 0}, 8, 0};
    status = api->run(model, &input, NULL, &rejected, &error);
    const trtmc_string_view message =
        error ? core->error_message(error) : (trtmc_string_view){NULL, 0};
    const char* prefix = "fixture requires nonempty";
    check(message.size >= strlen(prefix) && memcmp(message.data, prefix, strlen(prefix)) == 0,
          "valid empty representation reaches family policy rather than a shared minimum");
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "family rejects empty required history");
    input.history = history;
    trtmc_config_entry_v1 preset = {{"voice_preset", 12},
                                    {TRTMC_CONFIG_STRING, {.string = {"speaker", 7}}}};
    const trtmc_config_view_v1 config = {&preset, 1};
    status = api->run(model, &input, &config, &rejected, &error);
    status_is(status, TRTMC_INVALID_CONFIG, error, "named preset conflicts are family validation");
    status = api->run(restricted, &input, NULL, &rejected, &error);
    status_is(status, TRTMC_UNSUPPORTED, error,
              "history table cannot bypass target model declarations");
    semantic[0] = coarse[0] = fine[0] = 999;
    status = api->result_view(result, &view, &error);
    status_is(status, TRTMC_OK, error, "PCM remains valid after caller token-buffer mutation");
    check(view.audio.samples[0] == 11 && view.audio.samples[2] == 21 &&
              view.audio.samples[5] == 100,
          "history outputs do not borrow input token buffers");
    return result;
}

static trtmc_result*
test_history_batch(trtmc_model* model, trtmc_model* restricted, trtmc_result* scalar,
                   const trtmc_batch_text_audio_token_history_to_audio_api_v1** retained_api) {
    const trtmc_batch_text_audio_token_history_to_audio_api_v1* api =
        (const trtmc_batch_text_audio_token_history_to_audio_api_v1*)table(
            model, TRTMC_TASK_BATCH_TEXT_AUDIO_TOKEN_HISTORY_TO_AUDIO);
    if (!api)
        return NULL;
    *retained_api = api;
    int32_t semantic[] = {11, 12, 13}, coarse[] = {21, 22, 23, 24}, fine[24];
    for (int32_t index = 0; index < 24; ++index)
        fine[index] = 100 + index;
    const trtmc_semantic_acoustic_history_view_v1 history = {
        {semantic, 3}, {{coarse, 4}, 2, 2}, {{fine, 24}, 8, 3}};
    trtmc_config_entry_v1 overrides[] = {{{"gain", 4}, {TRTMC_CONFIG_F64, {.f64 = 0.5}}},
                                         {{"normalize", 9}, {TRTMC_CONFIG_BOOL, {.boolean = 0}}}};
    trtmc_batch_text_to_audio_item_v1 items[] = {{{str("first")}, {NULL, 0}},
                                                 {{str("second!")}, {overrides, 2}}};
    trtmc_batch_text_audio_token_history_to_audio_request_v1 input = {history, items, 2};
    trtmc_error* error = NULL;
    trtmc_result *result = NULL, *rejected = NULL;
    trtmc_status status = api->run_batch(model, &input, &result, &error);
    status_is(status, TRTMC_OK, error, "C history batch runs one typed family method");
    uint64_t count = 0;
    status = api->result_count(result, &count, &error);
    status_is(status, TRTMC_OK, error, "C history batch result count is readable");
    trtmc_audio_result_view_v1 first = {0}, second = {0};
    status = api->result_item_view(result, 0, &first, &error);
    status_is(status, TRTMC_OK, error, "first history-conditioned PCM item is readable");
    status = api->result_item_view(result, 1, &second, &error);
    status_is(status, TRTMC_OK, error, "second history-conditioned PCM item is readable");
    check(count == 2 && first.audio.sample_count == 10 && second.audio.sample_count == 11 &&
              first.audio.samples[0] == 11 && second.audio.samples[0] == 5.5F &&
              first.audio.samples[3] == 23 && second.audio.samples[6] == 60.5F &&
              first.audio.samples[8] == 5 && second.audio.samples[8] == 3.5F &&
              second.audio.samples[9] == 0 && first.setup_ms == 1 && second.setup_ms == 1 &&
              first.inference_ms == 1 && second.inference_ms == 1,
          "C batch shares history, retains item order/config/length, and never adds scalar history "
          "calls");
    status = api->result_count(scalar, &count, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error,
              "scalar audio cannot be interpreted as an audio batch");
    check(count == 0, "wrong-kind count output is cleared");
    status = api->result_item_view(result, 2, &first, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "history batch result index is checked");
    overrides[1].value.as.boolean = 2;
    status = api->run_batch(model, &input, &rejected, &error);
    const trtmc_string_view message =
        error ? core->error_message(error) : (trtmc_string_view){NULL, 0};
    const char* prefix = "batch item[1]";
    check(message.size >= strlen(prefix) && memcmp(message.data, prefix, strlen(prefix)) == 0,
          "invalid later wire config includes its batch item index");
    status_is(status, TRTMC_INVALID_ARGUMENT, error,
              "batch bool encoding is checked before execution");
    check(rejected == NULL, "failed history batch has no partial output handle");
    overrides[1].value.as.boolean = 0;
    status = api->run_batch(restricted, &input, &rejected, &error);
    status_is(status, TRTMC_UNSUPPORTED, error,
              "history batch table checks the actual target model");
    const trtmc_api_header* incompatible = (const trtmc_api_header*)1;
    status =
        core->model_get_task_api(model, str(TRTMC_TASK_BATCH_TEXT_AUDIO_TOKEN_HISTORY_TO_AUDIO), 2,
                                 0, &incompatible, &error);
    status_is(status, TRTMC_VERSION_MISMATCH, error, "history batch major version is negotiated");
    check(incompatible == NULL, "incompatible history table output is cleared");
    input.count = 0;
    status = api->run_batch(model, &input, &rejected, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "history batch requires independent items");
    semantic[0] = coarse[0] = fine[0] = 999;
    return result;
}

int main(int argc, char** argv) {
    if (argc != 2 || trtmc_get_api(1, 0, &core) != TRTMC_OK || !core)
        return 2;
    char full[4096], limited[4096];
    const int a = snprintf(full, sizeof(full), "%s/audio-c-full.bundle", argv[1]);
    const int b = snprintf(limited, sizeof(limited), "%s/audio-c-limited.bundle", argv[1]);
    if (a < 0 || (size_t)a >= sizeof(full) || b < 0 || (size_t)b >= sizeof(limited))
        return 2;
    if (!write_bundle(full, "audio_all") || !write_bundle(limited, "asr_only"))
        return 2;
    trtmc_load_options_v1 options = {0};
    options.struct_size = sizeof(options);
    options.runtime_root = str(argv[1]);
    trtmc_error* error = NULL;
    trtmc_model *model = NULL, *restricted = NULL;
    trtmc_status status = core->model_load(str(full), &options, &model, &error);
    status_is(status, TRTMC_OK, error, "full audio fixture loaded");
    status = core->model_load(str(limited), &options, &restricted, &error);
    status_is(status, TRTMC_OK, error, "restricted audio fixture loaded");
    trtmc_result* history_result = NULL;
    const trtmc_text_audio_token_history_to_audio_api_v1* history_api = NULL;
    trtmc_result* history_batch_result = NULL;
    const trtmc_batch_text_audio_token_history_to_audio_api_v1* history_batch_api = NULL;
    if (model && restricted) {
        test_generated_batches(model);
        test_generated_batch_errors(model, restricted);
        test_audio(model, restricted);
        history_result = test_history(model, restricted, &history_api);
        history_batch_result =
            test_history_batch(model, restricted, history_result, &history_batch_api);
    }
    const trtmc_batch_text_to_audio_api_v1* retained_api =
        model
            ? (const trtmc_batch_text_to_audio_api_v1*)table(model, TRTMC_TASK_BATCH_TEXT_TO_AUDIO)
            : NULL;
    trtmc_result* retained = NULL;
    if (retained_api) {
        char prompt[] = "owned";
        const trtmc_batch_text_to_audio_item_v1 item = {{str(prompt)}, {NULL, 0}};
        const trtmc_batch_text_to_audio_request_v1 input = {&item, 1};
        status = retained_api->run_batch(model, &input, &retained, &error);
        status_is(status, TRTMC_OK, error, "retained C audio batch executes");
        memset(prompt, 'X', sizeof(prompt) - 1);
    }
    core->model_release(model);
    core->model_release(restricted);
    if (history_result) {
        trtmc_audio_result_view_v1 view = {0};
        status = history_api->result_view(history_result, &view, &error);
        status_is(status, TRTMC_OK, error, "history PCM survives original model release");
        check(view.audio.sample_count == 10 && view.audio.samples[7] == 123,
              "all history-conditioned output bytes are owned");
        core->result_release(history_result);
    }
    if (history_batch_result) {
        trtmc_audio_result_view_v1 view = {0};
        status = history_batch_api->result_item_view(history_batch_result, 1, &view, &error);
        status_is(status, TRTMC_OK, error, "history batch survives model and token-buffer release");
        check(view.audio.sample_count == 11 && view.audio.samples[5] == 50 &&
                  view.audio.samples[7] == 61.5F,
              "history-conditioned batch PCM is independent of caller storage");
        core->result_release(history_batch_result);
    }
    if (retained) {
        trtmc_audio_result_view_v1 view = {0};
        status = retained_api->result_item_view(retained, 0, &view, &error);
        status_is(status, TRTMC_OK, error, "C audio batch survives model/input release");
        check(view.audio.sample_count == 4 && view.audio.samples[0] == 0.05F,
              "retained audio batch owns its PCM independently of caller input");
        core->result_release(retained);
    }
    remove(full);
    remove(limited);
    fprintf(stderr, "%s\n", failures == 0 ? "ALL PASSED" : "SOME FAILED");
    return failures;
}

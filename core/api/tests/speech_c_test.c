/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/speech.h"
#include "trtmc/trtmc.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

_Static_assert(offsetof(trtmc_speech_session_api_v1, header) == 0, "first header");
static int failures;
static const trtmc_core_api_v1* core;
static trtmc_string_view str(const char* value) {
    trtmc_string_view result = {value, strlen(value)};
    return result;
}
static void check(int ok, const char* label) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}
static void status_is(trtmc_status status, trtmc_status expected, trtmc_error* error,
                      const char* label) {
    check(status == expected, label);
    core->error_release(error);
}
static int bundle(const char* path) {
    const char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    const char header[] = "{\"format\":1,\"family\":\"speech_fixture\",\"task\":\"ordinary\","
                          "\"backend\":\"fake\",\"sections\":{}}";
    FILE* file = fopen(path, "wb");
    if (!file)
        return 0;
    fwrite(magic, 1, sizeof(magic), file);
    for (unsigned shift = 0; shift < 64; shift += 8)
        fputc((int)(((uint64_t)(sizeof(header) - 1) >> shift) & 255U), file);
    fwrite(header, 1, sizeof(header) - 1, file);
    const int good = !ferror(file);
    return fclose(file) == 0 && good;
}
static trtmc_model* load(const char* path, const char* root) {
    trtmc_load_options_v1 options = {0};
    options.struct_size = sizeof(options);
    options.runtime_root = str(root);
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    trtmc_status status = core->model_load(str(path), &options, &model, &error);
    status_is(status, TRTMC_OK, error, "speech C model loads");
    return model;
}
static const trtmc_api_header* table(trtmc_model* model, const char* name) {
    const trtmc_api_header* out = NULL;
    trtmc_error* error = NULL;
    trtmc_status status = core->model_get_task_api(model, str(name), 1, 0, &out, &error);
    status_is(status, TRTMC_OK, error, "speech C Task table lookup");
    return out;
}

static void test_asr(const char* path, const char* root) {
    trtmc_model* model = load(path, root);
    if (!model)
        return;
    const trtmc_streaming_speech_transcription_api_v1* api =
        (const trtmc_streaming_speech_transcription_api_v1*)table(
            model, TRTMC_TASK_STREAMING_SPEECH_TRANSCRIPTION);
    if (!api) {
        core->model_release(model);
        return;
    }
    trtmc_streaming_speech_transcription_request_v1 input = {0};
    input.input.channels = 2;
    char first[] = "kept", last[] = "last";
    trtmc_string_view strings[] = {{first, 4}, {last, 4}};
    trtmc_config_entry_v1 entry = {{"strings", 7},
                                   {TRTMC_CONFIG_STRING_LIST, {.string_list = {strings, 2}}}};
    trtmc_config_view_v1 config = {&entry, 1};
    trtmc_asr_stream* stream = NULL;
    trtmc_error* error = NULL;
    trtmc_status status = api->create(model, &input, &config, &stream, &error);
    status_is(status, TRTMC_OK, error, "C ASR create copies config");
    first[0] = 'X';
    last[0] = 'X';
    trtmc_result* snapshot = NULL;
    status = api->config(stream, &snapshot, &error);
    status_is(status, TRTMC_OK, error, "effective config snapshot created");
    core->model_release(model);
    model = NULL;
    const float audio[] = {0.1F, 0.2F, 0.3F, 0.4F};
    trtmc_result* result = NULL;
    status = api->accept_audio(stream, audio, 4, 1, &result, &error);
    status_is(status, TRTMC_OK, error, "ASR works after original C model handle release");
    trtmc_speech_transcript_update_v1 update = {0};
    status = api->update_view(result, &update, &error);
    status_is(status, TRTMC_OK, error, "ASR snapshot view");
    check(update.is_final == 1 && update.accepted_samples == 2 && update.chunk_index == 1 &&
              update.sample_rate == 16000,
          "ASR final metadata retained");
    trtmc_config_view_v1 wrong = {0};
    status = api->config_view(result, &wrong, &error);
    status_is(status, TRTMC_INVALID_ARGUMENT, error, "update is not a config snapshot");
    api->release(stream);
    trtmc_config_view_v1 view = {0};
    status = api->config_view(snapshot, &view, &error);
    status_is(status, TRTMC_OK, error, "snapshot outlives session and model");
    check(view.count == 7 && view.entries[0].value.as.i64 == 0 &&
              view.entries[1].value.as.f64 == 0 && view.entries[2].value.as.boolean == 0 &&
              view.entries[3].value.as.string.size == 0,
          "fixed false zero empty values preserved");
    check(view.entries[4].value.as.i64_list.data[1] == INT64_C(9007199254740993) &&
              view.entries[5].value.as.f64_list.size == 2 &&
              view.entries[6].value.as.string_list.size == 2 &&
              memcmp(view.entries[6].value.as.string_list.data[0].data, "kept", 4) == 0,
          "all list kinds and nested bytes deep copied");
    core->result_release(snapshot);
    core->result_release(result);
}
typedef struct {
    int calls;
    int mode;
} CallbackContext;
static trtmc_status TRTMC_CALL callback(void* raw, const trtmc_audio_view_v1* audio,
                                        trtmc_audio_chunk_reply_v1* out) {
    CallbackContext* context = raw;
    ++context->calls;
    check(audio->has_sample_rate == 1 && audio->sample_rate == 24000 && audio->channels == 2,
          "callback has actual PCM layout");
    if (context->mode == 1)
        return TRTMC_END;
    if (context->mode == 2) {
        out->error_message = str("callback rejected data");
        return TRTMC_INVALID_ARGUMENT;
    }
    return TRTMC_OK;
}
static void test_tts(const char* path, const char* root) {
    trtmc_model* model = load(path, root);
    if (!model)
        return;
    const trtmc_streaming_text_to_speech_api_v1* api =
        (const trtmc_streaming_text_to_speech_api_v1*)table(model,
                                                            TRTMC_TASK_STREAMING_TEXT_TO_SPEECH);
    if (!api) {
        core->model_release(model);
        return;
    }
    trtmc_text_to_speech_request_v1 input = {0};
    input.text = str("hello");
    for (int mode = 0; mode < 3; ++mode) {
        CallbackContext context = {0, mode};
        trtmc_error* error = NULL;
        trtmc_result* result = NULL;
        trtmc_status status = api->run(model, &input, NULL, callback, &context, &result, &error);
        status_is(status, mode == 2 ? TRTMC_INVALID_ARGUMENT : TRTMC_OK, error,
                  "C callback continuation/stop/error status");
        check(context.calls == (mode == 0 ? 3 : 1), "callback count honors stop/error");
        if (result) {
            trtmc_streaming_audio_summary_v1 summary = {0};
            status = api->summary_view(result, &summary, &error);
            status_is(status, TRTMC_OK, error, "TTS summary view");
            check(summary.emitted_sample_count == (uint64_t)context.calls * 4 &&
                      summary.outcome == (mode == 0 ? TRTMC_AUDIO_DELIVERY_COMPLETE
                                                    : TRTMC_AUDIO_DELIVERY_STOPPED),
                  "typed final summary preserves delivery outcome");
        }
        core->result_release(result);
    }
    core->model_release(model);
}
static void test_offline(const char* path, const char* root) {
    trtmc_model* model = load(path, root);
    if (!model)
        return;
    const trtmc_offline_speech_dialogue_api_v1* api =
        (const trtmc_offline_speech_dialogue_api_v1*)table(model,
                                                           TRTMC_TASK_OFFLINE_SPEECH_DIALOGUE);
    if (!api) {
        core->model_release(model);
        return;
    }
    trtmc_speech_dialogue_request_v1 input = {0};
    input.input.channels = 2;
    trtmc_speech_session* session = NULL;
    trtmc_error* error = NULL;
    trtmc_status status = api->create(model, &input, NULL, &session, &error);
    status_is(status, TRTMC_OK, error, "offline C session create");
    const trtmc_speech_realtime_api_v1* control = NULL;
    status = api->session_api->get_realtime_api(session, 1, 0, &control, &error);
    status_is(status, TRTMC_UNSUPPORTED, error,
              "offline session exposes no false realtime interface");
    core->model_release(model);
    const float audio[] = {0, 0, 0.25F, 0.5F};
    status = api->session_api->append_audio(session, audio, 4, &error);
    status_is(status, TRTMC_OK, error, "offline append after model release");
    status = api->session_api->finish_input(session, &error);
    status_is(status, TRTMC_OK, error, "offline input finish");
    trtmc_result* batch = NULL;
    status = api->session_api->read_events(session, -1, &batch, &error);
    status_is(status, TRTMC_OK, error, "offline event snapshot");
    status = api->session_api->reset(session, &error);
    status_is(status, TRTMC_OK, error, "offline reset retains same handle");
    api->session_api->release(session);
    trtmc_speech_event_batch_view_v1 events = {0};
    status = api->session_api->events_view(batch, &events, &error);
    status_is(status, TRTMC_OK, error, "events outlive reset session and model");
    check(events.state == TRTMC_SPEECH_EPOCH_ENDED && events.count > 0,
          "owned epoch state retained");
    int found = 0;
    for (uint64_t i = 0; i < events.count; ++i)
        if (events.events[i].kind == TRTMC_SPEECH_AGENT_AUDIO) {
            found = events.events[i].audio.sample_count == 4 &&
                    events.events[i].audio.samples[0] == 0 &&
                    events.events[i].media_end_sample == 2;
        }
    check(found, "offline waveform and per-channel media bounds preserved");
    core->result_release(batch);
}

int main(int argc, char** argv) {
    if (argc != 2 || trtmc_get_api(1, 0, &core) != TRTMC_OK || !core)
        return 2;
    char path[4096];
    const int n = snprintf(path, sizeof(path), "%s/speech-c.bundle", argv[1]);
    if (n < 0 || (size_t)n >= sizeof(path) || !bundle(path))
        return 2;
    test_asr(path, argv[1]);
    test_tts(path, argv[1]);
    test_offline(path, argv[1]);
    remove(path);
    fprintf(stderr, "%s\n", failures ? "SOME FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}

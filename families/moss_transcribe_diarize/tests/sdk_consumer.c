/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Direct C SDK call of moss_transcribe_diarize's speech_transcription Task.
 * Prints the transcript, its ids and its segments as one JSON line. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/audio.h>
#include <trtmc/trtmc.h>

static const trtmc_core_api_v1* api = NULL;
static trtmc_error* last_error = NULL;

static void check(trtmc_status status, const char* label) {
    if (status == TRTMC_OK)
        return;
    if (api && last_error) {
        trtmc_string_view message = api->error_message(last_error);
        fprintf(stderr, "%s failed: %.*s\n", label, (int)message.size,
                message.data ? message.data : "");
        api->error_release(last_error);
    } else {
        fprintf(stderr, "%s failed\n", label);
    }
    exit(1);
}

static trtmc_string_view view_of(const char* text) {
    trtmc_string_view view;
    view.data = text;
    view.size = strlen(text);
    return view;
}

static void print_json_string(trtmc_string_view text) {
    putchar('"');
    for (uint64_t i = 0; i < text.size; ++i) {
        const unsigned char c = (unsigned char)text.data[i];
        if (c == '"' || c == '\\')
            printf("\\%c", c);
        else if (c < 0x20)
            printf("\\u%04x", c);
        else
            putchar(c);
    }
    putchar('"');
}

static float* read_pcm(const char* path, uint64_t* count) {
    FILE* file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END) != 0)
        return NULL;
    const long bytes = ftell(file);
    if (bytes <= 0 || bytes % (long)sizeof(float) != 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    float* samples = (float*)malloc((size_t)bytes);
    *count = (uint64_t)bytes / sizeof(float);
    if (!samples || fread(samples, sizeof(float), (size_t)*count, file) != *count) {
        free(samples);
        samples = NULL;
    }
    fclose(file);
    return samples;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s BUNDLE RUNTIME_ROOT PCM_F32_16KHZ_MONO MAX_NEW_TOKENS\n",
                argv[0]);
        return 2;
    }
    uint64_t count = 0;
    float* samples = read_pcm(argv[3], &count);
    if (!samples) {
        fprintf(stderr, "input must be nonempty float32 PCM\n");
        return 1;
    }
    check(trtmc_get_api(1, 0, &api), "trtmc_get_api");

    trtmc_load_options_v1 options;
    memset(&options, 0, sizeof(options));
    options.struct_size = sizeof(options);
    options.runtime_root = view_of(argv[2]);
    trtmc_model* model = NULL;
    check(api->model_load(view_of(argv[1]), &options, &model, &last_error), "model_load");

    const trtmc_api_header* header = NULL;
    check(api->model_get_task_api(model, view_of(TRTMC_TASK_SPEECH_TRANSCRIPTION), 1, 0, &header,
                                  &last_error),
          "speech_transcription task");
    const trtmc_speech_transcription_api_v1* task =
        (const trtmc_speech_transcription_api_v1*)header;

    trtmc_speech_transcription_request_v1 request;
    memset(&request, 0, sizeof(request));
    request.audio.samples = samples;
    request.audio.sample_count = count;
    request.audio.has_sample_rate = 1;
    request.audio.sample_rate = 16000;
    request.audio.channels = 1;

    trtmc_config_entry_v1 entry;
    memset(&entry, 0, sizeof(entry));
    entry.name = view_of("max_new_tokens");
    entry.value.kind = TRTMC_CONFIG_I64;
    entry.value.as.i64 = strtoll(argv[4], NULL, 10);
    trtmc_config_view_v1 config;
    config.entries = &entry;
    config.count = 1;

    trtmc_result* result = NULL;
    check(task->run(model, &request, &config, &result, &last_error), "speech_transcription run");
    trtmc_text_result_view_v1 view;
    memset(&view, 0, sizeof(view));
    check(task->result_view(result, &view, &last_error), "result_view");

    printf("{\"text\":");
    print_json_string(view.text);
    printf(",\"token_ids\":[");
    for (uint64_t i = 0; i < view.token_ids.size; ++i)
        printf("%s%d", i ? "," : "", view.token_ids.data[i]);
    printf("],\"segments\":[");
    for (uint64_t i = 0; i < view.segment_count; ++i) {
        printf("%s{\"start_seconds\":%.17g,\"end_seconds\":%.17g,\"text\":", i ? "," : "",
               view.segments[i].start_seconds, view.segments[i].end_seconds);
        print_json_string(view.segments[i].text);
        putchar('}');
    }
    printf("]}\n");

    api->result_release(result);
    api->model_release(model);
    free(samples);
    return 0;
}

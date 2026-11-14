/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C SDK call of speech_transcription: prints {"text", "token_ids"} as one JSON line.
 * Usage: consumer BUNDLE RUNTIME_ROOT PCM_F32 RATE MAX_NEW_TOKENS */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/audio.h>
#include <trtmc/trtmc.h>

static const trtmc_core_api_v1* api = NULL;

static void fail(trtmc_status status, trtmc_error* error, const char* label) {
    if (status == TRTMC_OK)
        return;
    if (api != NULL && error != NULL) {
        trtmc_string_view message = api->error_message(error);
        fprintf(stderr, "%s failed: %.*s\n", label, (int)message.size,
                message.data ? message.data : "");
        api->error_release(error);
    } else {
        fprintf(stderr, "%s failed\n", label);
    }
    exit(1);
}

static trtmc_string_view view(const char* text) {
    trtmc_string_view result;
    result.data = text;
    result.size = strlen(text);
    return result;
}

static float* read_pcm(const char* path, uint64_t* count) {
    FILE* file = fopen(path, "rb");
    long bytes;
    float* samples;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0 || (bytes = ftell(file)) <= 0 ||
        bytes % (long)sizeof(float) != 0 || fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "input must be nonempty float32 PCM\n");
        exit(1);
    }
    samples = (float*)malloc((size_t)bytes);
    if (samples == NULL || fread(samples, 1, (size_t)bytes, file) != (size_t)bytes) {
        fprintf(stderr, "incomplete PCM input\n");
        exit(1);
    }
    fclose(file);
    *count = (uint64_t)bytes / sizeof(float);
    return samples;
}

static void print_json_string(trtmc_string_view text) {
    uint64_t index;
    putchar('"');
    for (index = 0; index < text.size; ++index) {
        unsigned char c = (unsigned char)text.data[index];
        if (c == '"' || c == '\\')
            printf("\\%c", c);
        else if (c < 0x20)
            printf("\\u%04x", c);
        else
            putchar(c);
    }
    putchar('"');
}

int main(int argc, char** argv) {
    trtmc_error* error = NULL;
    trtmc_model* model = NULL;
    trtmc_load_options_v1 options;
    const trtmc_api_header* header = NULL;
    const trtmc_speech_transcription_api_v1* task;
    trtmc_speech_transcription_request_v1 request;
    trtmc_config_entry_v1 entry;
    trtmc_config_view_v1 config;
    trtmc_result* result = NULL;
    trtmc_text_result_view_v1 output;
    uint64_t count = 0, index;
    float* samples;
    long rate, limit;

    if (argc != 6) {
        fprintf(stderr, "Usage: consumer BUNDLE RUNTIME_ROOT PCM_F32 RATE MAX_NEW_TOKENS\n");
        return 2;
    }
    rate = strtol(argv[4], NULL, 10);
    limit = strtol(argv[5], NULL, 10);
    if (rate <= 0 || limit <= 0) {
        fprintf(stderr, "invalid sample rate or token limit\n");
        return 2;
    }
    samples = read_pcm(argv[3], &count);
    fail(trtmc_get_api(1, 0, &api), NULL, "trtmc_get_api");
    memset(&options, 0, sizeof(options));
    options.struct_size = sizeof(options);
    options.runtime_root = view(argv[2]);
    fail(api->model_load(view(argv[1]), &options, &model, &error), error, "model_load");
    fail(api->model_get_task_api(model, view(TRTMC_TASK_SPEECH_TRANSCRIPTION), 1, 0, &header,
                                 &error),
         error, "get speech_transcription");
    task = (const trtmc_speech_transcription_api_v1*)header;

    memset(&request, 0, sizeof(request));
    request.audio.samples = samples;
    request.audio.sample_count = count;
    request.audio.has_sample_rate = 1;
    request.audio.sample_rate = (uint32_t)rate;
    request.audio.channels = 1;
    memset(&entry, 0, sizeof(entry));
    entry.name = view("max_new_tokens");
    entry.value.kind = TRTMC_CONFIG_I64;
    entry.value.as.i64 = limit;
    config.entries = &entry;
    config.count = 1;
    fail(task->run(model, &request, &config, &result, &error), error, "speech_transcription run");
    memset(&output, 0, sizeof(output));
    fail(task->result_view(result, &output, &error), error, "speech_transcription result_view");

    printf("{\"text\":");
    print_json_string(output.text);
    printf(",\"token_ids\":[");
    for (index = 0; index < output.token_ids.size; ++index)
        printf("%s%d", index ? "," : "", (int)output.token_ids.data[index]);
    printf("]}\n");
    api->result_release(result);
    api->model_release(model);
    free(samples);
    return 0;
}

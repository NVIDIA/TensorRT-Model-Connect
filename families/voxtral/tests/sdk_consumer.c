/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal C SDK consumer for the voxtral family: speech_transcription through
 * the public C API (trtmc/trtmc.h + trtmc/audio.h). Prints one JSON line with
 * the transcript and its token ids.
 *
 * Usage: test_voxtral_sdk_c BUNDLE RUNTIME_ROOT PCM_F32 RATE CHANNELS MAX_NEW_TOKENS [LANGUAGE]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/audio.h>
#include <trtmc/trtmc.h>

static trtmc_string_view sv(const char* text) {
    trtmc_string_view view;
    view.data = text;
    view.size = strlen(text);
    return view;
}

static int fail(const trtmc_core_api_v1* api, trtmc_error* error, const char* label) {
    if (api != NULL && error != NULL) {
        trtmc_string_view message = api->error_message(error);
        fprintf(stderr, "%s failed: %.*s\n", label, (int)message.size,
                message.data != NULL ? message.data : "");
        api->error_release(error);
    } else {
        fprintf(stderr, "%s failed\n", label);
    }
    return 1;
}

static float* read_pcm(const char* path, size_t* count) {
    FILE* file = fopen(path, "rb");
    long bytes;
    float* samples;
    if (file == NULL)
        return NULL;
    if (fseek(file, 0, SEEK_END) != 0 || (bytes = ftell(file)) <= 0 ||
        bytes % (long)sizeof(float) != 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    samples = (float*)malloc((size_t)bytes);
    if (samples == NULL || fread(samples, 1, (size_t)bytes, file) != (size_t)bytes) {
        free(samples);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *count = (size_t)bytes / sizeof(float);
    return samples;
}

static void print_json_string(const char* data, size_t size) {
    size_t index;
    putchar('"');
    for (index = 0; index < size; ++index) {
        const unsigned char value = (unsigned char)data[index];
        if (value == '"' || value == '\\')
            printf("\\%c", value);
        else if (value < 0x20)
            printf("\\u%04x", value);
        else
            putchar(value);
    }
    putchar('"');
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* api = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_speech_transcription_api_v1* task;
    trtmc_error* error = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_load_options_v1 options;
    trtmc_speech_transcription_request_v1 request;
    trtmc_config_entry_v1 entry;
    trtmc_config_view_v1 config;
    trtmc_text_result_view_v1 view;
    size_t count = 0;
    float* pcm;
    uint64_t index;

    if (argc != 7 && argc != 8) {
        fprintf(stderr,
                "usage: %s BUNDLE RUNTIME_ROOT PCM_F32 RATE CHANNELS MAX_NEW_TOKENS [LANGUAGE]\n",
                argv[0]);
        return 2;
    }
    pcm = read_pcm(argv[3], &count);
    if (pcm == NULL) {
        fprintf(stderr, "cannot read float32 PCM from %s\n", argv[3]);
        return 2;
    }
    if (trtmc_get_api(1, 0, &api) != TRTMC_OK)
        return fail(NULL, NULL, "trtmc_get_api");
    memset(&options, 0, sizeof(options));
    options.struct_size = sizeof(options);
    options.runtime_root = sv(argv[2]);
    if (api->model_load(sv(argv[1]), &options, &model, &error) != TRTMC_OK)
        return fail(api, error, "model_load");
    if (api->model_get_task_api(model, sv(TRTMC_TASK_SPEECH_TRANSCRIPTION), 1, 0, &header,
                                &error) != TRTMC_OK)
        return fail(api, error, "model_get_task_api");
    task = (const trtmc_speech_transcription_api_v1*)header;

    memset(&request, 0, sizeof(request));
    request.audio.samples = pcm;
    request.audio.sample_count = count;
    request.audio.has_sample_rate = 1;
    request.audio.sample_rate = (uint32_t)strtoul(argv[4], NULL, 10);
    request.audio.channels = (uint32_t)strtoul(argv[5], NULL, 10);
    if (argc == 8) {
        request.has_source_language = 1;
        request.source_language = sv(argv[7]);
    }
    memset(&entry, 0, sizeof(entry));
    entry.name = sv("max_new_tokens");
    entry.value.kind = TRTMC_CONFIG_I64;
    entry.value.as.i64 = strtoll(argv[6], NULL, 10);
    config.entries = &entry;
    config.count = 1;
    if (task->run(model, &request, &config, &result, &error) != TRTMC_OK)
        return fail(api, error, "speech_transcription run");
    memset(&view, 0, sizeof(view));
    if (task->result_view(result, &view, &error) != TRTMC_OK)
        return fail(api, error, "speech_transcription result_view");

    printf("{\"text\": ");
    print_json_string(view.text.data, view.text.size);
    printf(", \"token_ids\": [");
    for (index = 0; index < view.token_ids.size; ++index)
        printf("%s%d", index == 0 ? "" : ", ", view.token_ids.data[index]);
    printf("]}\n");

    api->result_release(result);
    api->model_release(model);
    free(pcm);
    return 0;
}

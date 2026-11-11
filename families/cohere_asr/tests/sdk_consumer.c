/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/audio.h>
#include <trtmc/trtmc.h>
static trtmc_string_view text(const char* s) {
    trtmc_string_view v = {s, (uint64_t)strlen(s)};
    return v;
}
int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    trtmc_result** results = NULL;
    const trtmc_speech_transcription_api_v1* task = NULL;
    int status = 1, i;
    trtmc_load_options_v1 options = {0};
    if (argc < 4)
        return 2;
    if (trtmc_get_api(1, 0, &core) != TRTMC_OK || !core)
        return 1;
    options.struct_size = sizeof(options);
    options.runtime_root = text(argv[2]);
    if (core->model_load(text(argv[1]), &options, &model, &error) != TRTMC_OK)
        goto cleanup;
    if (core->model_get_task_api(model, text(TRTMC_TASK_SPEECH_TRANSCRIPTION), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    task = (const trtmc_speech_transcription_api_v1*)header;
    results = (trtmc_result**)calloc((size_t)(argc - 3), sizeof(*results));
    if (!results)
        goto cleanup;
    for (i = 3; i < argc; ++i) {
        FILE* file = fopen(argv[i], "rb");
        long bytes;
        float* audio;
        trtmc_speech_transcription_request_v1 request = {0};
        if (!file)
            goto cleanup;
        if (fseek(file, 0, SEEK_END) || (bytes = ftell(file)) <= 0 || bytes % (long)sizeof(float)) {
            fclose(file);
            goto cleanup;
        }
        rewind(file);
        audio = (float*)malloc((size_t)bytes);
        if (!audio) {
            fclose(file);
            goto cleanup;
        }
        if (fread(audio, 1, (size_t)bytes, file) != (size_t)bytes) {
            free(audio);
            fclose(file);
            goto cleanup;
        }
        fclose(file);
        request.audio.samples = audio;
        request.audio.sample_count = (uint64_t)bytes / sizeof(float);
        request.audio.has_sample_rate = 1;
        request.audio.sample_rate = 16000;
        request.audio.channels = 1;
        request.has_source_language = 1;
        request.source_language = text("ar");
        if (task->run(model, &request, NULL, &results[i - 3], &error) != TRTMC_OK) {
            free(audio);
            goto cleanup;
        }
        free(audio);
    }
    core->model_release(model);
    model = NULL;
    for (i = 0; i < argc - 3; ++i) {
        trtmc_text_result_view_v1 view = {0};
        uint64_t j;
        if (task->result_view(results[i], &view, &error) != TRTMC_OK)
            goto cleanup;
        fputs("{\"text_hex\":\"", stdout);
        for (j = 0; j < view.text.size; ++j)
            printf("%02x", (unsigned char)view.text.data[j]);
        fputs("\",\"token_ids\":[", stdout);
        for (j = 0; j < view.token_ids.size; ++j)
            printf("%s%" PRId32, j ? "," : "", view.token_ids.data[j]);
        puts("]}");
    }
    status = 0;
cleanup:
    if (error) {
        trtmc_string_view message = core->error_message(error);
        fwrite(message.data, 1, (size_t)message.size, stderr);
        fputc('\n', stderr);
        core->error_release(error);
    }
    if (results) {
        for (i = 0; i < argc - 3; ++i)
            core->result_release(results[i]);
        free(results);
    }
    core->model_release(model);
    return status;
}

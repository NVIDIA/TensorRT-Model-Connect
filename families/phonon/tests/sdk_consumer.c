/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/audio.h>
#include <trtmc/trtmc.h>

static trtmc_string_view sv(const char* s) {
    trtmc_string_view result = {s, strlen(s)};
    return result;
}
static void check(trtmc_status status, const trtmc_core_api_v1* api, trtmc_error* error) {
    if (status != TRTMC_OK) {
        if (api && error) {
            const trtmc_string_view message = api->error_message(error);
            fprintf(stderr, "%.*s\n", (int)message.size, message.data);
            api->error_release(error);
        }
        exit(1);
    }
}
int main(int argc, char** argv) {
    if (argc != 3)
        return 2;
    const trtmc_core_api_v1* api = NULL;
    trtmc_error* error = NULL;
    check(trtmc_get_api(1, 0, &api), NULL, NULL);
    trtmc_load_options_v1 options = {0};
    options.struct_size = sizeof(options);
    options.runtime_root = sv(argv[2]);
    trtmc_model* model = NULL;
    trtmc_status status = api->model_load(sv(argv[1]), &options, &model, &error);
    check(status, api, error);
    const trtmc_api_header* header = NULL;
    status = api->model_get_task_api(model, sv(TRTMC_TASK_TEXT_TO_SPEECH), 1, 0, &header, &error);
    check(status, api, error);
    const trtmc_text_to_speech_api_v1* task = (const trtmc_text_to_speech_api_v1*)header;
    trtmc_text_to_speech_request_v1 request = {0};
    request.text = sv("Hello from Phonon.");
    trtmc_config_entry_v1 entry = {0};
    entry.name = sv("temperature");
    entry.value.kind = TRTMC_CONFIG_F64;
    entry.value.as.f64 = 0.0;
    const trtmc_config_view_v1 config = {&entry, 1};
    trtmc_result* result = NULL;
    status = task->run(model, &request, &config, &result, &error);
    check(status, api, error);
    trtmc_audio_result_view_v1 view = {0};
    status = task->result_view(result, &view, &error);
    check(status, api, error);
    if (!view.audio.has_sample_rate || view.audio.sample_rate != 24000 ||
        view.audio.channels != 1 || view.audio.sample_count != 46080)
        return 1;
    for (uint64_t i = 0; i < view.audio.sample_count; ++i)
        if (!isfinite(view.audio.samples[i]))
            return 1;
    printf("Phonon C SDK: %llu samples at %u Hz\n", (unsigned long long)view.audio.sample_count,
           view.audio.sample_rate);
    api->result_release(result);
    api->model_release(model);
    return 0;
}

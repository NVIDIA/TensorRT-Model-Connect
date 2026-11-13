/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal C SDK consumer for the vibevoice family: text_to_speech through the
 * public C API with two family config entries.
 *
 * Usage:
 *   test_vibevoice_sdk_c <bundle_path> <runtime_root>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/audio.h>
#include <trtmc/trtmc.h>

static void check(trtmc_status status, const trtmc_core_api_v1* api, trtmc_error** err,
                  const char* label) {
    if (status != TRTMC_OK) {
        if (api && *err) {
            trtmc_string_view msg = api->error_message(*err);
            fprintf(stderr, "%s failed: %.*s\n", label, (int)msg.size, msg.data ? msg.data : "");
            api->error_release(*err);
        } else {
            fprintf(stderr, "%s failed\n", label);
        }
        exit(1);
    }
}

static trtmc_string_view sv(const char* s) {
    trtmc_string_view v;
    v.data = s;
    v.size = strlen(s);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <bundle_path> <runtime_root>\n", argv[0]);
        return 1;
    }
    const trtmc_core_api_v1* api = NULL;
    trtmc_error* err = NULL;
    check(trtmc_get_api(1, 0, &api), NULL, &err, "trtmc_get_api");

    trtmc_load_options_v1 opts;
    memset(&opts, 0, sizeof(opts));
    opts.struct_size = sizeof(opts);
    opts.runtime_root = sv(argv[2]);
    trtmc_model* model = NULL;
    check(api->model_load(sv(argv[1]), &opts, &model, &err), api, &err, "model_load");

    const trtmc_api_header* header = NULL;
    check(api->model_get_task_api(model, sv(TRTMC_TASK_TEXT_TO_SPEECH), 1, 0, &header, &err), api,
          &err, "get text_to_speech");
    const trtmc_text_to_speech_api_v1* task = (const trtmc_text_to_speech_api_v1*)header;

    trtmc_text_to_speech_request_v1 request;
    memset(&request, 0, sizeof(request));
    request.text = sv("Hello from the C SDK.");

    trtmc_config_entry_v1 entries[2];
    memset(entries, 0, sizeof(entries));
    entries[0].name = sv("seed");
    entries[0].value.kind = TRTMC_CONFIG_I64;
    entries[0].value.as.i64 = 11;
    entries[1].name = sv("max_new_tokens");
    entries[1].value.kind = TRTMC_CONFIG_I64;
    entries[1].value.as.i64 = 48;
    trtmc_config_view_v1 config;
    config.entries = entries;
    config.count = 2;

    trtmc_result* result = NULL;
    check(task->run(model, &request, &config, &result, &err), api, &err, "text_to_speech run");
    trtmc_audio_result_view_v1 view;
    memset(&view, 0, sizeof(view));
    check(task->result_view(result, &view, &err), api, &err, "text_to_speech result_view");
    if (view.audio.sample_count == 0 || view.audio.sample_rate != 24000 ||
        view.audio.channels != 1 || view.audio.sample_count % 3200 != 0) {
        fprintf(stderr, "text_to_speech: unexpected audio (%llu samples, %u Hz, %u channels)\n",
                (unsigned long long)view.audio.sample_count, view.audio.sample_rate,
                view.audio.channels);
        api->result_release(result);
        api->model_release(model);
        return 1;
    }
    printf("text_to_speech: samples=%llu sample_rate=%u\n",
           (unsigned long long)view.audio.sample_count, view.audio.sample_rate);
    api->result_release(result);
    api->model_release(model);
    printf("vibevoice C SDK consumer: all checks passed.\n");
    return 0;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal C SDK consumer for the stable_audio family: one short text_to_audio
 * request through the public C API (trtmc/trtmc.h + trtmc/audio.h), with a
 * string, two integer and a float config entry. It requests the model's whole window,
 * so it runs against any bundle of the family.
 *
 * Usage:
 *   test_stable_audio_sdk_c <bundle_path> <runtime_root>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/audio.h>
#include <trtmc/trtmc.h>

static void check(trtmc_status status, const trtmc_core_api_v1* api, trtmc_error** err,
                  const char* label) {
    if (status == TRTMC_OK)
        return;
    if (api && *err) {
        trtmc_string_view msg = api->error_message(*err);
        fprintf(stderr, "%s failed: %.*s\n", label, (int)msg.size, msg.data ? msg.data : "");
        api->error_release(*err);
    } else {
        fprintf(stderr, "%s failed\n", label);
    }
    exit(1);
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
    check(api->model_get_task_api(model, sv(TRTMC_TASK_TEXT_TO_AUDIO), 1, 0, &header, &err), api,
          &err, "get text_to_audio");
    const trtmc_text_to_audio_api_v1* task = (const trtmc_text_to_audio_api_v1*)header;

    trtmc_text_to_audio_request_v1 request;
    memset(&request, 0, sizeof(request));
    request.prompt = sv("Soft rain on a tin roof.");

    trtmc_config_entry_v1 entries[4];
    memset(entries, 0, sizeof(entries));
    entries[0].name = sv("num_inference_steps");
    entries[0].value.kind = TRTMC_CONFIG_I64;
    entries[0].value.as.i64 = 4;
    entries[1].name = sv("audio_start_in_s");
    entries[1].value.kind = TRTMC_CONFIG_F64;
    entries[1].value.as.f64 = 0.0;
    entries[2].name = sv("seed");
    entries[2].value.kind = TRTMC_CONFIG_I64;
    entries[2].value.as.i64 = 7;
    entries[3].name = sv("negative_prompt");
    entries[3].value.kind = TRTMC_CONFIG_STRING;
    entries[3].value.as.string = sv("Low quality.");
    trtmc_config_view_v1 config;
    config.entries = entries;
    config.count = 4;

    trtmc_result* result = NULL;
    check(task->run(model, &request, &config, &result, &err), api, &err, "text_to_audio run");
    trtmc_audio_result_view_v1 view;
    memset(&view, 0, sizeof(view));
    check(task->result_view(result, &view, &err), api, &err, "text_to_audio result_view");

    /* The model's whole window: interleaved stereo frames at 44.1 kHz. */
    int ok = view.audio.sample_rate == 44100 && view.audio.channels == 2 &&
             view.audio.sample_count > 0 && view.audio.sample_count % 2 == 0;
    printf("text_to_audio: samples=%llu sample_rate=%u channels=%u\n",
           (unsigned long long)view.audio.sample_count, view.audio.sample_rate,
           view.audio.channels);
    api->result_release(result);
    api->model_release(model);
    if (!ok) {
        fprintf(stderr, "text_to_audio: expected stereo frames at 44100 Hz\n");
        return 1;
    }
    printf("stable_audio C SDK consumer: all tasks passed\n");
    return 0;
}

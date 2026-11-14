/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal C SDK consumer for the stable_audio_3 family.
 * Renders a short clip through text_to_audio with the public C API and checks
 * the output format: 44.1 kHz stereo, exactly duration * 44100 frames.
 *
 * Usage:
 *   test_stable_audio_3_sdk_c <bundle_path> <runtime_root>
 */

#include <math.h>
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
    check(api->model_get_task_api(model, sv(TRTMC_TASK_TEXT_TO_AUDIO), 1, 0, &header, &err), api,
          &err, "get text_to_audio");
    const trtmc_text_to_audio_api_v1* task = (const trtmc_text_to_audio_api_v1*)header;

    trtmc_text_to_audio_request_v1 request;
    memset(&request, 0, sizeof(request));
    request.prompt = sv("Gentle rain on a tin roof with distant thunder");

    trtmc_config_entry_v1 entries[3];
    memset(entries, 0, sizeof(entries));
    entries[0].name = sv("duration");
    entries[0].value.kind = TRTMC_CONFIG_F64;
    entries[0].value.as.f64 = 4.0;
    entries[1].name = sv("steps");
    entries[1].value.kind = TRTMC_CONFIG_I64;
    entries[1].value.as.i64 = 8;
    entries[2].name = sv("seed");
    entries[2].value.kind = TRTMC_CONFIG_I64;
    entries[2].value.as.i64 = 11;
    trtmc_config_view_v1 config;
    config.entries = entries;
    config.count = 3;

    trtmc_result* result = NULL;
    check(task->run(model, &request, &config, &result, &err), api, &err, "text_to_audio run");
    trtmc_audio_result_view_v1 view;
    memset(&view, 0, sizeof(view));
    check(task->result_view(result, &view, &err), api, &err, "text_to_audio result_view");

    int status = 0;
    if (view.audio.sample_rate != 44100 || view.audio.channels != 2 ||
        view.audio.sample_count != 4ULL * 44100ULL * 2ULL) {
        fprintf(stderr, "text_to_audio: unexpected format rate=%u channels=%u samples=%llu\n",
                view.audio.sample_rate, view.audio.channels,
                (unsigned long long)view.audio.sample_count);
        status = 1;
    }
    for (unsigned long long index = 0; status == 0 && index < view.audio.sample_count; ++index) {
        const float value = view.audio.samples[index];
        if (!isfinite(value) || value < -1.0F || value > 1.0F) {
            fprintf(stderr, "text_to_audio: sample %llu out of range\n", index);
            status = 1;
        }
    }
    if (status == 0)
        printf("text_to_audio: samples=%llu sample_rate=%u channels=%u\n",
               (unsigned long long)view.audio.sample_count, view.audio.sample_rate,
               view.audio.channels);
    api->result_release(result);
    api->model_release(model);
    if (status == 0)
        printf("stable_audio_3 C SDK consumer: all tasks passed.\n");
    return status;
}

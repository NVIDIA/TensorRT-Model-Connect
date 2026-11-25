/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal C SDK consumer for the canary family.
 * Exercises speech_transcription through the public C API (trtmc/trtmc.h + trtmc/audio.h).
 *
 * Usage:
 *   sdk_consumer_canary_c <bundle_path> <runtime_root>
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
        } else {
            fprintf(stderr, "%s failed: (no api or error)\n", label);
        }
        if (api && *err)
            api->error_release(*err);
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
    const char* bundle_path = argv[1];
    const char* runtime_root = argv[2];

    const trtmc_core_api_v1* api = NULL;
    trtmc_error* err = NULL;
    check(trtmc_get_api(1, 0, &api), NULL, &err, "trtmc_get_api");

    trtmc_load_options_v1 opts;
    memset(&opts, 0, sizeof(opts));
    opts.struct_size = sizeof(opts);
    opts.runtime_root = sv(runtime_root);

    trtmc_model* model = NULL;
    check(api->model_load(sv(bundle_path), &opts, &model, &err), api, &err, "model_load");

    const trtmc_api_header* task_header = NULL;
    check(api->model_get_task_api(model, sv(TRTMC_TASK_SPEECH_TRANSCRIPTION), 1, 0, &task_header,
                                  &err),
          api, &err, "get speech_transcription");

    const trtmc_speech_transcription_api_v1* task =
        (const trtmc_speech_transcription_api_v1*)task_header;

    float pcm[16000];
    memset(pcm, 0, sizeof(pcm));

    trtmc_speech_transcription_request_v1 req;
    memset(&req, 0, sizeof(req));
    req.audio.samples = pcm;
    req.audio.sample_count = 16000;
    req.audio.channels = 1;
    req.audio.has_sample_rate = 1;
    req.audio.sample_rate = 16000;

    trtmc_config_view_v1 cfg;
    memset(&cfg, 0, sizeof(cfg));

    trtmc_result* result = NULL;
    check(task->run(model, &req, &cfg, &result, &err), api, &err, "speech_transcription run");

    /* Release model before result to verify result owns its data */
    check(api->model_release(model), api, &err, "model_release");

    trtmc_text_result_view_v1 view;
    memset(&view, 0, sizeof(view));
    check(task->result_view(result, &view, &err), api, &err, "result_view");

    check(api->result_release(result), api, &err, "result_release");
    return 0;
}

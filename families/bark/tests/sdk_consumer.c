/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal C SDK consumer for the bark family.
 * Exercises text_to_audio through the public C API (trtmc/trtmc.h + trtmc/audio.h).
 *
 * Usage:
 *   sdk_consumer_bark_c <bundle_path> <runtime_root>
 *
 * Environment:
 *   TRTMC_BARK_PROMPT   text prompt (default: "Hello from Bark.")
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/audio.h>
#include <trtmc/trtmc.h>

/* ── helpers ─────────────────────────────────────────────────────────────── */

static void check(trtmc_status status, const trtmc_core_api_v1* api, trtmc_error** err,
                  const char* label) {
    if (status != TRTMC_OK) {
        fprintf(stderr, "%s failed: %s\n", label, api ? api->error_message(*err) : "(no api)");
        if (api && *err)
            api->error_release(*err);
        exit(1);
    }
}

static trtmc_string_view sv(const char* s) {
    trtmc_string_view v;
    v.data = s;
    v.length = strlen(s);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <bundle_path> <runtime_root>\n", argv[0]);
        return 1;
    }
    const char* bundle_path = argv[1];
    const char* runtime_root = argv[2];

    const char* env_prompt = getenv("TRTMC_BARK_PROMPT");
    const char* prompt = env_prompt ? env_prompt : "Hello from Bark.";

    /* ── load API ─────────────────────────────────────────────────────────── */
    const trtmc_core_api_v1* api = NULL;
    trtmc_error* err = NULL;
    check(trtmc_get_api(1, 0, &api), NULL, &err, "trtmc_get_api");

    /* ── load model ───────────────────────────────────────────────────────── */
    trtmc_load_options_v1 opts;
    memset(&opts, 0, sizeof(opts));
    opts.runtime_root = sv(runtime_root);

    trtmc_model* model = NULL;
    check(api->model_load(sv(bundle_path), &opts, &model, &err), api, &err, "model_load");

    /* ── text_to_audio ────────────────────────────────────────────────────── */
    {
        const trtmc_api_header* task_header = NULL;
        check(
            api->model_get_task_api(model, sv(TRTMC_TASK_TEXT_TO_AUDIO), 1, 0, &task_header, &err),
            api, &err, "get text_to_audio");

        const trtmc_text_to_audio_api_v1* task = (const trtmc_text_to_audio_api_v1*)task_header;

        trtmc_text_to_audio_request_v1 req;
        memset(&req, 0, sizeof(req));
        req.prompt = sv(prompt);

        trtmc_config_view_v1 cfg;
        memset(&cfg, 0, sizeof(cfg));

        trtmc_result* result = NULL;
        check(task->run(model, &req, &cfg, &result, &err), api, &err, "text_to_audio run");

        trtmc_audio_result_view_v1 view;
        memset(&view, 0, sizeof(view));
        check(task->result_view(result, &view, &err), api, &err, "text_to_audio result_view");

        if (view.audio.sample_count == 0) {
            fprintf(stderr, "text_to_audio: expected non-empty audio\n");
            api->result_release(result);
            api->model_release(model);
            return 1;
        }
        if (view.audio.sample_rate != 24000) {
            fprintf(stderr, "text_to_audio: expected sample_rate == 24000, got %u\n",
                    view.audio.sample_rate);
            api->result_release(result);
            api->model_release(model);
            return 1;
        }
        if (view.audio.channels != 1) {
            fprintf(stderr, "text_to_audio: expected channels == 1, got %u\n", view.audio.channels);
            api->result_release(result);
            api->model_release(model);
            return 1;
        }

        printf("text_to_audio: samples=%llu sample_rate=%u channels=%u\n",
               (unsigned long long)view.audio.sample_count, view.audio.sample_rate,
               view.audio.channels);
        api->result_release(result);
    }

    api->model_release(model);
    printf("bark C SDK consumer: all tasks passed.\n");
    return 0;
}

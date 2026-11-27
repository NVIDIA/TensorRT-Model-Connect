/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal C SDK consumer for the gpt2 family.
 * Exercises text_continuation through the public C API (trtmc/trtmc.h).
 *
 * Usage:
 *   sdk_consumer_gpt2_c <bundle_path> <runtime_root>
 *
 * Environment:
 *   TRTMC_GPT2_PROMPT   text prompt (default: "The quick brown fox jumps over the lazy dog")
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/trtmc.h>

/* ── helpers ─────────────────────────────────────────────────────────────── */

static void check(trtmc_status status, const trtmc_core_api_v1* api, trtmc_error** err,
                  const char* label) {
    if (status != TRTMC_OK) {
        if (api && *err) {
            trtmc_string_view msg = api->error_message(*err);
            fprintf(stderr, "%s failed: %.*s\n", label, (int)msg.size, msg.data ? msg.data : "");
            api->error_release(*err);
        } else {
            fprintf(stderr, "%s failed: (no api or error)\n", label);
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
    const char* bundle_path = argv[1];
    const char* runtime_root = argv[2];

    const char* env_prompt = getenv("TRTMC_GPT2_PROMPT");
    const char* prompt = env_prompt ? env_prompt : "The quick brown fox jumps over the lazy dog";

    /* ── load API ─────────────────────────────────────────────────────────── */
    const trtmc_core_api_v1* api = NULL;
    trtmc_error* err = NULL;
    check(trtmc_get_api(1, 0, &api), NULL, &err, "trtmc_get_api");

    /* ── load model ───────────────────────────────────────────────────────── */
    trtmc_load_options_v1 opts;
    memset(&opts, 0, sizeof(opts));
    opts.struct_size = sizeof(opts);
    opts.runtime_root = sv(runtime_root);

    trtmc_model* model = NULL;
    check(api->model_load(sv(bundle_path), &opts, &model, &err), api, &err, "model_load");

    /* ── text_continuation ────────────────────────────────────────────────── */
    {
        const trtmc_api_header* task_header = NULL;
        check(api->model_get_task_api(model, sv(TRTMC_TASK_TEXT_CONTINUATION), 1, 0,
                                      &task_header, &err),
              api, &err, "get text_continuation");

        const trtmc_text_continuation_api_v1* task =
            (const trtmc_text_continuation_api_v1*)task_header;

        trtmc_text_continuation_request_v1 req;
        memset(&req, 0, sizeof(req));
        req.prefix.kind = TRTMC_TEXT_UTF8;
        req.prefix.as.text = sv(prompt);

        trtmc_config_view_v1 cfg;
        memset(&cfg, 0, sizeof(cfg));

        trtmc_result* result = NULL;
        check(task->run(model, &req, &cfg, &result, &err), api, &err, "text_continuation run");

        trtmc_text_result_view_v1 view;
        memset(&view, 0, sizeof(view));
        check(task->result_view(result, &view, &err), api, &err, "text_continuation result_view");

        if (view.text.size == 0 && view.token_ids.size == 0) {
            fprintf(stderr, "text_continuation: expected non-empty text or token_ids\n");
            api->result_release(result);
            api->model_release(model);
            return 1;
        }

        printf("text_continuation: text=%.*s token_ids_count=%llu\n", (int)view.text.size,
               view.text.data ? view.text.data : "", (unsigned long long)view.token_ids.size);
        api->result_release(result);

        /* ── invalid prompt rejection test ────────────────────────────────── */
        req.prefix.as.text = sv("");
        trtmc_result* bad_res = NULL;
        trtmc_error* bad_err = NULL;
        trtmc_status status = task->run(model, &req, &cfg, &bad_res, &bad_err);
        if (status == TRTMC_OK) {
            fprintf(stderr, "expected empty prompt to be rejected\n");
            api->result_release(bad_res);
            api->model_release(model);
            return 1;
        }
        if (bad_err) {
            api->error_release(bad_err);
        }
    }

    api->model_release(model);
    printf("gpt2 C SDK consumer: all tasks passed.\n");
    return 0;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Minimal C SDK consumer for the albert family.
 * Exercises text_to_token_features, text_to_embedding, and text_pair_to_relevance
 * through the public C API (trtmc/trtmc.h + trtmc/features.h).
 *
 * Usage:
 *   sdk_consumer_albert_c <bundle_path> <runtime_root>
 *
 * Environment:
 *   TRTMC_ALBERT_TEXT       input text for token-features and embedding (default: "hello world")
 *   TRTMC_ALBERT_QUERY      query string for relevance test (default: "What is AI?")
 *   TRTMC_ALBERT_DOCUMENT   document string for relevance test (default: "AI is intelligence.")
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/features.h>
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

    const char* text = getenv("TRTMC_ALBERT_TEXT");
    const char* query = getenv("TRTMC_ALBERT_QUERY");
    const char* document = getenv("TRTMC_ALBERT_DOCUMENT");
    if (!text)
        text = "hello world";
    if (!query)
        query = "What is AI?";
    if (!document)
        document = "AI is intelligence.";

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

    /* ── text_to_pooled_features ─────────────────────────────────────────── */
    {
        const trtmc_api_header* task_header = NULL;
        check(api->model_get_task_api(model, sv(TRTMC_TASK_TEXT_TO_POOLED_FEATURES), 1, 0,
                                      &task_header, &err),
              api, &err, "get text_to_pooled_features");

        const trtmc_text_to_pooled_features_api_v1* task =
            (const trtmc_text_to_pooled_features_api_v1*)task_header;

        trtmc_text_to_pooled_features_request_v1 req;
        memset(&req, 0, sizeof(req));
        req.text.kind = TRTMC_TEXT_UTF8;
        req.text.as.text = sv(text);

        trtmc_config_view_v1 cfg;
        memset(&cfg, 0, sizeof(cfg));

        trtmc_result* result = NULL;
        check(task->run(model, &req, &cfg, &result, &err), api, &err, "pooled_features run");

        trtmc_pooled_features_view_v1 view;
        memset(&view, 0, sizeof(view));
        check(task->result_view(result, &view, &err), api, &err, "pooled_features result_view");

        if (view.count == 0) {
            fprintf(stderr, "text_to_pooled_features: expected non-empty features\n");
            api->result_release(result);
            api->model_release(model);
            return 1;
        }
        printf("text_to_pooled_features: dim=%llu pooling=%.*s normalization=%.*s\n",
               (unsigned long long)view.count, (int)view.pooling.size, view.pooling.data,
               (int)view.normalization.size, view.normalization.data);
        api->result_release(result);
    }

    /* ── text_to_token_features ───────────────────────────────────────────── */
    {
        const trtmc_api_header* task_header = NULL;
        check(api->model_get_task_api(model, sv(TRTMC_TASK_TEXT_TO_TOKEN_FEATURES), 1, 0,
                                      &task_header, &err),
              api, &err, "get text_to_token_features");

        const trtmc_text_to_token_features_api_v1* task =
            (const trtmc_text_to_token_features_api_v1*)task_header;

        trtmc_text_to_token_features_request_v1 req;
        memset(&req, 0, sizeof(req));
        req.text.kind = TRTMC_TEXT_UTF8;
        req.text.as.text = sv(text);

        trtmc_config_view_v1 cfg;
        memset(&cfg, 0, sizeof(cfg));

        trtmc_result* result = NULL;
        check(task->run(model, &req, &cfg, &result, &err), api, &err, "token_features run");

        trtmc_token_features_view_v1 view;
        memset(&view, 0, sizeof(view));
        check(task->result_view(result, &view, &err), api, &err, "token_features result_view");

        if (view.features.count == 0) {
            fprintf(stderr, "text_to_token_features: expected non-empty features\n");
            api->result_release(result);
            api->model_release(model);
            return 1;
        }
        printf("text_to_token_features: tokens=%llu features=%llu\n",
               (unsigned long long)view.token_count, (unsigned long long)view.features.count);
        api->result_release(result);
    }

    /* ── text_to_embedding ───────────────────────────────────────────────── */
    {
        const trtmc_api_header* task_header = NULL;
        check(api->model_get_task_api(model, sv(TRTMC_TASK_TEXT_TO_EMBEDDING), 1, 0, &task_header,
                                      &err),
              api, &err, "get text_to_embedding");

        const trtmc_text_to_embedding_api_v1* task =
            (const trtmc_text_to_embedding_api_v1*)task_header;

        trtmc_text_to_embedding_request_v1 req;
        memset(&req, 0, sizeof(req));
        req.text = sv(text);
        req.role = TRTMC_EMBEDDING_DEFAULT;

        trtmc_config_view_v1 cfg;
        memset(&cfg, 0, sizeof(cfg));

        trtmc_result* result = NULL;
        check(task->run(model, &req, &cfg, &result, &err), api, &err, "embedding run");

        trtmc_semantic_embedding_view_v1 view;
        memset(&view, 0, sizeof(view));
        check(task->result_view(result, &view, &err), api, &err, "embedding result_view");

        if (view.count == 0) {
            fprintf(stderr, "text_to_embedding: expected non-empty embedding\n");
            api->result_release(result);
            api->model_release(model);
            return 1;
        }
        printf("text_to_embedding: dim=%llu pooling=%.*s normalization=%.*s\n",
               (unsigned long long)view.count, (int)view.pooling.size, view.pooling.data,
               (int)view.normalization.size, view.normalization.data);
        api->result_release(result);
    }

    /* ── text_pair_to_relevance ──────────────────────────────────────────── */
    {
        const trtmc_api_header* task_header = NULL;
        check(api->model_get_task_api(model, sv(TRTMC_TASK_TEXT_PAIR_TO_RELEVANCE), 1, 0,
                                      &task_header, &err),
              api, &err, "get text_pair_to_relevance");

        const trtmc_text_pair_to_relevance_api_v1* task =
            (const trtmc_text_pair_to_relevance_api_v1*)task_header;

        trtmc_text_pair_to_relevance_request_v1 req;
        memset(&req, 0, sizeof(req));
        req.query = sv(query);
        req.document = sv(document);

        trtmc_config_view_v1 cfg;
        memset(&cfg, 0, sizeof(cfg));

        trtmc_result* result = NULL;
        check(task->run(model, &req, &cfg, &result, &err), api, &err, "relevance run");

        trtmc_relevance_view_v1 view;
        memset(&view, 0, sizeof(view));
        check(task->result_view(result, &view, &err), api, &err, "relevance result_view");

        printf("text_pair_to_relevance: score=%.6f\n", view.score);
        api->result_release(result);
    }

    api->model_release(model);
    printf("albert C SDK consumer: all tasks passed.\n");
    return 0;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C SDK consumer for the family-owned end-to-end test:
 * BUNDLE RUNTIME_ROOT ROLE TEXT, ROLE = default | query | document. */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <trtmc/trtmc.h>

static trtmc_string_view string_view(const char* text) {
    trtmc_string_view view = {text, strlen(text)};
    return view;
}

static int equal(trtmc_string_view view, const char* text) {
    return view.size == strlen(text) && memcmp(view.data, text, view.size) == 0;
}

static int role_of(const char* name, uint32_t* role) {
    if (strcmp(name, "default") == 0)
        *role = TRTMC_EMBEDDING_DEFAULT;
    else if (strcmp(name, "query") == 0)
        *role = TRTMC_EMBEDDING_QUERY;
    else if (strcmp(name, "document") == 0)
        *role = TRTMC_EMBEDDING_DOCUMENT;
    else
        return 0;
    return 1;
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_text_to_embedding_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_text_to_embedding_request_v1 request = {0};
    trtmc_semantic_embedding_view_v1 view = {0};
    uint64_t count = 0;
    int status = 1;
    if (argc != 5 || !role_of(argv[3], &request.role) || trtmc_get_api(1, 0, &core) != TRTMC_OK)
        return 2;
    options.struct_size = sizeof(options);
    options.runtime_root = string_view(argv[2]);
    if (core->model_load(string_view(argv[1]), &options, &model, &error) != TRTMC_OK)
        goto cleanup;
    if (core->model_get_task_api(model, string_view(TRTMC_TASK_TEXT_TO_EMBEDDING), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    task = (const trtmc_text_to_embedding_api_v1*)header;
    if (core->config_field_count(model, string_view(TRTMC_TASK_TEXT_TO_EMBEDDING), 1, 0, &count,
                                 &error) != TRTMC_OK ||
        count != 0)
        goto cleanup;
    request.text = string_view(argv[4]);
    if (task->run(model, &request, NULL, &result, &error) != TRTMC_OK)
        goto cleanup;
    core->model_release(model);
    model = NULL;
    if (task->result_view(result, &view, &error) != TRTMC_OK || view.count == 0 ||
        !equal(view.pooling, "mean") || !equal(view.normalization, "l2"))
        goto cleanup;
    for (uint64_t index = 0; index < view.count; ++index)
        if (!isfinite(view.values[index]))
            goto cleanup;
    printf("{\"task\":\"text_to_embedding\",\"pooling\":\"mean\",\"normalization\":\"l2\","
           "\"values\":[");
    for (uint64_t index = 0; index < view.count; ++index)
        printf("%s%.9g", index ? "," : "", view.values[index]);
    printf("]}\n");
    if (fflush(stdout) == 0 && !ferror(stdout))
        status = 0;
cleanup:
    if (error != NULL) {
        const trtmc_string_view message = core->error_message(error);
        fprintf(stderr, "%.*s\n", (int)message.size, message.data);
    }
    core->error_release(error);
    core->result_release(result);
    core->model_release(model);
    return status;
}

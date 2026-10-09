/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

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

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_text_to_pooled_features_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_text_to_pooled_features_request_v1 request = {0};
    trtmc_pooled_features_view_v1 view = {0};
    uint64_t count = 0;
    int status = 1;
    if (argc != 4 || trtmc_get_api(1, 0, &core) != TRTMC_OK)
        return 2;
    options.struct_size = sizeof(options);
    options.runtime_root = string_view(argv[2]);
    if (core->model_load(string_view(argv[1]), &options, &model, &error) != TRTMC_OK)
        goto cleanup;
    if (core->model_get_task_api(model, string_view(TRTMC_TASK_TEXT_TO_POOLED_FEATURES), 1, 0,
                                 &header, &error) != TRTMC_OK)
        goto cleanup;
    task = (const trtmc_text_to_pooled_features_api_v1*)header;
    if (core->config_field_count(model, string_view(TRTMC_TASK_TEXT_TO_POOLED_FEATURES), 1, 0,
                                 &count, &error) != TRTMC_OK ||
        count != 0)
        goto cleanup;
    request.text.kind = TRTMC_TEXT_UTF8;
    request.text.as.text = string_view(argv[3]);
    if (task->run(model, &request, NULL, &result, &error) != TRTMC_OK)
        goto cleanup;
    core->model_release(model);
    model = NULL;
    if (task->result_view(result, &view, &error) != TRTMC_OK || view.count == 0 ||
        !equal(view.pooling, "cls") || !equal(view.normalization, "none"))
        goto cleanup;
    for (uint64_t index = 0; index < view.count; ++index)
        if (!isfinite(view.values[index]))
            goto cleanup;
    printf("{\"task\":\"text_to_pooled_features\",\"pooling\":\"cls\","
           "\"normalization\":\"none\",\"values\":[");
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

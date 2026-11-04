/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/text.h>
#include <trtmc/trtmc.h>

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}

static void json_string(trtmc_string_view value) {
    uint64_t i;
    putchar('"');
    for (i = 0; i < value.size; ++i) {
        const unsigned char byte = (unsigned char)value.data[i];
        if (byte == '"' || byte == '\\') {
            putchar('\\');
            putchar(byte);
        } else if (byte < 32) {
            printf("\\u%04x", (unsigned)byte);
        } else {
            putchar(byte);
        }
    }
    putchar('"');
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_conditional_text_generation_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_conditional_text_generation_request_v1 request = {0};
    trtmc_text_result_view_v1 view = {0};
    uint64_t i, field_count = 0;
    int status = 1;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s BUNDLE RUNTIME_ROOT TEXT\n", argv[0]);
        return 2;
    }
    if (trtmc_get_api(1, 0, &core) != TRTMC_OK || !core) {
        fprintf(stderr, "unable to obtain the v1 C API\n");
        goto cleanup;
    }
    options.struct_size = sizeof(options);
    options.runtime_root = text(argv[2]);
    if (core->model_load(text(argv[1]), &options, &model, &error) != TRTMC_OK)
        goto cleanup;
    if (core->model_get_task_api(model, text(TRTMC_TASK_CONDITIONAL_TEXT_GENERATION), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*task)) {
        fprintf(stderr, "incomplete conditional_text_generation C table\n");
        goto cleanup;
    }
    task = (const trtmc_conditional_text_generation_api_v1*)header;
    if (core->config_field_count(model, text(TRTMC_TASK_CONDITIONAL_TEXT_GENERATION), 1, 0,
                                 &field_count, &error) != TRTMC_OK)
        goto cleanup;
    if (field_count != 1) {
        fprintf(stderr, "bart must expose exactly 1 config field: max_new_tokens\n");
        goto cleanup;
    }

    request.source.kind = TRTMC_TEXT_UTF8;
    request.source.as.text = text(argv[3]);
    if (task->run(model, &request, NULL, &result, &error) != TRTMC_OK)
        goto cleanup;

    core->model_release(model);
    model = NULL;

    if (task->result_view(result, &view, &error) != TRTMC_OK)
        goto cleanup;

    printf("{\"task\":\"conditional_text_generation\",\"text\":");
    json_string(view.text);
    printf(",\"tokens\":[");
    for (i = 0; i < view.token_count; ++i) {
        if (i)
            putchar(',');
        printf("%" PRId32, view.tokens[i]);
    }
    printf("]}\n");
    status = 0;

cleanup:
    if (error && core) {
        const trtmc_string_view message = core->error_message(error);
        fprintf(stderr, "%.*s\n", (int)message.size, message.data);
        core->error_release(error);
    }
    if (result && core)
        core->result_release(result);
    if (model && core)
        core->model_release(model);
    return status;
}

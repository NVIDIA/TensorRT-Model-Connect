/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public SDK consumer for the family-owned end-to-end test. */
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/features.h>
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
    const trtmc_text_to_pooled_features_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_text_to_pooled_features_request_v1 request = {0};
    trtmc_pooled_features_view_v1 view = {0};
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
    if (core->model_get_task_api(model, text(TRTMC_TASK_TEXT_TO_POOLED_FEATURES), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*task)) {
        fprintf(stderr, "incomplete pooled-features C table\n");
        goto cleanup;
    }
    task = (const trtmc_text_to_pooled_features_api_v1*)header;
    if (core->config_field_count(model, text(TRTMC_TASK_TEXT_TO_POOLED_FEATURES), 1, 0,
                                 &field_count, &error) != TRTMC_OK)
        goto cleanup;
    if (field_count != 0) {
        fprintf(stderr, "MPNet must expose no runtime Config fields\n");
        goto cleanup;
    }
    request.text.kind = TRTMC_TEXT_UTF8;
    request.text.as.text = text(argv[3]);
    if (task->run(model, &request, NULL, &result, &error) != TRTMC_OK)
        goto cleanup;
    core->model_release(model);
    model = NULL;
    if (task->result_view(result, &view, &error) != TRTMC_OK)
        goto cleanup;
    if (!view.count) {
        fprintf(stderr, "MPNet must provide a nonempty pooled feature vector\n");
        goto cleanup;
    }
    for (i = 0; i < view.count; ++i) {
        if (!isfinite(view.values[i])) {
            fprintf(stderr, "pooled features contain a nonfinite value\n");
            goto cleanup;
        }
    }
    printf("{\"task\":\"text_to_pooled_features\",\"pooling\":");
    json_string(view.pooling);
    printf(",\"normalization\":");
    json_string(view.normalization);
    printf(",\"dim\":%" PRIu64 ",\"values\":[", view.count);
    for (i = 0; i < view.count; ++i) {
        if (i)
            putchar(',');
        printf("%.*g", FLT_DECIMAL_DIG, (double)view.values[i]);
    }
    puts("]}");
    status = fflush(stdout) == 0 && !ferror(stdout) ? 0 : 1;
cleanup:
    if (error && core) {
        const trtmc_string_view message = core->error_message(error);
        fprintf(stderr, "C API error %d: ", (int)core->error_code(error));
        fwrite(message.data, 1, (size_t)message.size, stderr);
        fputc('\n', stderr);
        core->error_release(error);
    }
    if (core) {
        core->result_release(result);
        core->model_release(model);
    }
    return status;
}

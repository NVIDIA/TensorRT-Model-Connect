/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* A direct public C SDK caller. No family or application headers. */
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/trtmc.h>

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, strlen(value)};
    return result;
}
static void json_string(trtmc_string_view value) {
    putchar('"');
    for (uint64_t i = 0; i < value.size; ++i) {
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
    const trtmc_text_continuation_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_text_continuation_request_v1 request = {0};
    trtmc_text_result_view_v1 view = {0};
    trtmc_load_options_v1 options = {0};
    trtmc_config_entry_v1* entries = NULL;
    int32_t* ids = NULL;
    FILE* file = NULL;
    uint64_t field_count = 0;
    int status = 1;
    if (argc < 5) {
        fprintf(stderr, "Usage: %s BUNDLE RUNTIME_ROOT text|tokens INPUT [KEY=VALUE ...]\n",
                argv[0]);
        return 2;
    }
    if (strcmp(argv[3], "text") == 0) {
        request.prefix.kind = TRTMC_TEXT_UTF8;
        request.prefix.as.text = text(argv[4]);
    } else if (strcmp(argv[3], "tokens") == 0) {
        file = fopen(argv[4], "rb");
        if (!file || fseek(file, 0, SEEK_END))
            goto cleanup;
        const long bytes = ftell(file);
        if (bytes < 0 || (unsigned long)bytes > SIZE_MAX || bytes % (long)sizeof(int32_t) ||
            fseek(file, 0, SEEK_SET))
            goto cleanup;
        const size_t count = (size_t)bytes / sizeof(int32_t);
        ids = count ? (int32_t*)malloc((size_t)bytes) : NULL;
        if (count && (!ids || fread(ids, sizeof(int32_t), count, file) != count))
            goto cleanup;
        fclose(file);
        file = NULL;
        request.prefix.kind = TRTMC_TEXT_TOKEN_IDS;
        request.prefix.as.token_ids = (trtmc_i32_view){ids, count};
    } else {
        goto cleanup;
    }
    if (trtmc_get_api(1, 0, &core) != TRTMC_OK || !core)
        goto cleanup;
    options.struct_size = sizeof(options);
    options.runtime_root = text(argv[2]);
    if (core->model_load(text(argv[1]), &options, &model, &error) != TRTMC_OK ||
        core->model_get_task_api(model, text(TRTMC_TASK_TEXT_CONTINUATION), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*task))
        goto cleanup;
    task = (const trtmc_text_continuation_api_v1*)header;
    if (core->config_field_count(model, text(TRTMC_TASK_TEXT_CONTINUATION), 1, 0, &field_count,
                                 &error) != TRTMC_OK)
        goto cleanup;
    entries = (trtmc_config_entry_v1*)calloc((size_t)argc, sizeof(*entries));
    if (!entries)
        goto cleanup;
    for (int i = 5; i < argc; ++i) {
        char* equal = strchr(argv[i], '=');
        if (!equal)
            goto cleanup;
        *equal = '\0';
        trtmc_config_entry_v1* entry = &entries[i - 5];
        entry->name = text(argv[i]);
        for (uint64_t index = 0; index < field_count; ++index) {
            trtmc_config_field_v1 field = {0};
            if (core->config_field_info(model, text(TRTMC_TASK_TEXT_CONTINUATION), 1, 0, index,
                                        &field, &error) != TRTMC_OK)
                goto cleanup;
            if (field.name.size == entry->name.size &&
                memcmp(field.name.data, entry->name.data, (size_t)field.name.size) == 0)
                entry->value.kind = field.kind;
        }
        const char* value = equal + 1;
        char* end = NULL;
        errno = 0;
        switch (entry->value.kind) {
        case TRTMC_CONFIG_I64:
            entry->value.as.i64 = strtoll(value, &end, 10);
            if (errno || end == value || *end)
                goto cleanup;
            break;
        case TRTMC_CONFIG_F64:
            entry->value.as.f64 = strtod(value, &end);
            if (errno || end == value || *end || !isfinite(entry->value.as.f64))
                goto cleanup;
            break;
        case TRTMC_CONFIG_BOOL:
            if (strcmp(value, "true") && strcmp(value, "false"))
                goto cleanup;
            entry->value.as.boolean = strcmp(value, "true") == 0;
            break;
        case TRTMC_CONFIG_STRING:
            entry->value.as.string = text(value);
            break;
        default:
            goto cleanup;
        }
    }
    const trtmc_config_view_v1 config = {entries, (uint64_t)(argc - 5)};
    if (task->run(model, &request, &config, &result, &error) != TRTMC_OK)
        goto cleanup;
    free(ids);
    ids = NULL;
    core->model_release(model);
    model = NULL;
    if (task->result_view(result, &view, &error) != TRTMC_OK)
        goto cleanup;
    printf("{\"text\":");
    json_string(view.text);
    printf(",\"token_ids\":[");
    for (uint64_t i = 0; i < view.token_ids.size; ++i) {
        if (i)
            putchar(',');
        printf("%" PRId32, view.token_ids.data[i]);
    }
    printf("],\"setup_ms\":%.17g,\"prefill_ms\":%.17g,\"decode_ms\":%.17g}\n", view.setup_ms,
           view.prefill_ms, view.decode_ms);
    status = fflush(stdout) == 0 && !ferror(stdout) ? 0 : 1;
cleanup:
    if (error && core) {
        const trtmc_string_view message = core->error_message(error);
        fwrite(message.data, 1, (size_t)message.size, stderr);
        fputc('\n', stderr);
        core->error_release(error);
    }
    if (core) {
        core->result_release(result);
        core->model_release(model);
    }
    if (file)
        fclose(file);
    free(entries);
    free(ids);
    return status;
}

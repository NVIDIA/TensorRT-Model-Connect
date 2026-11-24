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
    trtmc_load_options_v1 options = {0};
    trtmc_text_continuation_request_v1 request = {0};
    trtmc_text_continuation_result_view_v1 view = {0};
    trtmc_config_entry_v1* entries = NULL;
    int32_t* ids = NULL;
    FILE* file = NULL;
    uint64_t field_count = 0;
    int status = 1;
    if (argc < 5) {
        fprintf(stderr,
                "Usage: %s BUNDLE RUNTIME_ROOT text|tokens INPUT [KEY=VALUE ...]\n",
                argv[0]);
        return 2;
    }
    if (trtmc_query_api(TRTMC_API_CORE_V1, &header, &error) != TRTMC_STATUS_OK)
        goto finish;
    core = (const trtmc_core_api_v1*)header;
    if (trtmc_query_api(TRTMC_API_TEXT_CONTINUATION_V1, &header, &error) !=
        TRTMC_STATUS_OK)
        goto finish;
    task = (const trtmc_text_continuation_api_v1*)header;
    options.struct_size = sizeof(options);
    options.runtime_root = argv[2];
    if (core->model_load(argv[1], &options, &model, &error) != TRTMC_STATUS_OK)
        goto finish;
    if (core->model_config_fields(model, task->task.data, &field_count, NULL,
                                  &error) != TRTMC_STATUS_OK)
        goto finish;
    trtmc_config_field_view_v1* fields =
        calloc(field_count, sizeof(trtmc_config_field_view_v1));
    if (!fields) {
        fprintf(stderr, "allocation failure\n");
        goto finish;
    }
    if (core->model_config_fields(model, task->task.data, &field_count, fields,
                                  &error) != TRTMC_STATUS_OK) {
        free(fields);
        goto finish;
    }
    entries = calloc((size_t)(argc - 5), sizeof(trtmc_config_entry_v1));
    for (int i = 5; i < argc; ++i) {
        char* argument = argv[i];
        char* equal = strchr(argument, '=');
        if (!equal) {
            fprintf(stderr, "invalid Config argument: %s\n", argument);
            free(fields);
            goto finish;
        }
        *equal = '\0';
        const char* value_str = equal + 1;
        const trtmc_config_field_view_v1* field = NULL;
        for (uint64_t f = 0; f < field_count; ++f) {
            if (fields[f].name.size == strlen(argument) &&
                memcmp(fields[f].name.data, argument, fields[f].name.size) == 0) {
                field = &fields[f];
                break;
            }
        }
        if (!field) {
            fprintf(stderr, "unknown Config field: %s\n", argument);
            free(fields);
            goto finish;
        }
        entries[i - 5].name = text(argument);
        entries[i - 5].kind = field->kind;
        if (field->kind == TRTMC_CONFIG_KIND_STRING) {
            entries[i - 5].value.string_value = text(value_str);
        } else if (field->kind == TRTMC_CONFIG_KIND_BOOL) {
            entries[i - 5].value.bool_value = strcmp(value_str, "true") == 0;
        } else if (field->kind == TRTMC_CONFIG_KIND_I64) {
            entries[i - 5].value.i64_value = strtoll(value_str, NULL, 10);
        } else if (field->kind == TRTMC_CONFIG_KIND_F64) {
            entries[i - 5].value.f64_value = strtod(value_str, NULL);
        }
    }
    free(fields);
    request.struct_size = sizeof(request);
    if (strcmp(argv[3], "text") == 0) {
        request.prefix_kind = TRTMC_TEXT_CONTINUATION_PREFIX_TEXT;
        request.prefix.text = text(argv[4]);
    } else if (strcmp(argv[3], "tokens") == 0) {
        file = fopen(argv[4], "rb");
        if (!file) {
            fprintf(stderr, "could not open token file\n");
            goto finish;
        }
        if (fseek(file, 0, SEEK_END) != 0)
            goto finish;
        long bytes = ftell(file);
        if (bytes < 0 || bytes % (long)sizeof(int32_t) != 0) {
            fprintf(stderr, "token input must contain int32 values\n");
            goto finish;
        }
        rewind(file);
        request.prefix_kind = TRTMC_TEXT_CONTINUATION_PREFIX_TOKEN_IDS;
        request.prefix.token_ids.count = (uint64_t)bytes / sizeof(int32_t);
        ids = malloc((size_t)bytes);
        if (bytes > 0 && fread(ids, (size_t)bytes, 1, file) != 1) {
            fprintf(stderr, "could not read token file\n");
            goto finish;
        }
        request.prefix.token_ids.values = ids;
    } else {
        fprintf(stderr, "mode must be text or tokens\n");
        goto finish;
    }
    trtmc_config_view_v1 supplied = {(uint64_t)(argc - 5), entries};
    if (task->run(model, &request, &supplied, &result, &error) != TRTMC_STATUS_OK)
        goto finish;
    /* Release the model handle before reading the result view. The result owns
     * all data it exposes. */
    core->model_release(model);
    model = NULL;
    if (task->result_view(result, &view, &error) != TRTMC_STATUS_OK)
        goto finish;
    fputs("{\"text\":", stdout);
    json_string(view.text);
    fputs(",\"token_ids\":[", stdout);
    for (uint64_t i = 0; i < view.token_id_count; ++i) {
        if (i)
            putchar(',');
        printf("%" PRId32, view.token_ids[i]);
    }
    printf("],\"setup_ms\":%.17g,\"prefill_ms\":%.17g,\"decode_ms\":%.17g}\n",
           view.setup_ms, view.prefill_ms, view.decode_ms);
    fflush(stdout);
    status = ferror(stdout) ? 1 : 0;
finish:
    if (file)
        fclose(file);
    free(ids);
    free(entries);
    if (result)
        core->result_release(result);
    if (model)
        core->model_release(model);
    if (error) {
        fprintf(stderr, "trtmc error: %s\n", core->error_message(error));
        core->error_release(error);
    }
    return status;
}

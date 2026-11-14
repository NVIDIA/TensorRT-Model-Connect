/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C API consumer for the official PatchTSMixer checkpoint. */
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/numeric.h>
#include <trtmc/trtmc.h>

_Static_assert(sizeof(float) == 4, "input uses float32");

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
static void json_strings(trtmc_strings_view values) {
    uint64_t i;
    putchar('[');
    for (i = 0; i < values.size; ++i) {
        if (i)
            putchar(',');
        json_string(values.data[i]);
    }
    putchar(']');
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_series_to_point_forecast_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_series_request_v1 request = {0};
    trtmc_point_forecast_view_v1 view = {0};
    float* input = NULL;
    FILE* file = NULL;
    long input_bytes;
    size_t input_count = 0;
    uint64_t i;
    int status = 1;
    if (argc != 4) {
        fprintf(stderr, "Usage: %s BUNDLE RUNTIME_ROOT VALUES_F32\n", argv[0]);
        return 2;
    }
    file = fopen(argv[3], "rb");
    if (!file || fseek(file, 0, SEEK_END) != 0 || (input_bytes = ftell(file)) <= 0 ||
        (unsigned long)input_bytes % (7 * sizeof(float)) != 0 || fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "input must contain complete seven-channel float32 timesteps\n");
        goto cleanup;
    }
    input_count = (size_t)input_bytes / sizeof(float);
    input = (float*)malloc((size_t)input_bytes);
    if (!input || fread(input, sizeof(float), input_count, file) != input_count ||
        fgetc(file) != EOF || ferror(file)) {
        fprintf(stderr, "unable to read the complete float32 history\n");
        goto cleanup;
    }
    fclose(file);
    file = NULL;
    if (trtmc_get_api(1, 0, &core) != TRTMC_OK || !core) {
        fprintf(stderr, "unable to obtain the v1 C API\n");
        goto cleanup;
    }
    options.struct_size = sizeof(options);
    options.runtime_root = text(argv[2]);
    if (core->model_load(text(argv[1]), &options, &model, &error) != TRTMC_OK)
        goto cleanup;
    if (core->model_get_task_api(model, text(TRTMC_TASK_SERIES_TO_POINT_FORECAST), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*task)) {
        fprintf(stderr, "incomplete forecast C table\n");
        goto cleanup;
    }
    task = (const trtmc_series_to_point_forecast_api_v1*)header;
    request.past_values = (trtmc_f32_matrix_view_v1){input, input_count, input_count / 7, 7};
    if (task->run(model, &request, NULL, &result, &error) != TRTMC_OK)
        goto cleanup;
    /* The synchronous call no longer borrows input; results outlive model handles. */
    free(input);
    input = NULL;
    core->model_release(model);
    model = NULL;
    if (task->result_view(result, &view, &error) != TRTMC_OK)
        goto cleanup;
    if (view.values.rows != 96 || view.values.columns != 7 || view.values.count != 96 * 7 ||
        view.axes.horizon_steps.size != 96) {
        fprintf(stderr, "forecast must retain the exact [96,7] horizon/channel layout\n");
        goto cleanup;
    }
    for (i = 0; i < view.values.count; ++i) {
        if (!isfinite(view.values.data[i])) {
            fprintf(stderr, "forecast contains a nonfinite value\n");
            goto cleanup;
        }
    }
    printf("{\"task\":\"series_to_point_forecast\",\"input_shape\":[%zu,7],"
           "\"shape\":[96,7],\"axes\":[\"horizon\",\"channel\"],\"horizon_steps\":[",
           input_count / 7);
    for (i = 0; i < view.axes.horizon_steps.size; ++i) {
        if (i)
            putchar(',');
        printf("%" PRId64, view.axes.horizon_steps.data[i]);
    }
    printf("],\"channel_names\":");
    json_strings(view.axes.channel_names);
    printf(",\"channel_units\":");
    json_strings(view.axes.channel_units);
    printf(",\"values\":[");
    for (i = 0; i < view.values.count; ++i) {
        if (i)
            putchar(',');
        printf("%.*g", FLT_DECIMAL_DIG, (double)view.values.data[i]);
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
    if (file)
        fclose(file);
    free(input);
    return status;
}

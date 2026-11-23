/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Real-checkpoint consumer of the public C SDK, including error and lifetime paths. */
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/trtmc.h>

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}

static float* read_f32(const char* path, uint64_t* count) {
    FILE* file = fopen(path, "rb");
    long bytes;
    float* values = NULL;
    *count = 0;
    if (file && fseek(file, 0, SEEK_END) == 0 && (bytes = ftell(file)) > 0 &&
        bytes % (long)sizeof(float) == 0 && fseek(file, 0, SEEK_SET) == 0) {
        values = (float*)malloc((size_t)bytes);
        if (values && fread(values, 1, (size_t)bytes, file) == (size_t)bytes &&
            fgetc(file) == EOF && !ferror(file))
            *count = (uint64_t)bytes / sizeof(float);
    }
    if (file)
        fclose(file);
    if (!*count) {
        free(values);
        values = NULL;
    }
    return values;
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_series_to_point_forecast_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result *result = NULL, *second = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_series_request_v1 request = {0}, invalid = {0};
    trtmc_point_forecast_view_v1 view = {0}, retained = {0};
    trtmc_config_entry_v1 entry = {0};
    trtmc_config_view_v1 config = {&entry, 1};
    float *input = NULL, *mask = NULL;
    uint8_t* observed = NULL;
    uint64_t count = 0, mask_count = 0, i;
    uint64_t field_count = 0;
    trtmc_config_field_v1 field = {0};
    char* end = NULL;
    long frequency;
    int status = 1;
    _Static_assert(sizeof(float) == 4, "input uses float32");
    if (argc != 6) {
        fprintf(stderr, "Usage: %s BUNDLE RUNTIME_ROOT VALUES_F32 MASK_F32 FREQUENCY\n", argv[0]);
        return 2;
    }
    frequency = strtol(argv[5], &end, 10);
    if (!end || *end || frequency < 0 || frequency > INT32_MAX)
        return 2;
    input = read_f32(argv[3], &count);
    mask = read_f32(argv[4], &mask_count);
    if (!input || !mask || count != mask_count)
        goto cleanup;
    observed = (uint8_t*)malloc((size_t)count);
    if (!observed)
        goto cleanup;
    for (i = 0; i < count; ++i) {
        if (mask[i] != 0 && mask[i] != 1)
            goto cleanup;
        observed[i] = mask[i] == 1;
    }
    if (trtmc_get_api(1, 0, &core) != TRTMC_OK || !core)
        goto cleanup;
    options.struct_size = sizeof(options);
    options.runtime_root = text(argv[2]);
    if (core->model_load(text(argv[1]), &options, &model, &error) != TRTMC_OK)
        goto cleanup;
    if (core->model_get_task_api(model, text(TRTMC_TASK_SERIES_TO_POINT_FORECAST), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->major != 1 || header->minor != 0 || header->byte_size < sizeof(*task))
        goto cleanup;
    task = (const trtmc_series_to_point_forecast_api_v1*)header;
    if (core->config_field_count(model, text(TRTMC_TASK_SERIES_TO_POINT_FORECAST), 1, 0,
                                 &field_count, &error) != TRTMC_OK ||
        field_count != 1 ||
        core->config_field_info(model, text(TRTMC_TASK_SERIES_TO_POINT_FORECAST), 1, 0, 0, &field,
                                &error) != TRTMC_OK ||
        field.name.size != 9 || memcmp(field.name.data, "frequency", 9) ||
        field.kind != TRTMC_CONFIG_I64 || !field.has_fixed_default ||
        field.default_value.as.i64 != 0)
        goto cleanup;
    entry.name = text("frequency");
    entry.value.kind = TRTMC_CONFIG_I64;
    entry.value.as.i64 = frequency;
    request.past_values = (trtmc_f32_matrix_view_v1){input, count, count, 1};
    request.observed = observed;
    request.observed_count = count;
    /* Invalid input must fail without returning a result; the model remains usable. */
    if (task->run(model, &invalid, &config, &result, &error) != TRTMC_INVALID_ARGUMENT || result)
        goto cleanup;
    core->error_release(error);
    error = NULL;
    entry.value.kind = TRTMC_CONFIG_BOOL;
    entry.value.as.boolean = 1;
    if (task->run(model, &request, &config, &result, &error) != TRTMC_INVALID_CONFIG || result)
        goto cleanup;
    core->error_release(error);
    error = NULL;
    entry.value.kind = TRTMC_CONFIG_I64;
    entry.value.as.i64 = frequency;
    if (task->run(model, &request, &config, &result, &error) != TRTMC_OK ||
        task->result_view(result, &retained, &error) != TRTMC_OK)
        goto cleanup;
    if (task->run(model, &request, &config, &second, &error) != TRTMC_OK ||
        task->result_view(second, &view, &error) != TRTMC_OK)
        goto cleanup;
    if (retained.values.count != view.values.count ||
        memcmp(retained.values.data, view.values.data, (size_t)view.values.count * sizeof(float)))
        goto cleanup;
    core->result_release(second);
    second = NULL;
    free(input);
    input = NULL;
    free(observed);
    observed = NULL;
    core->model_release(model);
    model = NULL;
    /* The result still owns its values and metadata after model/input release. */
    if (task->result_view(result, &view, &error) != TRTMC_OK || view.values.rows == 0 ||
        view.values.columns != 1 || view.values.count != view.values.rows ||
        view.axes.horizon_steps.size != view.values.rows || view.axes.channel_names.size ||
        view.axes.channel_units.size)
        goto cleanup;
    printf("{\"task\":\"series_to_point_forecast\",\"kind\":\"point\","
           "\"axes\":[\"horizon\",\"channel\"],\"shape\":[%" PRIu64 ",1],"
           "\"channel_names\":[],\"channel_units\":[],\"horizon_steps\":[",
           view.values.rows);
    for (i = 0; i < view.axes.horizon_steps.size; ++i) {
        if (view.axes.horizon_steps.data[i] != (int64_t)i + 1)
            goto cleanup;
        printf("%s%" PRId64, i ? "," : "", view.axes.horizon_steps.data[i]);
    }
    printf("],\"values\":[");
    for (i = 0; i < view.values.count; ++i) {
        if (!isfinite(view.values.data[i]))
            goto cleanup;
        printf("%s%.*g", i ? "," : "", FLT_DECIMAL_DIG, (double)view.values.data[i]);
    }
    puts("]}");
    status = fflush(stdout) == 0 && !ferror(stdout) ? 0 : 1;
cleanup:
    if (error && core) {
        const trtmc_string_view message = core->error_message(error);
        fwrite(message.data, 1, (size_t)message.size, stderr);
        fputc('\n', stderr);
        core->error_release(error);
    }
    if (core) {
        core->result_release(second);
        core->result_release(result);
        core->model_release(model);
    }
    free(input);
    free(mask);
    free(observed);
    if (status)
        fprintf(stderr, "TimesFM C SDK contract failed\n");
    return status;
}

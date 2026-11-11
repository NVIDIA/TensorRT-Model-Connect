/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/trtmc.h>

typedef trtmc_status(TRTMC_CALL* Run)(trtmc_model*, const trtmc_series_request_v1*,
                                      const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
typedef union {
    trtmc_point_forecast_view_v1 point;
    trtmc_regression_values_view_v1 values;
    trtmc_regression_distribution_view_v1 distribution;
} View;

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}
static float* read_f32(const char* path, uint64_t* count) {
    FILE* file = fopen(path, "rb");
    float* values = NULL;
    long bytes;
    *count = 0;
    if (file && fseek(file, 0, SEEK_END) == 0 && (bytes = ftell(file)) > 0 &&
        bytes % (long)sizeof(float) == 0 && fseek(file, 0, SEEK_SET) == 0) {
        values = (float*)malloc((size_t)bytes);
        if (values && fread(values, 1, (size_t)bytes, file) == (size_t)bytes && !ferror(file))
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
static trtmc_status view_result(int mode, const trtmc_api_header* api, const trtmc_result* result,
                                View* view, trtmc_error** error) {
    if (mode == 1)
        return ((const trtmc_series_to_point_forecast_api_v1*)api)
            ->result_view(result, &view->point, error);
    if (mode == 2)
        return ((const trtmc_series_to_regression_values_api_v1*)api)
            ->result_view(result, &view->values, error);
    return ((const trtmc_series_to_regression_distribution_api_v1*)api)
        ->result_view(result, &view->distribution, error);
}
static int equal_values(const float* a, const float* b, uint64_t count) {
    return memcmp(a, b, (size_t)count * sizeof(float)) == 0;
}
static int same(int mode, const View* a, const View* b) {
    uint64_t i;
    if (mode == 1)
        return a->point.values.count == b->point.values.count &&
               a->point.values.rows == b->point.values.rows &&
               a->point.values.columns == b->point.values.columns &&
               equal_values(a->point.values.data, b->point.values.data, a->point.values.count);
    if (mode == 2)
        return a->values.values.size == b->values.values.size &&
               equal_values(a->values.values.data, b->values.values.data, a->values.values.size);
    if (a->distribution.target_count != b->distribution.target_count ||
        a->distribution.distribution != b->distribution.distribution ||
        a->distribution.parameter_count != b->distribution.parameter_count)
        return 0;
    for (i = 0; i < a->distribution.parameter_count; ++i) {
        const trtmc_distribution_parameter_v1* left = &a->distribution.parameters[i];
        const trtmc_distribution_parameter_v1* right = &b->distribution.parameters[i];
        if (left->target_count != right->target_count || left->name.size != right->name.size ||
            memcmp(left->name.data, right->name.data, (size_t)left->name.size) ||
            !equal_values(left->values, right->values, left->target_count))
            return 0;
    }
    return 1;
}
static int array(const float* values, uint64_t count) {
    uint64_t i;
    putchar('[');
    for (i = 0; i < count; ++i) {
        if (!isfinite(values[i]))
            return 0;
        printf("%s%.*g", i ? "," : "", FLT_DECIMAL_DIG, (double)values[i]);
    }
    putchar(']');
    return 1;
}
static int emit(int mode, const View* data) {
    uint64_t i;
    if (mode == 1) {
        const trtmc_point_forecast_view_v1* view = &data->point;
        if (!view->values.rows || !view->values.columns ||
            view->values.count != view->values.rows * view->values.columns ||
            view->axes.horizon_steps.size != view->values.rows || view->axes.channel_names.size ||
            view->axes.channel_units.size)
            return 0;
        printf("\"kind\":\"point\",\"shape\":[%" PRIu64 ",%" PRIu64
               "],\"axes\":[\"horizon\",\"channel\"],\"channel_names\":[],\"channel_units\":[],"
               "\"horizon_steps\":[",
               view->values.rows, view->values.columns);
        for (i = 0; i < view->axes.horizon_steps.size; ++i) {
            if (view->axes.horizon_steps.data[i] != (int64_t)i + 1)
                return 0;
            printf("%s%" PRId64, i ? "," : "", view->axes.horizon_steps.data[i]);
        }
        printf("],\"values\":");
        return array(view->values.data, view->values.count);
    }
    if (mode == 2) {
        const trtmc_regression_values_view_v1* view = &data->values;
        if (!view->values.size || view->target_names.size || view->target_units.size)
            return 0;
        printf("\"kind\":\"regression_values\",\"target_count\":%" PRIu64
               ",\"target_names\":[],\"target_units\":[],\"axes\":[\"target\"],\"values\":",
               view->values.size);
        return array(view->values.data, view->values.size);
    }
    {
        const trtmc_regression_distribution_view_v1* view = &data->distribution;
        const char* kind = view->distribution == TRTMC_DISTRIBUTION_NORMAL      ? "normal"
                           : view->distribution == TRTMC_DISTRIBUTION_STUDENT_T ? "student_t"
                           : view->distribution == TRTMC_DISTRIBUTION_NEGATIVE_BINOMIAL
                               ? "negative_binomial"
                               : NULL;
        if (!kind || !view->target_count || !view->parameter_count || view->target_names.size ||
            view->target_units.size)
            return 0;
        printf(
            "\"kind\":\"regression_distribution\",\"distribution\":\"%s\",\"target_count\":%" PRIu64
            ",\"target_names\":[],\"target_units\":[],\"parameters\":{",
            kind, view->target_count);
        for (i = 0; i < view->parameter_count; ++i) {
            const trtmc_distribution_parameter_v1* item = &view->parameters[i];
            if (item->target_count != view->target_count)
                return 0;
            printf("%s\"%.*s\":", i ? "," : "", (int)item->name.size, item->name.data);
            if (!array(item->values, item->target_count))
                return 0;
        }
        putchar('}');
        return 1;
    }
}
int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header *api = NULL, *unsupported = NULL;
    trtmc_model* model = NULL;
    trtmc_result *result = NULL, *second = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_series_request_v1 request = {0}, invalid = {0};
    trtmc_config_entry_v1 entry = {0};
    trtmc_config_view_v1 config = {&entry, 1};
    trtmc_config_field_v1 field = {0};
    View retained = {0}, repeated = {0}, current = {0};
    Run run = NULL;
    float *values = NULL, *mask = NULL;
    uint8_t* observed = NULL;
    uint64_t count = 0, mask_count = 0, fields = 0, i, channels;
    char* end = NULL;
    int mode = 0, status = 1;
    _Static_assert(sizeof(float) == 4, "float32 input");
    if (argc != 7)
        return 2;
    if (!strcmp(argv[5], TRTMC_TASK_SERIES_TO_POINT_FORECAST))
        mode = 1;
    else if (!strcmp(argv[5], TRTMC_TASK_SERIES_TO_REGRESSION_VALUES))
        mode = 2;
    else if (!strcmp(argv[5], TRTMC_TASK_SERIES_TO_REGRESSION_DISTRIBUTION))
        mode = 3;
    if (!mode)
        return 2;
    channels = strtoull(argv[6], &end, 10);
    if (end == argv[6] || *end || !channels)
        return 2;
    values = read_f32(argv[3], &count);
    mask = read_f32(argv[4], &mask_count);
    if (!values || !mask || mask_count != count || count % channels)
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
    if (core->model_load(text(argv[1]), &options, &model, &error) != TRTMC_OK ||
        core->model_get_task_api(model, text(argv[5]), 1, 0, &api, &error) != TRTMC_OK || !api ||
        api->major != 1 || api->minor != 0)
        goto cleanup;
    if (mode == 1) {
        if (api->byte_size < sizeof(trtmc_series_to_point_forecast_api_v1))
            goto cleanup;
        run = ((const trtmc_series_to_point_forecast_api_v1*)api)->run;
    } else if (mode == 2) {
        if (api->byte_size < sizeof(trtmc_series_to_regression_values_api_v1))
            goto cleanup;
        run = ((const trtmc_series_to_regression_values_api_v1*)api)->run;
    } else {
        if (api->byte_size < sizeof(trtmc_series_to_regression_distribution_api_v1))
            goto cleanup;
        run = ((const trtmc_series_to_regression_distribution_api_v1*)api)->run;
    }
    if (core->model_get_task_api(model,
                                 text(mode == 1 ? TRTMC_TASK_SERIES_TO_REGRESSION_VALUES
                                                : TRTMC_TASK_SERIES_TO_POINT_FORECAST),
                                 1, 0, &unsupported, &error) != TRTMC_UNSUPPORTED ||
        unsupported)
        goto cleanup;
    core->error_release(error);
    error = NULL;
    if (core->config_field_count(model, text(argv[5]), 1, 0, &fields, &error) != TRTMC_OK ||
        fields != 1 ||
        core->config_field_info(model, text(argv[5]), 1, 0, 0, &field, &error) != TRTMC_OK ||
        field.name.size != 9 || memcmp(field.name.data, "frequency", 9) ||
        field.kind != TRTMC_CONFIG_I64 || !field.has_fixed_default ||
        field.default_value.as.i64 != 0)
        goto cleanup;
    entry.name = text("frequency");
    entry.value.kind = TRTMC_CONFIG_I64;
    entry.value.as.i64 = 0;
    request.past_values = (trtmc_f32_matrix_view_v1){values, count, 0, 0};
    request.observed = observed;
    request.observed_count = count;
    if (run(model, &invalid, &config, &result, &error) != TRTMC_INVALID_ARGUMENT || result)
        goto cleanup;
    core->error_release(error);
    error = NULL;
    entry.value.kind = TRTMC_CONFIG_BOOL;
    entry.value.as.boolean = 1;
    if (run(model, &request, &config, &result, &error) != TRTMC_INVALID_CONFIG || result)
        goto cleanup;
    core->error_release(error);
    error = NULL;
    entry.value.kind = TRTMC_CONFIG_I64;
    entry.value.as.i64 = 0;
    if (run(model, &request, &config, &result, &error) != TRTMC_OK ||
        view_result(mode, api, result, &retained, &error) != TRTMC_OK ||
        run(model, &request, &config, &second, &error) != TRTMC_OK ||
        view_result(mode, api, second, &repeated, &error) != TRTMC_OK ||
        !same(mode, &retained, &repeated))
        goto cleanup;
    core->result_release(second);
    second = NULL;
    free(values);
    values = NULL;
    free(observed);
    observed = NULL;
    core->model_release(model);
    model = NULL;
    if (view_result(mode, api, result, &current, &error) != TRTMC_OK ||
        !same(mode, &retained, &current))
        goto cleanup;
    printf("{\"task\":\"%s\",", argv[5]);
    if (!emit(mode, &current))
        goto cleanup;
    puts("}");
    status = fflush(stdout) == 0 && !ferror(stdout) ? 0 : 1;
cleanup:
    if (core) {
        if (error) {
            trtmc_string_view message = core->error_message(error);
            fwrite(message.data, 1, (size_t)message.size, stderr);
            fputc('\n', stderr);
            core->error_release(error);
        }
        core->result_release(second);
        core->result_release(result);
        core->model_release(model);
    }
    free(values);
    free(mask);
    free(observed);
    if (status)
        fprintf(stderr, "PatchTST C SDK contract failed\n");
    return status;
}

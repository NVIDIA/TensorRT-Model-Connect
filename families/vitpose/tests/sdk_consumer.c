/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C SDK consumer for the family-owned end-to-end test: the body keypoints of one person in
 * a float32 RGB image, optionally inside a person box. Prints one JSON line to stdout. */
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/perception.h>
#include <trtmc/trtmc.h>

_Static_assert(sizeof(float) == 4, "input uses float32");

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}
static uint32_t dimension(const char* text_value) {
    char* end = NULL;
    uintmax_t value;
    errno = 0;
    value = strtoumax(text_value, &end, 10);
    if (errno || end == text_value || *end || !value || value > INT32_MAX)
        return 0;
    return (uint32_t)value;
}
static int coordinate(const char* text_value, double* value) {
    char* end = NULL;
    errno = 0;
    *value = strtod(text_value, &end);
    return !errno && end != text_value && !*end && isfinite(*value);
}
static int same(trtmc_string_view value, const char* expected) {
    return value.size == strlen(expected) && !memcmp(value.data, expected, (size_t)value.size);
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
    const trtmc_image_text_to_points_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_image_query_request_v1 request = {0};
    trtmc_grounded_points_view_v1 view = {0};
    trtmc_config_field_v1 field = {0};
    trtmc_config_entry_v1 entry = {0};
    trtmc_config_view_v1 config = {0};
    double box[4];
    float* input = NULL;
    FILE* file = NULL;
    uint32_t height, width;
    size_t count;
    uint64_t i, field_count = 0;
    int status = 1;
    if (argc != 6 && argc != 10) {
        fprintf(stderr,
                "Usage: %s BUNDLE RUNTIME_ROOT RGB_F32 HEIGHT WIDTH [X_MIN Y_MIN X_MAX Y_MAX]\n",
                argv[0]);
        return 2;
    }
    height = dimension(argv[4]);
    width = dimension(argv[5]);
    if (!height || !width || (size_t)height > SIZE_MAX / width / 3 / sizeof(float)) {
        fprintf(stderr, "invalid or overflowing image dimensions\n");
        return 2;
    }
    for (i = 0; argc == 10 && i < 4; ++i) {
        if (!coordinate(argv[6 + i], &box[i])) {
            fprintf(stderr, "person box values must be finite numbers\n");
            return 2;
        }
    }
    count = (size_t)height * width * 3;
    input = (float*)malloc(count * sizeof(float));
    file = fopen(argv[3], "rb");
    if (!input || !file || fread(input, sizeof(float), count, file) != count ||
        fgetc(file) != EOF || ferror(file)) {
        fprintf(stderr, "input must contain exactly HEIGHT x WIDTH x 3 float32 RGB values\n");
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
    if (core->model_get_task_api(model, text(TRTMC_TASK_IMAGE_TEXT_TO_POINTS), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*task)) {
        fprintf(stderr, "incomplete image-text-to-points C table\n");
        goto cleanup;
    }
    task = (const trtmc_image_text_to_points_api_v1*)header;
    if (core->config_field_count(model, text(TRTMC_TASK_IMAGE_TEXT_TO_POINTS), 1, 0, &field_count,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (field_count != 1 ||
        core->config_field_info(model, text(TRTMC_TASK_IMAGE_TEXT_TO_POINTS), 1, 0, 0, &field,
                                &error) != TRTMC_OK ||
        !same(field.name, "person_box") || field.kind != TRTMC_CONFIG_F64_LIST ||
        field.has_fixed_default) {
        if (!error)
            fprintf(stderr, "ViTPose must expose exactly the person_box F64 list field\n");
        goto cleanup;
    }
    request.image =
        (trtmc_image_input_v1){input, count * sizeof(float), height, width, 3, TRTMC_IMAGE_FLOAT32};
    request.text = text("person");
    if (argc == 10) {
        entry.name = text("person_box");
        entry.value.kind = TRTMC_CONFIG_F64_LIST;
        entry.value.as.f64_list.data = box;
        entry.value.as.f64_list.size = 4;
        config.entries = &entry;
        config.count = 1;
    }
    if (task->run(model, &request, argc == 10 ? &config : NULL, &result, &error) != TRTMC_OK)
        goto cleanup;
    free(input);
    input = NULL;
    core->model_release(model);
    model = NULL;
    if (task->result_view(result, &view, &error) != TRTMC_OK)
        goto cleanup;
    for (i = 0; i < view.count; ++i) {
        if (!view.points[i].has_confidence) {
            fprintf(stderr, "every keypoint must carry its heatmap score\n");
            goto cleanup;
        }
    }
    printf("{\"task\":\"image_text_to_points\",\"points\":[");
    for (i = 0; i < view.count; ++i)
        printf("%s%.9g,%.9g", i ? "," : "", (double)view.points[i].point.x,
               (double)view.points[i].point.y);
    printf("],\"labels\":[");
    for (i = 0; i < view.count; ++i) {
        if (i)
            putchar(',');
        json_string(view.points[i].label);
    }
    printf("],\"scores\":[");
    for (i = 0; i < view.count; ++i)
        printf("%s%.9g", i ? "," : "", (double)view.points[i].confidence);
    printf("],\"raw_response\":");
    json_string(view.raw_response);
    printf(",\"parse_complete\":%s}\n", view.parse_complete ? "true" : "false");
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

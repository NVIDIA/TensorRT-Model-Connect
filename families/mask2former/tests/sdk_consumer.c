/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C SDK consumer for the family-owned end-to-end test: semantic segmentation of one float32
 * RGB image. Writes the int32 label map to OUT_LABELS and one JSON summary line to stdout. */
#include <errno.h>
#include <inttypes.h>
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
    const trtmc_image_to_semantic_segmentation_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_perception_image_request_v1 request = {0};
    trtmc_semantic_segmentation_view_v1 view = {0};
    float* input = NULL;
    FILE* file = NULL;
    uint32_t height, width;
    size_t count;
    uint64_t i, field_count = 0;
    int status = 1;
    if (argc != 7) {
        fprintf(stderr, "Usage: %s BUNDLE RUNTIME_ROOT RGB_F32 HEIGHT WIDTH OUT_LABELS\n", argv[0]);
        return 2;
    }
    height = dimension(argv[4]);
    width = dimension(argv[5]);
    if (!height || !width || (size_t)height > SIZE_MAX / width / 3 / sizeof(float)) {
        fprintf(stderr, "invalid or overflowing image dimensions\n");
        return 2;
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
    if (core->model_get_task_api(model, text(TRTMC_TASK_IMAGE_TO_SEMANTIC_SEGMENTATION), 1, 0,
                                 &header, &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*task)) {
        fprintf(stderr, "incomplete semantic segmentation C table\n");
        goto cleanup;
    }
    task = (const trtmc_image_to_semantic_segmentation_api_v1*)header;
    if (core->config_field_count(model, text(TRTMC_TASK_IMAGE_TO_SEMANTIC_SEGMENTATION), 1, 0,
                                 &field_count, &error) != TRTMC_OK)
        goto cleanup;
    if (field_count != 0) {
        fprintf(stderr, "Mask2Former must expose no runtime Config fields\n");
        goto cleanup;
    }
    request.image =
        (trtmc_image_input_v1){input, count * sizeof(float), height, width, 3, TRTMC_IMAGE_FLOAT32};
    if (task->run(model, &request, NULL, &result, &error) != TRTMC_OK)
        goto cleanup;
    free(input);
    input = NULL;
    core->model_release(model);
    model = NULL;
    if (task->result_view(result, &view, &error) != TRTMC_OK)
        goto cleanup;
    if (view.height != height || view.width != width ||
        view.pixel_count != (uint64_t)height * width || !view.class_ids.size) {
        fprintf(stderr, "semantic segmentation must label every pixel of the input\n");
        goto cleanup;
    }
    file = fopen(argv[6], "wb");
    if (!file ||
        fwrite(view.labels, sizeof(int32_t), (size_t)view.pixel_count, file) !=
            (size_t)view.pixel_count ||
        fclose(file) != 0) {
        file = NULL;
        fprintf(stderr, "unable to write the label map\n");
        goto cleanup;
    }
    file = NULL;
    printf("{\"task\":\"image_to_semantic_segmentation\",\"height\":%" PRIu32 ",\"width\":%" PRIu32
           ",\"pixel_count\":%" PRIu64 ",\"class_ids\":[",
           view.height, view.width, view.pixel_count);
    for (i = 0; i < view.class_ids.size; ++i) {
        if (i)
            putchar(',');
        printf("%" PRId32, view.class_ids.data[i]);
    }
    printf("],\"class_names\":[");
    for (i = 0; i < view.class_names.size; ++i) {
        if (i)
            putchar(',');
        json_string(view.class_names.data[i]);
    }
    printf("],\"vocabulary_id\":");
    json_string(view.vocabulary_id);
    printf(",\"has_ignore_label\":%" PRIu32 ",\"has_background_label\":%" PRIu32
           ",\"score_count\":%" PRIu64 "}\n",
           view.has_ignore_label, view.has_background_label, view.score_count);
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

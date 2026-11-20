/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C11 SDK consumer for the family-owned OneFormer end-to-end test.
 * Reads interleaved RGB float32 in [0, 1], runs image_to_semantic_segmentation,
 * releases the model, then prints the label map's per-class counts and its
 * FNV-1a 64 digest (over little-endian int32 labels) as one JSON line. */
#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
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

static uint32_t dimension(const char* value) {
    char* end = NULL;
    uintmax_t parsed;
    errno = 0;
    parsed = strtoumax(value, &end, 10);
    if (errno || end == value || *end || !parsed || parsed > INT32_MAX)
        return 0;
    return (uint32_t)parsed;
}

static void print_string(trtmc_string_view value) {
    uint64_t i;
    putchar('"');
    for (i = 0; i < value.size; ++i) {
        const char c = value.data[i];
        if (c == '"' || c == '\\')
            putchar('\\');
        putchar(c);
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
    uint64_t* counts = NULL;
    FILE* file = NULL;
    uint32_t height, width;
    size_t count;
    uint64_t i, field_count = 0, digest = 1469598103934665603ULL;
    int status = 1;

    if (argc != 6) {
        fprintf(stderr, "Usage: %s BUNDLE RUNTIME_ROOT RGB_F32 HEIGHT WIDTH\n", argv[0]);
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
        fprintf(stderr, "incomplete image_to_semantic_segmentation C table\n");
        goto cleanup;
    }
    task = (const trtmc_image_to_semantic_segmentation_api_v1*)header;
    if (core->config_field_count(model, text(TRTMC_TASK_IMAGE_TO_SEMANTIC_SEGMENTATION), 1, 0,
                                 &field_count, &error) != TRTMC_OK)
        goto cleanup;
    if (field_count != 0) {
        fprintf(stderr, "OneFormer must expose no runtime Config fields\n");
        goto cleanup;
    }

    request.image =
        (trtmc_image_input_v1){input, count * sizeof(float), height, width, 3, TRTMC_IMAGE_FLOAT32};
    if (task->run(model, &request, NULL, &result, &error) != TRTMC_OK)
        goto cleanup;
    /* The result must stay readable after its input and model are gone. */
    free(input);
    input = NULL;
    core->model_release(model);
    model = NULL;

    if (task->result_view(result, &view, &error) != TRTMC_OK)
        goto cleanup;
    if (view.height != height || view.width != width ||
        view.pixel_count != (uint64_t)height * width || !view.labels || !view.class_ids.size ||
        view.class_names.size != view.class_ids.size) {
        fprintf(stderr, "semantic result does not describe the input image\n");
        goto cleanup;
    }
    counts = (uint64_t*)calloc((size_t)view.class_ids.size, sizeof(uint64_t));
    if (!counts)
        goto cleanup;
    for (i = 0; i < view.pixel_count; ++i) {
        const int32_t label = view.labels[i];
        const uint32_t bits = (uint32_t)label;
        int byte;
        if (label < 0 || (uint64_t)label >= view.class_ids.size ||
            view.class_ids.data[label] != label) {
            fprintf(stderr, "label %" PRId32 " is not one of the result's class ids\n", label);
            goto cleanup;
        }
        ++counts[label];
        for (byte = 0; byte < 4; ++byte) {
            digest ^= (bits >> (8 * byte)) & 0xFFU;
            digest *= 1099511628211ULL;
        }
    }

    printf("{\"task\":\"image_to_semantic_segmentation\",\"height\":%" PRIu32 ",\"width\":%" PRIu32
           ",\"class_names\":[",
           view.height, view.width);
    for (i = 0; i < view.class_names.size; ++i) {
        if (i)
            putchar(',');
        print_string(view.class_names.data[i]);
    }
    fputs("],\"counts\":[", stdout);
    for (i = 0; i < view.class_ids.size; ++i)
        printf("%s%" PRIu64, i ? "," : "", counts[i]);
    printf("],\"fnv1a64\":\"%016" PRIx64 "\",\"has_scores\":%s}\n", digest,
           view.score_count ? "true" : "false");
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
    free(counts);
    free(input);
    return status;
}

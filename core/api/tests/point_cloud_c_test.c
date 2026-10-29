/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/point_cloud.h"
#include "trtmc/trtmc.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(offsetof(trtmc_points_to_semantic_segmentation_api_v1, header) == 0,
               "point-cloud table begins with the ABI header");

static const trtmc_core_api_v1* core;
static int failures;
static trtmc_string_view text(const char* value) {
    const trtmc_string_view out = {value, (uint64_t)strlen(value)};
    return out;
}
static void check(int condition, const char* message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}
static void checked(trtmc_status actual, trtmc_status expected, trtmc_error** error,
                    const char* message) {
    check(actual == expected, message);
    check(actual == TRTMC_OK ? *error == NULL : *error != NULL, "C error ownership is consistent");
    core->error_release(*error);
    *error = NULL;
}
static trtmc_model* load(const char* root, const char* mode) {
    const unsigned char magic[8] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    char path[4096], header[512];
    int size;
    unsigned shift;
    FILE* file;
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    if (snprintf(path, sizeof(path), "%s/point-cloud-c-%s.bundle", root, mode) >= (int)sizeof(path))
        exit(2);
    size = snprintf(header, sizeof(header),
                    "{\"format\":1,\"family\":\"point_cloud_fixture\",\"task\":\"%s\","
                    "\"backend\":\"fake\",\"sections\":{}}",
                    mode);
    if (size < 0 || (size_t)size >= sizeof(header) || !(file = fopen(path, "wb")))
        exit(2);
    if (fwrite(magic, 1, sizeof(magic), file) != sizeof(magic))
        exit(2);
    for (shift = 0; shift < 64; shift += 8)
        if (fputc((int)(((uint64_t)size >> shift) & 255), file) == EOF)
            exit(2);
    if (fwrite(header, 1, (size_t)size, file) != (size_t)size || fclose(file))
        exit(2);
    options.struct_size = sizeof(options);
    options.runtime_root = text(root);
    checked(core->model_load(text(path), &options, &model, &error), TRTMC_OK, &error,
            "C point-cloud fixture loads");
    return model;
}

int main(int argc, char** argv) {
    const float points[6] = {1, 2, 2, 3, 2, 2};
    trtmc_model* model;
    const trtmc_api_header* header = NULL;
    const trtmc_points_to_semantic_segmentation_api_v1* task;
    trtmc_error* error = NULL;
    trtmc_result *result = NULL, *other = NULL;
    trtmc_points_to_semantic_segmentation_view_v1 view = {0};
    trtmc_points_to_semantic_segmentation_request_v1 request = {points, 2, 3};
    trtmc_points_to_semantic_segmentation_request_v1 invalid;
    if (argc != 2 || trtmc_get_api(1, 0, &core) != TRTMC_OK)
        return 2;
    model = load(argv[1], "points_to_semantic_segmentation");
    if (!model)
        return 2;
    checked(core->model_get_task_api(model, text(TRTMC_TASK_POINTS_TO_SEMANTIC_SEGMENTATION), 1, 0,
                                     &header, &error),
            TRTMC_OK, &error, "C point-cloud table lookup");
    if (!header)
        return 2;
    task = (const trtmc_points_to_semantic_segmentation_api_v1*)header;
    check(header->major == 1 && header->minor == 0 && header->byte_size == sizeof(*task),
          "C complete versioned point-cloud table");
    checked(task->run(model, &request, NULL, &result, &error), TRTMC_OK, &error,
            "C point-cloud request");
    checked(task->result_view(result, &view, &error), TRTMC_OK, &error, "C point-cloud view");
    check(view.num_points == 2 && view.labels[0] == 1 && view.labels[1] == 3 &&
              view.num_classes == 4 && view.class_score_count == 8,
          "C labels and scores follow num_points rows of input_dim floats");
    invalid = request;
    invalid.points = NULL;
    checked(task->run(model, &invalid, NULL, &other, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C nonempty NULL points rejects before dereference");
    invalid = request;
    invalid.input_dim = 0;
    checked(task->run(model, &invalid, NULL, &other, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C zero input_dim rejects");
    invalid = request;
    invalid.num_points = (uint64_t)UINT32_MAX + 3;
    checked(task->run(model, &invalid, NULL, &other, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C num_points beyond uint32 rejects instead of truncating");
    check(other == NULL, "C rejected calls never return a result");
    checked(task->run(model, NULL, NULL, &other, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C NULL request rejects");
    core->model_release(model);
    checked(task->result_view(result, &view, &error), TRTMC_OK, &error,
            "C point-cloud result survives model release");
    check(view.labels[1] == 3, "C retained labels remain owned by the result");
    core->result_release(result);
    puts("C point cloud: labels follow num_points x input_dim rows");
    return failures ? 1 : 0;
}

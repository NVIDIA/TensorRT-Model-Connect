/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/features.h"
#include "trtmc/trtmc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const trtmc_core_api_v1* core;
static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, strlen(value)};
    return result;
}
static void checked(trtmc_status status, trtmc_error* error) {
    if (status != TRTMC_OK) {
        const trtmc_string_view message = core->error_message(error);
        fprintf(stderr, "BEVFusion C API failed: %.*s\n", (int)message.size, message.data);
        core->error_release(error);
        exit(1);
    }
}
static float* read_values(const char* path, size_t count) {
    FILE* file = fopen(path, "rb");
    float* result = (float*)malloc(count * sizeof(float));
    if (!file || !result || fread(result, sizeof(float), count, file) != count ||
        fgetc(file) != EOF || fclose(file)) {
        fprintf(stderr, "BEVFusion C input has an invalid length: %s\n", path);
        exit(1);
    }
    return result;
}
int main(int argc, char** argv) {
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    const trtmc_api_header* header = NULL;
    trtmc_spatial_class_scores_view_v1 view = {0};
    trtmc_load_options_v1 options = {0};
    if (argc != 6 || trtmc_get_api(1, 0, &core) != TRTMC_OK)
        return 2;
    float* camera = read_values(argv[3], 80 * 128 * 128);
    float* lidar = read_values(argv[4], 256 * 128 * 128);
    const trtmc_spatial_feature_input_v1 inputs[] = {
        {text("camera_bev"), {camera, 80 * 128 * 128}, 80, 128, 128},
        {text("lidar_bev"), {lidar, 256 * 128 * 128}, 256, 128, 128}};
    const trtmc_spatial_features_to_class_scores_request_v1 request = {inputs, 2};
    options.struct_size = sizeof(options);
    options.runtime_root = text(argv[2]);
    trtmc_status status = core->model_load(text(argv[1]), &options, &model, &error);
    checked(status, error);
    status = core->model_get_task_api(model, text(TRTMC_TASK_SPATIAL_FEATURES_TO_CLASS_SCORES), 1,
                                      0, &header, &error);
    checked(status, error);
    if (!header || header->byte_size < sizeof(trtmc_spatial_features_to_class_scores_api_v1))
        return 1;
    const trtmc_spatial_features_to_class_scores_api_v1* task =
        (const trtmc_spatial_features_to_class_scores_api_v1*)header;
    status = task->run(model, &request, NULL, &result, &error);
    checked(status, error);
    // The C result and all metadata must survive destruction of the real family.
    core->model_release(model);
    status = task->result_view(result, &view, &error);
    checked(status, error);
    const char* names[] = {"drivable_area", "ped_crossing", "walkway",
                           "stop_line",     "carpark_area", "divider"};
    if (view.classes != 6 || view.height != 200 || view.width != 200 ||
        view.values.size != 240000 || !view.values.data ||
        view.score_kind != TRTMC_SCORE_PROBABILITY || view.class_names.size != 6)
        return 1;
    for (size_t index = 0; index < 6; ++index)
        if (view.class_names.data[index].size != strlen(names[index]) ||
            memcmp(view.class_names.data[index].data, names[index], strlen(names[index])))
            return 1;
    FILE* file = fopen(argv[5], "wb");
    if (!file ||
        fwrite(view.values.data, sizeof(float), view.values.size, file) != view.values.size ||
        fclose(file))
        return 1;
    core->result_release(result);
    free(camera);
    free(lidar);
    return 0;
}

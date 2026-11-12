/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * C SDK consumer for the chroma family: text_to_image through trtmc/trtmc.h
 * and trtmc/image.h, plus one rejected request.
 *
 * Usage: test_chroma_sdk_c <bundle_path> <runtime_root>
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/image.h>
#include <trtmc/trtmc.h>

static trtmc_string_view sv(const char* text) {
    trtmc_string_view view;
    view.data = text;
    view.size = strlen(text);
    return view;
}

static int report(trtmc_status status, const trtmc_core_api_v1* api, trtmc_error** error,
                  const char* label) {
    if (status == TRTMC_OK)
        return 0;
    if (api && *error) {
        trtmc_string_view message = api->error_message(*error);
        fprintf(stderr, "%s failed: %.*s\n", label, (int)message.size,
                message.data ? message.data : "");
        api->error_release(*error);
        *error = NULL;
    } else {
        fprintf(stderr, "%s failed\n", label);
    }
    return 1;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <bundle_path> <runtime_root>\n", argv[0]);
        return 2;
    }
    const trtmc_core_api_v1* api = NULL;
    trtmc_error* error = NULL;
    if (report(trtmc_get_api(1, 0, &api), NULL, &error, "trtmc_get_api"))
        return 1;

    trtmc_load_options_v1 options;
    memset(&options, 0, sizeof(options));
    options.struct_size = sizeof(options);
    options.runtime_root = sv(argv[2]);
    trtmc_model* model = NULL;
    if (report(api->model_load(sv(argv[1]), &options, &model, &error), api, &error, "model_load"))
        return 1;

    const trtmc_api_header* header = NULL;
    int failed =
        report(api->model_get_task_api(model, sv(TRTMC_TASK_TEXT_TO_IMAGE), 1, 0, &header, &error),
               api, &error, "get text_to_image");
    const trtmc_text_to_image_api_v1* task = (const trtmc_text_to_image_api_v1*)header;

    trtmc_config_entry_v1 entries[3];
    memset(entries, 0, sizeof(entries));
    entries[0].name = sv("num_inference_steps");
    entries[0].value.kind = TRTMC_CONFIG_I64;
    entries[0].value.as.i64 = 2;
    entries[1].name = sv("guidance_scale");
    entries[1].value.kind = TRTMC_CONFIG_F64;
    entries[1].value.as.f64 = 3.0;
    entries[2].name = sv("seed");
    entries[2].value.kind = TRTMC_CONFIG_I64;
    entries[2].value.as.i64 = 7;
    trtmc_config_view_v1 config = {entries, 3};

    trtmc_text_to_image_request_v1 request;
    memset(&request, 0, sizeof(request));
    request.prompt = sv("A red apple on a wooden table");

    trtmc_result* result = NULL;
    if (!failed)
        failed = report(task->run(model, &request, &config, &result, &error), api, &error,
                        "text_to_image run");
    if (!failed) {
        trtmc_image_result_view_v1 view;
        memset(&view, 0, sizeof(view));
        failed = report(task->result_view(result, &view, &error), api, &error, "result_view");
        if (!failed && (view.channels != 3 || view.height == 0 || view.width == 0 ||
                        view.pixel_count != (uint64_t)view.height * view.width * 3)) {
            fprintf(stderr, "text_to_image returned an invalid image shape\n");
            failed = 1;
        }
        for (uint64_t index = 0; !failed && index < view.pixel_count; ++index) {
            const float value = view.pixels[index];
            if (!isfinite(value) || value < 0.0f || value > 1.0f) {
                fprintf(stderr, "pixel %llu is outside [0, 1]\n", (unsigned long long)index);
                failed = 1;
            }
        }
        if (!failed)
            printf("text_to_image: %ux%u RGB\n", view.width, view.height);
        api->result_release(result);
    }

    /* The image size is fixed by the bundle: a different height is rejected. */
    if (!failed) {
        trtmc_config_entry_v1 wrong;
        memset(&wrong, 0, sizeof(wrong));
        wrong.name = sv("height");
        wrong.value.kind = TRTMC_CONFIG_I64;
        wrong.value.as.i64 = 64;
        trtmc_config_view_v1 rejected = {&wrong, 1};
        result = NULL;
        if (task->run(model, &request, &rejected, &result, &error) == TRTMC_OK) {
            fprintf(stderr, "a height other than the bundle's was accepted\n");
            api->result_release(result);
            failed = 1;
        } else if (error) {
            api->error_release(error);
            error = NULL;
        }
    }

    api->model_release(model);
    if (!failed)
        printf("chroma C SDK consumer: passed\n");
    return failed;
}

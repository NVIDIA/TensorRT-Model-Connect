/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * C SDK consumer for the stable_diffusion_3 family. Runs text_to_image through
 * the public C API (trtmc/trtmc.h + trtmc/image.h) with explicit config
 * entries and checks the returned image.
 *
 * Usage:
 *   test_stable_diffusion_3_sdk_c <bundle_path> <runtime_root>
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/image.h>
#include <trtmc/trtmc.h>

static void check(trtmc_status status, const trtmc_core_api_v1* api, trtmc_error** err,
                  const char* label) {
    if (status != TRTMC_OK) {
        if (api && *err) {
            trtmc_string_view msg = api->error_message(*err);
            fprintf(stderr, "%s failed: %.*s\n", label, (int)msg.size, msg.data ? msg.data : "");
            api->error_release(*err);
        } else {
            fprintf(stderr, "%s failed\n", label);
        }
        exit(1);
    }
}

static trtmc_string_view sv(const char* s) {
    trtmc_string_view v;
    v.data = s;
    v.size = strlen(s);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <bundle_path> <runtime_root>\n", argv[0]);
        return 1;
    }
    const trtmc_core_api_v1* api = NULL;
    trtmc_error* err = NULL;
    check(trtmc_get_api(1, 0, &api), NULL, &err, "trtmc_get_api");

    trtmc_load_options_v1 opts;
    memset(&opts, 0, sizeof(opts));
    opts.struct_size = sizeof(opts);
    opts.runtime_root = sv(argv[2]);
    trtmc_model* model = NULL;
    check(api->model_load(sv(argv[1]), &opts, &model, &err), api, &err, "model_load");

    const trtmc_api_header* header = NULL;
    check(api->model_get_task_api(model, sv(TRTMC_TASK_TEXT_TO_IMAGE), 1, 0, &header, &err), api,
          &err, "get text_to_image");
    const trtmc_text_to_image_api_v1* task = (const trtmc_text_to_image_api_v1*)header;

    trtmc_text_to_image_request_v1 request;
    memset(&request, 0, sizeof(request));
    request.prompt = sv("a lighthouse on a rocky coast at sunset");

    trtmc_config_entry_v1 entries[3];
    memset(entries, 0, sizeof(entries));
    entries[0].name = sv("num_steps");
    entries[0].value.kind = TRTMC_CONFIG_I64;
    entries[0].value.as.i64 = 4;
    entries[1].name = sv("guidance_scale");
    entries[1].value.kind = TRTMC_CONFIG_F64;
    entries[1].value.as.f64 = 4.5;
    entries[2].name = sv("negative_prompt");
    entries[2].value.kind = TRTMC_CONFIG_STRING;
    entries[2].value.as.string = sv("blurry");
    trtmc_config_view_v1 config;
    config.entries = entries;
    config.count = 3;

    trtmc_result* result = NULL;
    check(task->run(model, &request, &config, &result, &err), api, &err, "text_to_image run");
    trtmc_image_result_view_v1 view;
    memset(&view, 0, sizeof(view));
    check(task->result_view(result, &view, &err), api, &err, "text_to_image result_view");

    int failed = view.channels != 3 || view.height == 0 || view.width == 0 ||
                 view.pixel_count != (uint64_t)view.height * view.width * 3U;
    for (uint64_t i = 0; !failed && i < view.pixel_count; ++i)
        failed = !isfinite(view.pixels[i]) || view.pixels[i] < 0.0F || view.pixels[i] > 1.0F;
    if (failed)
        fprintf(stderr, "text_to_image returned an invalid image\n");
    else
        printf("text_to_image: %ux%u channels=%u\n", view.height, view.width, view.channels);
    api->result_release(result);
    api->model_release(model);
    if (failed)
        return 1;
    printf("stable_diffusion_3 C SDK consumer: all checks passed.\n");
    return 0;
}

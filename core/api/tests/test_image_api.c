/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/image.h"
#include "trtmc/trtmc.h"

#include <stdio.h>
#include <string.h>

static int failures;
static const trtmc_core_api_v1* core;

static void check(int condition, const char* label) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}
static trtmc_string_view string(const char* value) {
    trtmc_string_view result = {value, strlen(value)};
    return result;
}
static int success(trtmc_status status, trtmc_error** error) {
    if (status != TRTMC_OK && *error) {
        trtmc_string_view message = core->error_message(*error);
        fprintf(stderr, "API error: %.*s\n", (int)message.size, message.data);
    }
    core->error_release(*error);
    *error = NULL;
    return status == TRTMC_OK;
}
static int write_bundle(const char* path, const char* mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    char header[512];
    int length = snprintf(header, sizeof(header),
                          "{\"format\":1,\"family\":\"image_fixture\",\"task\":\"%s\",\"backend\":"
                          "\"fake\",\"sections\":{}}",
                          mode);
    FILE* out;
    unsigned shift;
    if (length < 0 || (size_t)length >= sizeof(header))
        return 0;
    out = fopen(path, "wb");
    if (!out)
        return 0;
    if (fwrite(magic, 1, sizeof(magic), out) != sizeof(magic)) {
        fclose(out);
        return 0;
    }
    for (shift = 0; shift < 64; shift += 8)
        fputc((int)(((uint64_t)length >> shift) & 255), out);
    if (fwrite(header, 1, (size_t)length, out) != (size_t)length) {
        fclose(out);
        return 0;
    }
    return fclose(out) == 0;
}

int main(int argc, char** argv) {
    char bundle[4096], single_bundle[4096];
    trtmc_model *model = NULL, *single = NULL;
    trtmc_error* error = NULL;
    trtmc_result* result = NULL;
    const trtmc_api_header* table = NULL;
    const trtmc_text_to_image_api_v1* generate;
    const trtmc_images_text_to_image_edit_api_v1* edit;
    const trtmc_masked_image_text_to_image_api_v1* masked;
    const trtmc_batch_text_to_image_api_v1* batch;
    trtmc_load_options_v1 options = {0};
    trtmc_image_result_view_v1 view = {0};
    uint64_t count = 0;
    if (argc != 2)
        return 2;
    if (snprintf(bundle, sizeof(bundle), "%s/c-image.bundle", argv[1]) >= (int)sizeof(bundle) ||
        snprintf(single_bundle, sizeof(single_bundle), "%s/c-image-single.bundle", argv[1]) >=
            (int)sizeof(single_bundle))
        return 2;
    if (!write_bundle(bundle, "all_images") || !write_bundle(single_bundle, "single_only"))
        return 2;
    if (trtmc_get_api(1, 0, &core) != TRTMC_OK)
        return 2;
    options.struct_size = sizeof(options);
    options.runtime_root = string(argv[1]);
    if (!success(core->model_load(string(bundle), &options, &model, &error), &error) ||
        !success(core->model_load(string(single_bundle), &options, &single, &error), &error))
        return 2;
    check(success(core->model_task_count(model, &count, &error), &error) && count == 4,
          "one multi-inherited family exposes four distinct image contracts");
    check(success(core->model_get_task_api(model, string(TRTMC_TASK_TEXT_TO_IMAGE), 1, 0, &table,
                                           &error),
                  &error),
          "generation table");
    generate = (const trtmc_text_to_image_api_v1*)table;
    check(success(core->model_get_task_api(model, string(TRTMC_TASK_IMAGES_TEXT_TO_IMAGE_EDIT), 1,
                                           0, &table, &error),
                  &error),
          "editing table");
    edit = (const trtmc_images_text_to_image_edit_api_v1*)table;
    check(success(core->model_get_task_api(model, string(TRTMC_TASK_MASKED_IMAGE_TEXT_TO_IMAGE), 1,
                                           0, &table, &error),
                  &error),
          "masked editing table");
    masked = (const trtmc_masked_image_text_to_image_api_v1*)table;
    check(success(core->model_get_task_api(model, string(TRTMC_TASK_BATCH_TEXT_TO_IMAGE), 1, 0,
                                           &table, &error),
                  &error),
          "batch table");
    batch = (const trtmc_batch_text_to_image_api_v1*)table;
    if (!generate || !edit || !masked || !batch)
        return 2;
    {
        const float replay_values[] = {0, -0.25F, 0.5F};
        trtmc_text_to_image_request_v1 input = {string("red"), {NULL, 0}};
        check(success(generate->run(model, &input, NULL, &result, &error), &error),
              "default generation");
        check(success(generate->result_view(result, &view, &error), &error) &&
                  view.pixel_count == 3 && view.pixels[0] == 0.25F &&
                  view.pixels[1] == 3.0F / 32.0F,
              "prompt and family defaults reach image pixels");
        core->result_release(result);
        result = NULL;
        input.initial_latents = (trtmc_f32_view){replay_values, 3};
        check(success(generate->run(model, &input, NULL, &result, &error), &error),
              "C typed initial latents execute");
        check(success(generate->result_view(result, &view, &error), &error) &&
                  view.pixels[0] == 0.25F && view.pixels[1] == 3.0F / 32 - 0.25F &&
                  view.pixels[2] == 0.625F,
              "C preserves each latent float without shared shape inference");
        core->result_release(result);
        result = NULL;
        check(generate->run(single, &input, NULL, &result, &error) == TRTMC_UNSUPPORTED && !result,
              "family must reject unsupported replay instead of silently generating");
        core->error_release(error);
        error = NULL;
        input.initial_latents.data = NULL;
        check(generate->run(model, &input, NULL, &result, &error) == TRTMC_INVALID_ARGUMENT &&
                  !result,
              "NULL positive-length replay is rejected before family call");
        core->error_release(error);
        error = NULL;
        input.initial_latents = (trtmc_f32_view){replay_values, UINT64_MAX};
        check(generate->run(model, &input, NULL, &result, &error) == TRTMC_INVALID_ARGUMENT &&
                  !result,
              "overflowed replay length cannot cross into family code");
        core->error_release(error);
        error = NULL;
        input.initial_latents.size = 2;
        check(generate->run(model, &input, NULL, &result, &error) == TRTMC_INVALID_ARGUMENT &&
                  !result,
              "family rejects wrong latent count");
        core->error_release(error);
        error = NULL;
    }
    {
        const float first[] = {0.2F, 0.3F, 0.4F};
        const uint8_t last[] = {255, 32, 64};
        trtmc_image_input_v1 images[] = {{first, sizeof(first), 1, 1, 3, TRTMC_IMAGE_FLOAT32},
                                         {last, sizeof(last), 1, 1, 3, TRTMC_IMAGE_UINT8}};
        trtmc_images_text_to_image_edit_request_v1 input = {
            images, 2, string("combine"), {NULL, 0}};
        check(success(edit->run(model, &input, NULL, &result, &error), &error),
              "ordered multi-image input");
        check(success(edit->result_view(result, &view, &error), &error) &&
                  view.pixels[1] == first[0] && view.pixels[2] == 1.0F,
              "image order/formats are preserved");
        core->result_release(result);
        result = NULL;
        check(edit->run(single, &input, NULL, &result, &error) == TRTMC_UNSUPPORTED &&
                  result == NULL,
              "another model's table cannot enable an undeclared edit task");
        core->error_release(error);
        error = NULL;
        images[0].byte_size = 1;
        check(edit->run(model, &input, NULL, &result, &error) == TRTMC_INVALID_ARGUMENT && !result,
              "image shape/storage mismatch fails before family execution");
        core->error_release(error);
        error = NULL;
        images[0].byte_size = sizeof(first);
        {
            float mask = 0;
            trtmc_masked_image_text_to_image_request_v1 request = {
                images[0], {&mask, 1, 1, 1}, string("edit")};
            check(success(masked->run(model, &request, NULL, &result, &error), &error),
                  "preserving mask");
            check(success(masked->result_view(result, &view, &error), &error) &&
                      view.pixels[0] == first[0],
                  "zero mask preserves source");
            core->result_release(result);
            result = NULL;
            mask = 1;
            check(success(masked->run(model, &request, NULL, &result, &error), &error),
                  "generation mask");
            check(success(masked->result_view(result, &view, &error), &error) &&
                      view.pixels[0] == 0.25F,
                  "one mask selects generated content");
            core->result_release(result);
            result = NULL;
        }
    }
    {
        trtmc_config_entry_v1 levels[] = {{{"level", 5}, {TRTMC_CONFIG_F64, {.f64 = 0.5}}},
                                          {{"level", 5}, {TRTMC_CONFIG_F64, {.f64 = 0.75}}}};
        trtmc_batch_text_to_image_item_v1 items[] = {
            {{string("first"), {NULL, 0}}, {&levels[0], 1}},
            {{string("second"), {NULL, 0}}, {&levels[1], 1}}};
        trtmc_batch_text_to_image_request_v1 input = {items, 2};
        check(success(batch->run_batch(model, &input, &result, &error), &error),
              "one native batch call");
        core->model_release(model);
        model = NULL;
        check(success(batch->result_count(result, &count, &error), &error) && count == 2,
              "batch survives model release");
        check(success(batch->result_item(result, 0, &view, &error), &error) &&
                  view.pixels[0] == 0.5F && view.pixels[2] == 0.875F,
              "first item config and native-batch marker");
        check(success(batch->result_item(result, 1, &view, &error), &error) &&
                  view.pixels[0] == 0.75F && view.pixels[1] == 1.0F / 32.0F,
              "second item config and order");
        check(generate->result_view(result, &view, &error) == TRTMC_INVALID_ARGUMENT &&
                  !view.pixels,
              "single-image view rejects a batch result owner");
        core->error_release(error);
        error = NULL;
        core->result_release(result);
    }
    core->model_release(model);
    model = NULL;
    {
        trtmc_text_to_image_request_v1 input = {string("participate"), {NULL, 0}};
        result = NULL;
        check(write_bundle(bundle, "worker"), "worker fixture bundle");
        check(success(core->model_load(string(bundle), &options, &model, &error), &error),
              "worker fixture load");
        check(success(generate->run(model, &input, NULL, &result, &error), &error) &&
                  success(generate->result_view(result, &view, &error), &error) &&
                  view.pixel_count == 0 && !view.pixels && view.height == 0 && view.width == 0 &&
                  view.channels == 3,
              "C caller observes explicit non-output participant completion");
        core->result_release(result);
        core->model_release(model);
    }
    core->model_release(single);
    remove(bundle);
    remove(single_bundle);
    fprintf(stderr, failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}

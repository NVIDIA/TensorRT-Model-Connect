/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/perception.h"
#include "trtmc/trtmc.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(offsetof(trtmc_image_to_boxes_api_v1, header) == 0,
               "image-only detection table starts with its ABI header");

static const trtmc_core_api_v1* core;
static int failures;
static trtmc_string_view text(const char* value) {
    const trtmc_string_view view = {value, (uint64_t)strlen(value)};
    return view;
}
static void check(int condition, const char* message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}
static void checked(trtmc_status status, trtmc_status expected, trtmc_error** error,
                    const char* message) {
    check(status == expected, message);
    check(status == TRTMC_OK ? *error == NULL : *error != NULL, "owned C error is consistent");
    core->error_release(*error);
    *error = NULL;
}
static trtmc_model* load(const char* root, const char* mode) {
    static const unsigned char magic[8] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    char path[4096], header[512];
    int size;
    unsigned shift;
    FILE* file;
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    if (snprintf(path, sizeof(path), "%s/image-boxes-c-%s.bundle", root, mode) >= (int)sizeof(path))
        exit(2);
    size = snprintf(header, sizeof(header),
                    "{\"format\":1,\"family\":\"image_boxes_fixture\",\"task\":\"%s\","
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
            "C image-only fixture loads");
    return model;
}

int main(int argc, char** argv) {
    trtmc_model *model, *disabled, *guarded;
    const trtmc_api_header* header = NULL;
    const trtmc_image_to_boxes_api_v1* task;
    trtmc_error* error = NULL;
    trtmc_result *result = NULL, *rejected = NULL;
    trtmc_detected_boxes_view_v1 view = {0};
    float pixels[36] = {0.5F};
    trtmc_perception_image_request_v1 request = {
        {pixels, sizeof(pixels), 3, 4, 3, TRTMC_IMAGE_FLOAT32}};
    trtmc_config_entry_v1 entries[2] = {0};
    trtmc_config_view_v1 config = {entries, 1};
    if (argc != 2 || trtmc_get_api(1, 0, &core) != TRTMC_OK)
        return 2;
    model = load(argv[1], "image_to_boxes");
    disabled = load(argv[1], "disabled");
    guarded = load(argv[1], "must_not_run");
    if (!model || !disabled || !guarded)
        return 2;
    checked(core->model_get_task_api(model, text(TRTMC_TASK_IMAGE_TO_BOXES), 1, 0, &header, &error),
            TRTMC_OK, &error, "C image-only table lookup");
    if (!header)
        return 2;
    task = (const trtmc_image_to_boxes_api_v1*)header;
    check(header->byte_size == sizeof(*task) && header->major == 1 && header->minor == 0,
          "new Task has its own exact v1 table");
    checked(task->run(model, &request, NULL, &result, &error), TRTMC_OK, &error,
            "C image-only call requires no text");
    checked(task->result_view(result, &view, &error), TRTMC_OK, &error, "C typed detection view");
    check(view.count == 2 && view.image_height == 3 && view.image_width == 4 &&
              view.boxes[0].class_id == 42 && view.boxes[1].class_id == 7 &&
              view.boxes[0].score == 0.75F && view.boxes[1].score == 0 &&
              view.boxes[0].box.x_min == -2 && view.boxes[0].box.x_max == 7 &&
              view.boxes[0].box.y_min == 0.5F && view.boxes[0].box.y_max == 3,
          "C preserves dimensions, numeric IDs, scores and unmodified original-image pixels");
    checked(task->run(disabled, &request, NULL, &rejected, &error), TRTMC_UNSUPPORTED, &error,
            "borrowing another model's table cannot grant support");
    check(rejected == NULL, "unsupported call clears result output");
    entries[0].name = text("unknown");
    entries[0].value.kind = TRTMC_CONFIG_BOOL;
    entries[0].value.as.boolean = 1;
    checked(task->run(guarded, &request, &config, &rejected, &error), TRTMC_INVALID_CONFIG, &error,
            "C unknown Config is rejected before family execution");
    entries[0].name = text("score");
    checked(task->run(guarded, &request, &config, &rejected, &error), TRTMC_INVALID_CONFIG, &error,
            "C wrong Config type is rejected before family execution");
    entries[0].value.kind = TRTMC_CONFIG_F64;
    entries[0].value.as.f64 = 0;
    entries[1] = entries[0];
    config.count = 2;
    checked(task->run(guarded, &request, &config, &rejected, &error), TRTMC_INVALID_CONFIG, &error,
            "C duplicate Config is rejected before family execution");
    config.count = 1;
    checked(task->run(model, &request, &config, &rejected, &error), TRTMC_OK, &error,
            "C explicit zero Config remains present");
    checked(task->result_view(rejected, &view, &error), TRTMC_OK, &error, "C zero-score view");
    check(view.count == 2 && view.boxes[0].score == 0, "C zero score is not replaced by default");
    core->result_release(rejected);
    rejected = NULL;
    entries[0].name = text("empty");
    entries[0].value.kind = TRTMC_CONFIG_BOOL;
    entries[0].value.as.boolean = 1;
    checked(task->run(model, &request, &config, &rejected, &error), TRTMC_OK, &error,
            "C empty detections succeed");
    checked(task->result_view(rejected, &view, &error), TRTMC_OK, &error, "C empty detection view");
    check(view.count == 0 && view.image_height == 3 && view.image_width == 4,
          "C empty detections retain image dimensions");
    core->result_release(rejected);
    rejected = NULL;
    {
        const char* modes[] = {"bad_dimensions", "wrong_image_dimensions", "bad_box", "bad_score"};
        size_t i;
        for (i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
            trtmc_model* invalid = load(argv[1], modes[i]);
            checked(task->run(invalid, &request, NULL, &rejected, &error), TRTMC_INTERNAL_ERROR,
                    &error, "C malformed family detection result is rejected");
            check(rejected == NULL, "C malformed output never returns a result");
            core->model_release(invalid);
        }
    }
    checked(task->run(model, NULL, NULL, &rejected, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C null request is rejected");
    request.image.byte_size = 0;
    checked(task->run(model, &request, NULL, &rejected, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C malformed image storage is rejected");
    checked(task->result_view(result, NULL, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C null result-view output is rejected");
    core->model_release(model);
    core->model_release(disabled);
    core->model_release(guarded);
    checked(task->result_view(result, &view, &error), TRTMC_OK, &error,
            "C result still owns boxes after model release");
    check(view.count == 2 && view.boxes[0].class_id == 42, "C retained result is intact");
    core->result_release(result);
    puts("C image-only detection: 4x3 input, two fixed-label boxes");
    return failures ? 1 : 0;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C11 SDK consumer for the family-owned Qwen-Image 2.1 end-to-end test.
 * Replays caller-supplied initial latents through text_to_image, optionally with
 * classifier-free guidance, releases the model, then writes the owned float32
 * HWC pixels and prints their shape. */
#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/image.h>
#include <trtmc/trtmc.h>

_Static_assert(sizeof(float) == 4, "latents and pixels use float32");

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}

static int same(trtmc_string_view left, const char* right) {
    return left.size == strlen(right) && memcmp(left.data, right, (size_t)left.size) == 0;
}

/* Reads a whole file of float32 values; the family validates the count. */
static float* read_floats(const char* path, size_t* count) {
    FILE* file = fopen(path, "rb");
    float* values = NULL;
    long size = 0;
    if (!file)
        return NULL;
    if (fseek(file, 0, SEEK_END) == 0 && (size = ftell(file)) > 0 && size % 4 == 0 &&
        fseek(file, 0, SEEK_SET) == 0) {
        *count = (size_t)size / sizeof(float);
        values = (float*)malloc((size_t)size);
        if (values && fread(values, sizeof(float), *count, file) != *count) {
            free(values);
            values = NULL;
        }
    }
    fclose(file);
    return values;
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_text_to_image_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_text_to_image_request_v1 request = {0};
    trtmc_config_entry_v1 entries[3] = {0};
    trtmc_config_view_v1 config = {0};
    trtmc_image_result_view_v1 view = {0};
    float* latents = NULL;
    FILE* output = NULL;
    size_t latent_count = 0;
    uint64_t field_count = 0, index;
    char* end = NULL;
    long long steps;
    double cfg_scale = 0.0;
    int declares_steps = 0;
    int status = 1;

    if (argc != 7 && argc != 9) {
        fprintf(stderr,
                "Usage: %s BUNDLE RUNTIME_ROOT PROMPT LATENTS_F32 NUM_STEPS PIXELS_F32 "
                "[CFG_SCALE NEGATIVE_PROMPT]\n",
                argv[0]);
        return 2;
    }
    errno = 0;
    steps = strtoll(argv[5], &end, 10);
    if (errno || end == argv[5] || *end || steps < 1) {
        fprintf(stderr, "NUM_STEPS must be a positive integer\n");
        return 2;
    }
    if (argc == 9) {
        errno = 0;
        cfg_scale = strtod(argv[7], &end);
        if (errno || end == argv[7] || *end) {
            fprintf(stderr, "CFG_SCALE must be a number\n");
            return 2;
        }
    }
    latents = read_floats(argv[4], &latent_count);
    if (!latents) {
        fprintf(stderr, "LATENTS_F32 must be a non-empty float32 file\n");
        return 2;
    }

    if (trtmc_get_api(1, 0, &core) != TRTMC_OK || !core) {
        fprintf(stderr, "unable to obtain the v1 C API\n");
        goto cleanup;
    }
    options.struct_size = sizeof(options);
    options.runtime_root = text(argv[2]);
    if (core->model_load(text(argv[1]), &options, &model, &error) != TRTMC_OK)
        goto cleanup;
    if (core->model_get_task_api(model, text(TRTMC_TASK_TEXT_TO_IMAGE), 1, 0, &header, &error) !=
        TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*task)) {
        fprintf(stderr, "incomplete text_to_image C table\n");
        goto cleanup;
    }
    task = (const trtmc_text_to_image_api_v1*)header;

    if (core->config_field_count(model, text(TRTMC_TASK_TEXT_TO_IMAGE), 1, 0, &field_count,
                                 &error) != TRTMC_OK)
        goto cleanup;
    for (index = 0; index < field_count; ++index) {
        trtmc_config_field_v1 field;
        if (core->config_field_info(model, text(TRTMC_TASK_TEXT_TO_IMAGE), 1, 0, index, &field,
                                    &error) != TRTMC_OK)
            goto cleanup;
        if (same(field.name, "num_steps") && field.kind == TRTMC_CONFIG_I64)
            declares_steps = 1;
    }
    if (!declares_steps) {
        fprintf(stderr, "Qwen-Image 2.1 must declare the integer Config field num_steps\n");
        goto cleanup;
    }

    entries[0].name = text("num_steps");
    entries[0].value.kind = TRTMC_CONFIG_I64;
    entries[0].value.as.i64 = (int64_t)steps;
    config.entries = entries;
    config.count = 1;
    if (argc == 9) {
        entries[1].name = text("cfg_scale");
        entries[1].value.kind = TRTMC_CONFIG_F64;
        entries[1].value.as.f64 = cfg_scale;
        entries[2].name = text("negative_prompt");
        entries[2].value.kind = TRTMC_CONFIG_STRING;
        entries[2].value.as.string = text(argv[8]);
        config.count = 3;
    }
    request.prompt = text(argv[3]);
    request.initial_latents.data = latents;
    request.initial_latents.size = (uint64_t)latent_count;
    if (task->run(model, &request, &config, &result, &error) != TRTMC_OK)
        goto cleanup;
    /* The result must stay readable after its inputs and model are gone. */
    free(latents);
    latents = NULL;
    core->model_release(model);
    model = NULL;

    if (task->result_view(result, &view, &error) != TRTMC_OK)
        goto cleanup;
    if (!view.pixels || !view.height || !view.width || !view.channels ||
        view.pixel_count != (uint64_t)view.height * view.width * view.channels) {
        fprintf(stderr, "image result does not describe one HWC image\n");
        goto cleanup;
    }
    output = fopen(argv[6], "wb");
    if (!output || fwrite(view.pixels, sizeof(float), (size_t)view.pixel_count, output) !=
                       (size_t)view.pixel_count) {
        fprintf(stderr, "failed to write PIXELS_F32\n");
        goto cleanup;
    }
    if (fclose(output) != 0) {
        output = NULL;
        fprintf(stderr, "failed to close PIXELS_F32\n");
        goto cleanup;
    }
    output = NULL;

    printf("{\"task\":\"text_to_image\",\"height\":%" PRIu32 ",\"width\":%" PRIu32
           ",\"channels\":%" PRIu32 "}\n",
           view.height, view.width, view.channels);
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
    if (output)
        fclose(output);
    free(latents);
    return status;
}

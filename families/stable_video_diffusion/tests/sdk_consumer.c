/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C11 SDK consumer for the family-owned Stable Video Diffusion E2E.
 * Reads interleaved RGB uint8 pixels, checks that a text prompt is refused,
 * generates the video with the given seed and step count, releases the input
 * and the model, then writes the THWC float32 frames and prints a summary. */
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/trtmc.h>
#include <trtmc/video.h>

/* The published pipeline's default frame rate, which the run does not override. */
#define DEFAULT_FPS 7.0

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}

static int64_t integer(const char* value, int64_t minimum, int64_t maximum) {
    char* end = NULL;
    intmax_t parsed;
    errno = 0;
    parsed = strtoimax(value, &end, 10);
    if (errno || end == value || *end || parsed < minimum || parsed > maximum)
        return -1;
    return (int64_t)parsed;
}

static int same_name(trtmc_string_view name, const char* expected) {
    return name.size == strlen(expected) && memcmp(name.data, expected, (size_t)name.size) == 0;
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_initial_image_text_to_video_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_initial_image_text_to_video_request_v1 request = {0};
    trtmc_video_result_view_v1 view = {0};
    trtmc_config_entry_v1 entries[2];
    trtmc_config_view_v1 config = {0};
    trtmc_config_field_v1 field = {0};
    unsigned char* input = NULL;
    FILE* file = NULL;
    int64_t height, width, seed, steps;
    size_t count;
    uint64_t i, j, field_count = 0;
    int seen_seed = 0, seen_steps = 0, status = 1;

    if (argc != 9) {
        fprintf(stderr, "Usage: %s BUNDLE RUNTIME_ROOT RGB_U8 HEIGHT WIDTH SEED STEPS FRAMES_F32\n",
                argv[0]);
        return 2;
    }
    height = integer(argv[4], 1, 16384);
    width = integer(argv[5], 1, 16384);
    seed = integer(argv[6], 0, 4294967295LL);
    steps = integer(argv[7], 1, 1000);
    if (height < 0 || width < 0 || seed < 0 || steps < 0) {
        fprintf(stderr, "invalid height, width, seed or step count\n");
        return 2;
    }
    count = (size_t)height * (size_t)width * 3;
    input = (unsigned char*)malloc(count);
    file = fopen(argv[3], "rb");
    if (!input || !file || fread(input, 1, count, file) != count || fgetc(file) != EOF ||
        ferror(file)) {
        fprintf(stderr, "input must contain exactly HEIGHT x WIDTH x 3 uint8 RGB values\n");
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
    if (core->model_get_task_api(model, text(TRTMC_TASK_INITIAL_IMAGE_TEXT_TO_VIDEO), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*task)) {
        fprintf(stderr, "incomplete initial_image_text_to_video C table\n");
        goto cleanup;
    }
    task = (const trtmc_initial_image_text_to_video_api_v1*)header;
    if (core->config_field_count(model, text(TRTMC_TASK_INITIAL_IMAGE_TEXT_TO_VIDEO), 1, 0,
                                 &field_count, &error) != TRTMC_OK)
        goto cleanup;
    for (i = 0; i < field_count; ++i) {
        if (core->config_field_info(model, text(TRTMC_TASK_INITIAL_IMAGE_TEXT_TO_VIDEO), 1, 0, i,
                                    &field, &error) != TRTMC_OK)
            goto cleanup;
        if (field.kind == TRTMC_CONFIG_I64 && field.has_fixed_default) {
            seen_seed |= same_name(field.name, "seed");
            seen_steps |= same_name(field.name, "num_steps");
        }
    }
    if (field_count != 7 || !seen_seed || !seen_steps) {
        fprintf(stderr, "expected seven Config fields including integer seed and num_steps\n");
        goto cleanup;
    }

    request.initial_image = (trtmc_image_input_v1){input,           count, (uint32_t)height,
                                                   (uint32_t)width, 3,     TRTMC_IMAGE_UINT8};
    /* The model has no text conditioning: a prompt is an error, not ignored. */
    request.prompt = text("a red car");
    if (task->run(model, &request, NULL, &result, &error) == TRTMC_OK) {
        fprintf(stderr, "a non-empty prompt was accepted\n");
        goto cleanup;
    }
    core->error_release(error);
    error = NULL;

    request.prompt = text("");
    entries[0].name = text("seed");
    entries[0].value.kind = TRTMC_CONFIG_I64;
    entries[0].value.as.i64 = seed;
    entries[1].name = text("num_steps");
    entries[1].value.kind = TRTMC_CONFIG_I64;
    entries[1].value.as.i64 = steps;
    config.entries = entries;
    config.count = 2;
    if (task->run(model, &request, &config, &result, &error) != TRTMC_OK)
        goto cleanup;
    /* The result must stay readable after its input and model are gone. */
    free(input);
    input = NULL;
    core->model_release(model);
    model = NULL;

    if (task->result_view(result, &view, &error) != TRTMC_OK)
        goto cleanup;
    if (!view.frame_count || !view.frames || view.conditioned_prefix_frames != 0 ||
        view.timestamps_seconds.size != view.frame_count) {
        fprintf(stderr, "video result has no frames, a conditioned prefix or no clock\n");
        goto cleanup;
    }
    file = fopen(argv[8], "wb");
    if (!file)
        goto cleanup;
    for (i = 0; i < view.frame_count; ++i) {
        const trtmc_image_result_view_v1* frame = &view.frames[i];
        if (frame->height != (uint32_t)height || frame->width != (uint32_t)width ||
            frame->channels != 3 || frame->pixel_count != count || !frame->pixels ||
            fabs(view.timestamps_seconds.data[i] - (double)i / DEFAULT_FPS) > 1e-12) {
            fprintf(stderr, "frame %" PRIu64 " does not match the request\n", i);
            goto cleanup;
        }
        for (j = 0; j < frame->pixel_count; ++j) {
            if (!(frame->pixels[j] >= 0.0F && frame->pixels[j] <= 1.0F)) {
                fprintf(stderr, "frame %" PRIu64 " has a value outside [0, 1]\n", i);
                goto cleanup;
            }
        }
        if (fwrite(frame->pixels, sizeof(float), (size_t)frame->pixel_count, file) !=
            (size_t)frame->pixel_count)
            goto cleanup;
    }
    if (fclose(file) != 0) {
        file = NULL;
        goto cleanup;
    }
    file = NULL;
    printf("{\"task\":\"initial_image_text_to_video\",\"frames\":%" PRIu64 ",\"height\":%" PRId64
           ",\"width\":%" PRId64 ",\"inference_ms\":%.3f}\n",
           view.frame_count, height, width, view.inference_ms);
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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C SDK consumer for the family-owned end-to-end test.
 *
 * Usage: BUNDLE RUNTIME_ROOT FRAMES u8|f32 FRAME_COUNT HEIGHT WIDTH MASK OUTPUT
 *
 * FRAMES holds FRAME_COUNT x HEIGHT x WIDTH x 3 RGB values (uint8, or float32
 * in [0, 1]); MASK holds FRAME_COUNT x HEIGHT x WIDTH float32 values. The
 * consumer first checks four requests the family must refuse, then runs the
 * clip without timestamps, writes the THWC float32 result to OUTPUT and
 * prints one JSON line. */
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/trtmc.h>
#include <trtmc/video.h>

_Static_assert(sizeof(float) == 4, "frames and masks use float32");

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}

static uint64_t positive(const char* value) {
    char* end = NULL;
    const unsigned long long parsed = strtoull(value, &end, 10);
    if (end == value || *end || !parsed || parsed > INT32_MAX)
        return 0;
    return (uint64_t)parsed;
}

static int read_exact(const char* path, void* data, size_t bytes) {
    FILE* file = fopen(path, "rb");
    int ok = file && fread(data, 1, bytes, file) == bytes && fgetc(file) == EOF && !ferror(file);
    if (file)
        fclose(file);
    return ok;
}

/* Runs a request that must be refused as an invalid argument. */
static int refused(const trtmc_core_api_v1* core,
                   const trtmc_masked_video_text_to_video_api_v1* task, trtmc_model* model,
                   const trtmc_masked_video_text_to_video_request_v1* request, const char* name) {
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    const trtmc_status status = task->run(model, request, NULL, &result, &error);
    core->result_release(result);
    if (error)
        core->error_release(error);
    if (status != TRTMC_INVALID_ARGUMENT) {
        fprintf(stderr, "request '%s' returned status %d instead of an invalid argument\n", name,
                (int)status);
        return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_masked_video_text_to_video_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_masked_video_text_to_video_request_v1 request = {0};
    trtmc_video_result_view_v1 view = {0};
    trtmc_image_input_v1* inputs = NULL;
    unsigned char* pixels = NULL;
    float* mask = NULL;
    float* empty = NULL;
    FILE* output = NULL;
    uint64_t frames, height, width, plane, values, i, j, field_count = 0;
    size_t element;
    int uint8_input, status = 1;
    if (argc != 10) {
        fprintf(
            stderr,
            "Usage: %s BUNDLE RUNTIME_ROOT FRAMES u8|f32 FRAME_COUNT HEIGHT WIDTH MASK OUTPUT\n",
            argv[0]);
        return 2;
    }
    uint8_input = strcmp(argv[4], "u8") == 0;
    frames = positive(argv[5]);
    height = positive(argv[6]);
    width = positive(argv[7]);
    if ((!uint8_input && strcmp(argv[4], "f32") != 0) || frames < 2 || !height || !width) {
        fprintf(stderr, "invalid format or clip dimensions\n");
        return 2;
    }
    element = uint8_input ? sizeof(unsigned char) : sizeof(float);
    /* Bound every product by division before forming it. */
    if (width > UINT64_MAX / height ||
        height * width > (uint64_t)(SIZE_MAX / 3 / sizeof(float)) / frames) {
        fprintf(stderr, "clip size overflows host storage\n");
        return 2;
    }
    plane = height * width;
    values = frames * plane * 3;
    pixels = (unsigned char*)malloc((size_t)values * element);
    mask = (float*)malloc((size_t)(frames * plane) * sizeof(float));
    empty = (float*)calloc((size_t)(frames * plane), sizeof(float));
    inputs = (trtmc_image_input_v1*)calloc((size_t)frames, sizeof(*inputs));
    if (!pixels || !mask || !empty || !inputs ||
        !read_exact(argv[3], pixels, (size_t)values * element) ||
        !read_exact(argv[8], mask, (size_t)(frames * plane) * sizeof(float))) {
        fprintf(stderr, "frames and mask must hold exactly the stated clip\n");
        goto cleanup;
    }
    for (i = 0; i < frames; ++i) {
        inputs[i].data = pixels + (size_t)(i * plane * 3) * element;
        inputs[i].byte_size = plane * 3 * element;
        inputs[i].height = (uint32_t)height;
        inputs[i].width = (uint32_t)width;
        inputs[i].channels = 3;
        inputs[i].format = uint8_input ? TRTMC_IMAGE_UINT8 : TRTMC_IMAGE_FLOAT32;
    }
    if (trtmc_get_api(1, 0, &core) != TRTMC_OK || !core) {
        fprintf(stderr, "unable to obtain the v1 C API\n");
        goto cleanup;
    }
    options.struct_size = sizeof(options);
    options.runtime_root = text(argv[2]);
    if (core->model_load(text(argv[1]), &options, &model, &error) != TRTMC_OK)
        goto cleanup;
    if (core->model_get_task_api(model, text(TRTMC_TASK_MASKED_VIDEO_TEXT_TO_VIDEO), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*task)) {
        fprintf(stderr, "incomplete masked video C table\n");
        goto cleanup;
    }
    task = (const trtmc_masked_video_text_to_video_api_v1*)header;
    if (core->config_field_count(model, text(TRTMC_TASK_MASKED_VIDEO_TEXT_TO_VIDEO), 1, 0,
                                 &field_count, &error) != TRTMC_OK)
        goto cleanup;
    if (field_count != 0) {
        fprintf(stderr, "videomae must expose no runtime Config fields\n");
        goto cleanup;
    }

    request.source.frames = inputs;
    request.source.frame_count = frames;
    request.mask.values = mask;
    request.mask.count = frames * plane;
    request.mask.frames = frames;
    request.mask.height = height;
    request.mask.width = width;
    request.prompt = text("a red car");
    if (!refused(core, task, model, &request, "prompt"))
        goto cleanup;
    request.prompt = text("");
    mask[0] = 1.0F - mask[0]; /* the first tube becomes mixed */
    if (!refused(core, task, model, &request, "mixed_tube"))
        goto cleanup;
    mask[0] = 1.0F - mask[0];
    request.mask.values = empty;
    if (!refused(core, task, model, &request, "empty_mask"))
        goto cleanup;
    request.mask.values = mask;
    request.source.frame_count = frames - 1;
    request.mask.frames = frames - 1;
    request.mask.count = (frames - 1) * plane;
    if (!refused(core, task, model, &request, "frame_count"))
        goto cleanup;
    request.source.frame_count = frames;
    request.mask.frames = frames;
    request.mask.count = frames * plane;

    if (task->run(model, &request, NULL, &result, &error) != TRTMC_OK)
        goto cleanup;
    if (task->result_view(result, &view, &error) != TRTMC_OK)
        goto cleanup;
    if (view.frame_count != frames || view.conditioned_prefix_frames != 0 ||
        view.timestamps_seconds.size != 0) {
        fprintf(stderr, "result must hold one frame per source frame and no timeline\n");
        goto cleanup;
    }
    output = fopen(argv[9], "wb");
    if (!output) {
        fprintf(stderr, "cannot write %s\n", argv[9]);
        goto cleanup;
    }
    for (i = 0; i < frames; ++i) {
        const trtmc_image_result_view_v1* frame = &view.frames[i];
        if (frame->height != height || frame->width != width || frame->channels != 3 ||
            frame->pixel_count != plane * 3) {
            fprintf(stderr, "result frame %" PRIu64 " has another shape\n", i);
            goto cleanup;
        }
        for (j = 0; j < frame->pixel_count; ++j)
            if (!(frame->pixels[j] >= 0.0F && frame->pixels[j] <= 1.0F)) {
                fprintf(stderr, "result frame %" PRIu64 " leaves [0, 1]\n", i);
                goto cleanup;
            }
        if (fwrite(frame->pixels, sizeof(float), (size_t)frame->pixel_count, output) !=
            (size_t)frame->pixel_count) {
            fprintf(stderr, "cannot write %s\n", argv[9]);
            goto cleanup;
        }
    }
    if (fclose(output) != 0) {
        output = NULL;
        fprintf(stderr, "cannot write %s\n", argv[9]);
        goto cleanup;
    }
    output = NULL;
    printf("{\"task\":\"masked_video_text_to_video\",\"format\":\"%s\",\"frames\":%" PRIu64
           ",\"height\":%" PRIu64 ",\"width\":%" PRIu64
           ",\"channels\":3,\"conditioned_prefix_frames\":%" PRIu64
           ",\"timestamps_seconds\":[],\"refused\":[\"prompt\",\"mixed_tube\",\"empty_mask\","
           "\"frame_count\"],\"setup_ms\":%.17g,\"inference_ms\":%.17g}\n",
           uint8_input ? "u8" : "f32", view.frame_count, height, width,
           view.conditioned_prefix_frames, view.setup_ms, view.inference_ms);
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
    free(inputs);
    free(empty);
    free(mask);
    free(pixels);
    return status;
}

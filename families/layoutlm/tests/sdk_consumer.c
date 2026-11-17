/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C SDK consumer for the family-owned end-to-end test: one document
 * image as float32 RGB pixels in [0, 1] and one question in, the answer out. */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/language.h>
#include <trtmc/trtmc.h>

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}

static void json_string(trtmc_string_view value) {
    uint64_t i;
    putchar('"');
    for (i = 0; i < value.size; ++i) {
        const unsigned char byte = (unsigned char)value.data[i];
        if (byte == '"' || byte == '\\') {
            putchar('\\');
            putchar(byte);
        } else if (byte < 32) {
            printf("\\u%04x", (unsigned)byte);
        } else {
            putchar(byte);
        }
    }
    putchar('"');
}

static float* read_pixels(const char* path, uint64_t count) {
    FILE* file = fopen(path, "rb");
    float* pixels = NULL;
    if (!file)
        return NULL;
    pixels = (float*)malloc((size_t)count * sizeof(float));
    if (pixels && fread(pixels, sizeof(float), (size_t)count, file) != (size_t)count) {
        free(pixels);
        pixels = NULL;
    }
    if (pixels && fgetc(file) != EOF) {
        free(pixels);
        pixels = NULL;
    }
    fclose(file);
    return pixels;
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_images_text_to_text_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_images_text_part_v1 parts[2];
    trtmc_images_text_message_v1 message = {0};
    trtmc_images_text_to_text_request_v1 request = {0};
    trtmc_text_result_view_v1 view = {0};
    uint64_t i, field_count = 0, width, height;
    float* pixels = NULL;
    int status = 1;
    if (argc != 7) {
        fprintf(stderr, "Usage: %s BUNDLE RUNTIME_ROOT RGB_F32_FILE WIDTH HEIGHT QUESTION\n",
                argv[0]);
        return 2;
    }
    width = strtoull(argv[4], NULL, 10);
    height = strtoull(argv[5], NULL, 10);
    if (width == 0 || height == 0 || width > 65536 || height > 65536) {
        fprintf(stderr, "image dimensions must lie in 1..65536\n");
        return 2;
    }
    pixels = read_pixels(argv[3], width * height * 3);
    if (!pixels) {
        fprintf(stderr, "float32 RGB file size does not match WIDTH x HEIGHT x 3\n");
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
    if (core->model_get_task_api(model, text(TRTMC_TASK_IMAGES_TEXT_TO_TEXT), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*task)) {
        fprintf(stderr, "incomplete images-text-to-text C table\n");
        goto cleanup;
    }
    task = (const trtmc_images_text_to_text_api_v1*)header;
    if (core->config_field_count(model, text(TRTMC_TASK_IMAGES_TEXT_TO_TEXT), 1, 0, &field_count,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (field_count != 0) {
        fprintf(stderr, "layoutlm must expose no runtime Config fields\n");
        goto cleanup;
    }
    memset(parts, 0, sizeof(parts));
    parts[0].kind = TRTMC_MEDIA_IMAGE;
    parts[0].content.image.data = pixels;
    parts[0].content.image.byte_size = width * height * 3 * sizeof(float);
    parts[0].content.image.height = (uint32_t)height;
    parts[0].content.image.width = (uint32_t)width;
    parts[0].content.image.channels = 3;
    parts[0].content.image.format = TRTMC_IMAGE_FLOAT32;
    parts[1].kind = TRTMC_MEDIA_TEXT;
    parts[1].content.text = text(argv[6]);
    message.role = TRTMC_ROLE_USER;
    message.parts = parts;
    message.part_count = 2;
    request.messages = &message;
    request.message_count = 1;
    if (task->run(model, &request, NULL, &result, &error) != TRTMC_OK)
        goto cleanup;
    core->model_release(model);
    model = NULL;
    if (task->result_view(result, &view, &error) != TRTMC_OK)
        goto cleanup;
    printf("{\"task\":\"images_text_to_text\",\"format\":\"float32\",\"text\":");
    json_string(view.text);
    printf(",\"token_ids\":[");
    for (i = 0; i < view.token_ids.size; ++i)
        printf("%s%" PRId32, i ? "," : "", view.token_ids.data[i]);
    puts("]}");
    status = fflush(stdout) == 0 && !ferror(stdout) ? 0 : 1;
cleanup:
    if (error && core) {
        const trtmc_string_view message_text = core->error_message(error);
        fprintf(stderr, "C API error %d: ", (int)core->error_code(error));
        fwrite(message_text.data, 1, (size_t)message_text.size, stderr);
        fputc('\n', stderr);
        core->error_release(error);
    }
    if (core) {
        core->result_release(result);
        core->model_release(model);
    }
    free(pixels);
    return status;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C11 SDK consumer for the family-owned TrOCR end-to-end test.
 * Loads the model once and runs images_text_to_text on every image given, in
 * order, each with one image and an empty text part (and max_new_tokens when
 * it is not 0). The model is released before the results are read; each
 * result is printed as one JSON line: the generated token ids and the text as
 * hexadecimal UTF-8 bytes. Images are interleaved RGB float32 in [0, 1]. */
#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/language.h>
#include <trtmc/trtmc.h>

_Static_assert(sizeof(float) == 4, "input uses float32");

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}

static int same(trtmc_string_view view, const char* value) {
    return view.size == strlen(value) && (!view.size || !memcmp(view.data, value, view.size));
}

static uint64_t number(const char* value, uint64_t limit, int allow_zero) {
    char* end = NULL;
    uintmax_t parsed;
    errno = 0;
    parsed = strtoumax(value, &end, 10);
    if (errno || end == value || *end || (!parsed && !allow_zero) || parsed > limit)
        return UINT64_MAX;
    return (uint64_t)parsed;
}

/* Reads HEIGHT x WIDTH x 3 float32 values; returns NULL and reports otherwise. */
static float* read_image(const char* path, uint64_t height, uint64_t width, size_t* count) {
    float* input;
    FILE* file;
    if (height == UINT64_MAX || width == UINT64_MAX ||
        (size_t)height > SIZE_MAX / width / 3 / sizeof(float)) {
        fprintf(stderr, "invalid or overflowing image dimensions\n");
        return NULL;
    }
    *count = (size_t)height * width * 3;
    input = (float*)malloc(*count * sizeof(float));
    file = fopen(path, "rb");
    if (!input || !file || fread(input, sizeof(float), *count, file) != *count ||
        fgetc(file) != EOF || ferror(file)) {
        fprintf(stderr, "input must contain exactly HEIGHT x WIDTH x 3 float32 RGB values\n");
        free(input);
        input = NULL;
    }
    if (file)
        fclose(file);
    return input;
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_images_text_to_text_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result** results = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_images_text_part_v1 parts[2];
    trtmc_images_text_message_v1 message;
    trtmc_images_text_to_text_request_v1 request = {0};
    trtmc_config_entry_v1 entry;
    trtmc_config_view_v1 config = {0};
    trtmc_config_field_v1 field;
    trtmc_text_result_view_v1 view;
    float* input = NULL;
    uint64_t height, width, max_new_tokens, field_count = 0, i;
    size_t count = 0, image, images;
    int status = 1;

    if (argc < 7 || (argc - 4) % 3 != 0) {
        fprintf(stderr,
                "Usage: %s BUNDLE RUNTIME_ROOT MAX_NEW_TOKENS RGB_F32 HEIGHT WIDTH"
                " [RGB_F32 HEIGHT WIDTH ...]\n",
                argv[0]);
        return 2;
    }
    max_new_tokens = number(argv[3], INT32_MAX, 1);
    if (max_new_tokens == UINT64_MAX) {
        fprintf(stderr, "invalid MAX_NEW_TOKENS\n");
        return 2;
    }
    images = (size_t)(argc - 4) / 3;
    results = (trtmc_result**)calloc(images, sizeof(*results));
    if (!results)
        return 1;

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
        fprintf(stderr, "incomplete images_text_to_text C table\n");
        goto cleanup;
    }
    task = (const trtmc_images_text_to_text_api_v1*)header;
    if (core->config_field_count(model, text(TRTMC_TASK_IMAGES_TEXT_TO_TEXT), 1, 0, &field_count,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (field_count != 1 ||
        core->config_field_info(model, text(TRTMC_TASK_IMAGES_TEXT_TO_TEXT), 1, 0, 0, &field,
                                &error) != TRTMC_OK ||
        !same(field.name, "max_new_tokens") || field.kind != TRTMC_CONFIG_I64 ||
        field.has_fixed_default) {
        if (!error)
            fprintf(stderr, "TrOCR must expose exactly max_new_tokens without a fixed default\n");
        goto cleanup;
    }
    if (max_new_tokens) {
        entry.name = text("max_new_tokens");
        entry.value.kind = TRTMC_CONFIG_I64;
        entry.value.as.i64 = (int64_t)max_new_tokens;
        config.entries = &entry;
        config.count = 1;
    }

    /* One loaded model serves every request in order. */
    for (image = 0; image < images; ++image) {
        height = number(argv[5 + 3 * image], INT32_MAX, 0);
        width = number(argv[6 + 3 * image], INT32_MAX, 0);
        input = read_image(argv[4 + 3 * image], height, width, &count);
        if (!input)
            goto cleanup;
        parts[0].kind = TRTMC_MEDIA_IMAGE;
        parts[0].content.image =
            (trtmc_image_input_v1){input, count * sizeof(float), (uint32_t)height, (uint32_t)width,
                                   3,     TRTMC_IMAGE_FLOAT32};
        parts[1].kind = TRTMC_MEDIA_TEXT;
        parts[1].content.text = text("");
        message.role = TRTMC_ROLE_USER;
        message.parts = parts;
        message.part_count = 2;
        request.messages = &message;
        request.message_count = 1;
        if (task->run(model, &request, &config, &results[image], &error) != TRTMC_OK)
            goto cleanup;
        free(input);
        input = NULL;
    }
    /* The results must stay readable after their inputs and the model are gone. */
    core->model_release(model);
    model = NULL;

    for (image = 0; image < images; ++image) {
        if (task->result_view(results[image], &view, &error) != TRTMC_OK)
            goto cleanup;
        if ((view.token_ids.size && !view.token_ids.data) || (view.text.size && !view.text.data)) {
            fprintf(stderr, "text result is incomplete\n");
            goto cleanup;
        }
        fputs("{\"task\":\"images_text_to_text\",\"token_ids\":[", stdout);
        for (i = 0; i < view.token_ids.size; ++i)
            printf("%s%" PRId32, i ? "," : "", view.token_ids.data[i]);
        fputs("],\"text_hex\":\"", stdout);
        for (i = 0; i < view.text.size; ++i)
            printf("%02x", (unsigned)(unsigned char)view.text.data[i]);
        puts("\"}");
    }
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
        for (image = 0; image < images; ++image)
            core->result_release(results[image]);
        core->model_release(model);
    }
    free(results);
    free(input);
    return status;
}

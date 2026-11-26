/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C SDK consumer for the family-owned end-to-end test.
 *
 *   BUNDLE RUNTIME_ROOT RGB_F32 HEIGHT WIDTH PROMPT MAX_NEW_TOKENS
 *
 * Runs images_text_to_text on one user message (image, then text) and prints
 * {"text": ..., "token_ids": [...]} as JSON.
 */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/language.h>
#include <trtmc/trtmc.h>

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}
static uint32_t dimension(const char* value) {
    char* end = NULL;
    const unsigned long parsed = strtoul(value, &end, 10);
    return (!end || *end || parsed == 0 || parsed > 65535UL) ? 0U : (uint32_t)parsed;
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

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_images_text_to_text_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_images_text_part_v1 parts[2];
    trtmc_images_text_message_v1 message;
    trtmc_images_text_to_text_request_v1 request = {0};
    trtmc_config_entry_v1 entry;
    trtmc_config_view_v1 config;
    trtmc_text_result_view_v1 view = {0};
    float* pixels = NULL;
    FILE* file = NULL;
    uint32_t height, width;
    size_t count;
    long long limit;
    uint64_t i;
    int status = 1;

    if (argc != 8) {
        fprintf(stderr,
                "Usage: %s BUNDLE RUNTIME_ROOT RGB_F32 HEIGHT WIDTH PROMPT MAX_NEW_TOKENS\n",
                argv[0]);
        return 2;
    }
    height = dimension(argv[4]);
    width = dimension(argv[5]);
    limit = strtoll(argv[7], NULL, 10);
    if (!height || !width || limit <= 0) {
        fprintf(stderr, "invalid image dimensions or token limit\n");
        return 2;
    }
    count = (size_t)height * width * 3;
    pixels = (float*)malloc(count * sizeof(float));
    file = fopen(argv[3], "rb");
    if (!pixels || !file || fread(pixels, sizeof(float), count, file) != count ||
        fgetc(file) != EOF || ferror(file)) {
        fprintf(stderr, "input must contain exactly HEIGHT x WIDTH x 3 float32 RGB values\n");
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
    if (core->model_get_task_api(model, text(TRTMC_TASK_IMAGES_TEXT_TO_TEXT), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*task)) {
        fprintf(stderr, "incomplete images_text_to_text C table\n");
        goto cleanup;
    }
    task = (const trtmc_images_text_to_text_api_v1*)header;

    parts[0].kind = TRTMC_MEDIA_IMAGE;
    parts[0].content.image = (trtmc_image_input_v1){pixels, count * sizeof(float), height, width,
                                                    3,      TRTMC_IMAGE_FLOAT32};
    parts[1].kind = TRTMC_MEDIA_TEXT;
    parts[1].content.text = text(argv[6]);
    message.role = TRTMC_ROLE_USER;
    message.parts = parts;
    message.part_count = 2;
    request.messages = &message;
    request.message_count = 1;
    entry.name = text("max_new_tokens");
    entry.value.kind = TRTMC_CONFIG_I64;
    entry.value.as.i64 = limit;
    config.entries = &entry;
    config.count = 1;
    if (task->run(model, &request, &config, &result, &error) != TRTMC_OK)
        goto cleanup;
    /* The result must stay readable after its input and model are gone. */
    free(pixels);
    pixels = NULL;
    core->model_release(model);
    model = NULL;
    if (task->result_view(result, &view, &error) != TRTMC_OK)
        goto cleanup;
    if (!view.token_ids.size || view.token_ids.size > (uint64_t)limit) {
        fprintf(stderr, "images_text_to_text must return 1..MAX_NEW_TOKENS generated tokens\n");
        goto cleanup;
    }
    printf("{\"text\":");
    json_string(view.text);
    printf(",\"token_ids\":[");
    for (i = 0; i < view.token_ids.size; ++i)
        printf(i ? ",%" PRId32 : "%" PRId32, view.token_ids.data[i]);
    puts("]}");
    status = 0;

cleanup:
    if (error) {
        const trtmc_string_view message_text = core->error_message(error);
        fprintf(stderr, "%.*s\n", (int)message_text.size, message_text.data);
        core->error_release(error);
    }
    if (result)
        core->result_release(result);
    if (model)
        core->model_release(model);
    if (file)
        fclose(file);
    free(pixels);
    return status;
}

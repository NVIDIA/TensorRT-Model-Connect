/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C SDK consumer for the family-owned end-to-end test.
 *
 *   BUNDLE RUNTIME_ROOT MAX_NEW_TOKENS PROMPT IMAGE.rgb HEIGHT WIDTH [IMAGE.rgb HEIGHT WIDTH]...
 *
 * Each IMAGE.rgb holds packed HWC RGB8 pixels. Prints one JSON line with the
 * generated text and token ids.
 */
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

static unsigned char* read_pixels(const char* path, uint64_t expected) {
    unsigned char* pixels = NULL;
    FILE* file = fopen(path, "rb");
    if (!file)
        return NULL;
    pixels = (unsigned char*)malloc((size_t)expected + 1);
    if (pixels && (fread(pixels, 1, (size_t)expected + 1, file) != (size_t)expected)) {
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
    trtmc_load_options_v1 options;
    trtmc_images_text_part_v1* parts = NULL;
    unsigned char** pixels = NULL;
    trtmc_images_text_message_v1 message;
    trtmc_images_text_to_text_request_v1 request;
    trtmc_config_entry_v1 entry;
    trtmc_config_view_v1 config;
    trtmc_text_result_view_v1 view;
    uint64_t images = 0, i;
    int status = 1;
    if (argc < 8 || (argc - 5) % 3 != 0) {
        fprintf(stderr,
                "Usage: %s BUNDLE RUNTIME_ROOT MAX_NEW_TOKENS PROMPT IMAGE.rgb HEIGHT WIDTH...\n",
                argv[0]);
        return 2;
    }
    images = (uint64_t)(argc - 5) / 3;
    memset(&options, 0, sizeof(options));
    memset(&message, 0, sizeof(message));
    memset(&request, 0, sizeof(request));
    memset(&entry, 0, sizeof(entry));
    memset(&config, 0, sizeof(config));
    memset(&view, 0, sizeof(view));
    parts = (trtmc_images_text_part_v1*)calloc((size_t)images + 1, sizeof(*parts));
    pixels = (unsigned char**)calloc((size_t)images, sizeof(*pixels));
    if (!parts || !pixels)
        goto cleanup;
    for (i = 0; i < images; ++i) {
        const char* path = argv[5 + 3 * i];
        const uint32_t height = (uint32_t)strtoul(argv[6 + 3 * i], NULL, 10);
        const uint32_t width = (uint32_t)strtoul(argv[7 + 3 * i], NULL, 10);
        const uint64_t bytes = (uint64_t)height * width * 3;
        pixels[i] = read_pixels(path, bytes);
        if (!pixels[i]) {
            fprintf(stderr, "%s does not hold HEIGHT x WIDTH RGB8 pixels\n", path);
            goto cleanup;
        }
        parts[i].kind = TRTMC_MEDIA_IMAGE;
        parts[i].content.image.data = pixels[i];
        parts[i].content.image.byte_size = bytes;
        parts[i].content.image.height = height;
        parts[i].content.image.width = width;
        parts[i].content.image.channels = 3;
        parts[i].content.image.format = TRTMC_IMAGE_UINT8;
    }
    parts[images].kind = TRTMC_MEDIA_TEXT;
    parts[images].content.text = text(argv[4]);
    message.role = TRTMC_ROLE_USER;
    message.parts = parts;
    message.part_count = images + 1;
    request.messages = &message;
    request.message_count = 1;
    entry.name = text("max_new_tokens");
    entry.value.kind = TRTMC_CONFIG_I64;
    entry.value.as.i64 = strtoll(argv[3], NULL, 10);
    config.entries = &entry;
    config.count = 1;

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
    if (task->run(model, &request, &config, &result, &error) != TRTMC_OK)
        goto cleanup;
    if (task->result_view(result, &view, &error) != TRTMC_OK)
        goto cleanup;
    printf("{\"task\":\"images_text_to_text\",\"text\":");
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
    if (pixels)
        for (i = 0; i < images; ++i)
            free(pixels[i]);
    free(pixels);
    free(parts);
    return status;
}

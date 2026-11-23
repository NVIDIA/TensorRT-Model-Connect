/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * C SDK consumer of a Gemma 3n bundle: one images_text_to_text request with
 * one 8-bit RGB image and a text prompt, greedy decoding. Prints one JSON line
 * {"text": ..., "token_ids": [...]}.
 *
 * Usage: test_gemma3n_sdk_c <bundle> <runtime_root> <rgb_file> <height> <width>
 *                           <prompt> <max_new_tokens>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/image.h>
#include <trtmc/language.h>
#include <trtmc/trtmc.h>

static void check(trtmc_status status, const trtmc_core_api_v1* api, trtmc_error** error,
                  const char* label) {
    if (status == TRTMC_OK)
        return;
    if (api != NULL && *error != NULL) {
        trtmc_string_view message = api->error_message(*error);
        fprintf(stderr, "%s failed: %.*s\n", label, (int)message.size,
                message.data != NULL ? message.data : "");
        api->error_release(*error);
    } else {
        fprintf(stderr, "%s failed\n", label);
    }
    exit(1);
}

static trtmc_string_view view(const char* text) {
    trtmc_string_view result;
    result.data = text;
    result.size = strlen(text);
    return result;
}

static unsigned char* read_file(const char* path, size_t expected) {
    FILE* file = fopen(path, "rb");
    unsigned char* data = (unsigned char*)malloc(expected > 0 ? expected : 1);
    size_t read = 0;
    if (file == NULL || data == NULL) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(1);
    }
    read = fread(data, 1, expected, file);
    if (read != expected || fgetc(file) != EOF) {
        fprintf(stderr, "%s does not hold %zu RGB bytes\n", path, expected);
        exit(1);
    }
    fclose(file);
    return data;
}

static void print_json_string(trtmc_string_view text) {
    uint64_t i;
    putchar('"');
    for (i = 0; i < text.size; ++i) {
        const unsigned char c = (unsigned char)text.data[i];
        if (c == '"' || c == '\\')
            printf("\\%c", c);
        else if (c < 0x20)
            printf("\\u%04x", c);
        else
            putchar(c);
    }
    putchar('"');
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* api = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options;
    trtmc_model* model = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_images_text_to_text_api_v1* task = NULL;
    trtmc_images_text_part_v1 parts[2];
    trtmc_images_text_message_v1 message;
    trtmc_images_text_to_text_request_v1 request;
    trtmc_config_entry_v1 entry;
    trtmc_config_view_v1 config;
    trtmc_result* result = NULL;
    trtmc_text_result_view_v1 output;
    unsigned char* pixels = NULL;
    unsigned long height = 0, width = 0;
    uint64_t i;

    if (argc != 8) {
        fprintf(stderr,
                "usage: %s <bundle> <runtime_root> <rgb_file> <height> <width> <prompt> "
                "<max_new_tokens>\n",
                argv[0]);
        return 2;
    }
    height = strtoul(argv[4], NULL, 10);
    width = strtoul(argv[5], NULL, 10);
    pixels = read_file(argv[3], (size_t)height * width * 3U);

    check(trtmc_get_api(1, 0, &api), NULL, &error, "trtmc_get_api");
    memset(&options, 0, sizeof(options));
    options.struct_size = sizeof(options);
    options.runtime_root = view(argv[2]);
    check(api->model_load(view(argv[1]), &options, &model, &error), api, &error, "model_load");
    check(
        api->model_get_task_api(model, view(TRTMC_TASK_IMAGES_TEXT_TO_TEXT), 1, 0, &header, &error),
        api, &error, "images_text_to_text api");
    task = (const trtmc_images_text_to_text_api_v1*)header;

    memset(parts, 0, sizeof(parts));
    parts[0].kind = TRTMC_MEDIA_IMAGE;
    parts[0].content.image.data = pixels;
    parts[0].content.image.byte_size = (uint64_t)height * width * 3U;
    parts[0].content.image.height = (uint32_t)height;
    parts[0].content.image.width = (uint32_t)width;
    parts[0].content.image.channels = 3;
    parts[0].content.image.format = TRTMC_IMAGE_UINT8;
    parts[1].kind = TRTMC_MEDIA_TEXT;
    parts[1].content.text = view(argv[6]);
    memset(&message, 0, sizeof(message));
    message.role = TRTMC_ROLE_USER;
    message.parts = parts;
    message.part_count = 2;
    memset(&request, 0, sizeof(request));
    request.messages = &message;
    request.message_count = 1;
    memset(&entry, 0, sizeof(entry));
    entry.name = view("max_new_tokens");
    entry.value.kind = TRTMC_CONFIG_I64;
    entry.value.as.i64 = strtoll(argv[7], NULL, 10);
    config.entries = &entry;
    config.count = 1;

    check(task->run(model, &request, &config, &result, &error), api, &error, "run");
    memset(&output, 0, sizeof(output));
    check(task->result_view(result, &output, &error), api, &error, "result_view");
    if (output.text.size == 0 || output.token_ids.size == 0) {
        fprintf(stderr, "images_text_to_text returned an empty result\n");
        return 1;
    }
    printf("{\"text\": ");
    print_json_string(output.text);
    printf(", \"token_ids\": [");
    for (i = 0; i < output.token_ids.size; ++i)
        printf("%s%d", i ? ", " : "", output.token_ids.data[i]);
    printf("]}\n");

    api->result_release(result);
    api->model_release(model);
    free(pixels);
    return 0;
}

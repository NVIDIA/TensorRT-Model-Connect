/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/lighton_ocr/tests/cpp/image_io.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/trtmc.h>

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}

static void json_string(FILE* output, trtmc_string_view value) {
    uint64_t i;
    fputc('"', output);
    for (i = 0; i < value.size; ++i) {
        const unsigned char byte = (unsigned char)value.data[i];
        if (byte == '"' || byte == '\\') {
            fputc('\\', output);
            fputc(byte, output);
        } else if (byte < 32) {
            fprintf(output, "\\u%04x", (unsigned)byte);
        } else {
            fputc(byte, output);
        }
    }
    fputc('"', output);
}

static int64_t budget(const char* path) {
    char buffer[4096] = {0};
    int consumed = 0;
    int64_t maximum = 0;
    FILE* input = fopen(path, "rb");
    size_t count;
    if (!input)
        return -1;
    count = fread(buffer, 1, sizeof(buffer) - 1, input);
    if (ferror(input) || !feof(input)) {
        fclose(input);
        return -1;
    }
    fclose(input);
    buffer[count] = 0;
    if (sscanf(buffer, " { \"max_new_tokens\" : %" SCNd64 " } %n", &maximum, &consumed) != 1 ||
        consumed <= 0 || buffer[consumed] || maximum <= 0)
        return -1;
    return maximum;
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_images_text_to_text_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_config_entry_v1 entry = {0};
    trtmc_config_view_v1 config = {0};
    trtmc_images_text_part_v1 parts[2] = {0};
    trtmc_images_text_message_v1 message = {0};
    trtmc_images_text_to_text_request_v1 request = {0};
    trtmc_text_result_view_v1 view = {0};
    unsigned char* pixels = NULL;
    FILE* output = NULL;
    int width = 0, height = 0, status = 1;
    int64_t maximum;
    uint64_t i;
    if (argc != 6)
        return 2;
    maximum = budget(argv[4]);
    if (maximum <= 0) {
        fprintf(stderr, "C consumer requires a positive max_new_tokens JSON request\n");
        goto cleanup;
    }
    pixels = lighton_ocr_load_rgb(argv[3], &width, &height);
    if (!pixels || trtmc_get_api(1, 0, &core) != TRTMC_OK || !core)
        goto cleanup;
    options.struct_size = sizeof(options);
    options.runtime_root = text(argv[2]);
    if (core->model_load(text(argv[1]), &options, &model, &error) != TRTMC_OK)
        goto cleanup;
    if (core->model_get_task_api(model, text(TRTMC_TASK_IMAGES_TEXT_TO_TEXT), 1, 0, &header,
                                 &error) != TRTMC_OK ||
        !header || header->byte_size < sizeof(*task))
        goto cleanup;
    task = (const trtmc_images_text_to_text_api_v1*)header;
    entry.name = text("max_new_tokens");
    entry.value.kind = TRTMC_CONFIG_I64;
    entry.value.as.i64 = maximum;
    config.entries = &entry;
    config.count = 1;
    parts[0].kind = TRTMC_MEDIA_IMAGE;
    parts[0].content.image.data = pixels;
    parts[0].content.image.byte_size = (uint64_t)width * height * 3;
    parts[0].content.image.height = (uint32_t)height;
    parts[0].content.image.width = (uint32_t)width;
    parts[0].content.image.channels = 3;
    parts[0].content.image.format = TRTMC_IMAGE_UINT8;
    parts[1].kind = TRTMC_MEDIA_TEXT;
    parts[1].content.text = text("");
    message.role = TRTMC_ROLE_USER;
    message.parts = parts;
    message.part_count = 2;
    request.messages = &message;
    request.message_count = 1;
    if (task->run(model, &request, &config, &result, &error) != TRTMC_OK)
        goto cleanup;
    core->model_release(model);
    model = NULL;
    if (task->result_view(result, &view, &error) != TRTMC_OK)
        goto cleanup;
    output = fopen(argv[5], "wb");
    if (!output)
        goto cleanup;
    fputs("{\"text\":", output);
    json_string(output, view.text);
    fputs(",\"token_ids\":[", output);
    for (i = 0; i < view.token_ids.size; ++i)
        fprintf(output, "%s%" PRId32, i ? "," : "", view.token_ids.data[i]);
    fprintf(output,
            "],\"prefill_ms\":%.17g,\"decode_ms\":%.17g,"
            "\"result_retained_after_model_release\":true}\n",
            view.prefill_ms, view.decode_ms);
    status = fflush(output) == 0 && !ferror(output) ? 0 : 1;
cleanup:
    if (error && core) {
        const trtmc_string_view value = core->error_message(error);
        fwrite(value.data, 1, (size_t)value.size, stderr);
        fputc('\n', stderr);
        core->error_release(error);
    }
    if (output)
        fclose(output);
    lighton_ocr_free_rgb(pixels);
    if (core) {
        core->result_release(result);
        core->model_release(model);
    }
    return status;
}

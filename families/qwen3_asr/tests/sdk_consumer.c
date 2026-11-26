/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C11 SDK consumer for the family-owned Qwen3-ASR end-to-end test.
 * Transcribes caller-supplied float32 PCM through speech_transcription, releases the model, then
 * prints the owned transcript and token ids as one JSON line. */
#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/audio.h>
#include <trtmc/trtmc.h>

_Static_assert(sizeof(float) == 4, "PCM uses float32");

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}

static int same(trtmc_string_view left, const char* right) {
    return left.size == strlen(right) && memcmp(left.data, right, (size_t)left.size) == 0;
}

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

/* JSON string body; UTF-8 bytes pass through, control characters are escaped. */
static void print_json_string(trtmc_string_view value) {
    uint64_t index;
    putchar('"');
    for (index = 0; index < value.size; ++index) {
        const unsigned char c = (unsigned char)value.data[index];
        if (c == '"' || c == '\\')
            printf("\\%c", c);
        else if (c < 0x20)
            printf("\\u%04x", c);
        else
            putchar(c);
    }
    putchar('"');
}

static int parse_u32(const char* value, uint32_t* out) {
    char* end = NULL;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(value, &end, 10);
    if (errno || end == value || *end || parsed == 0 || parsed > UINT32_MAX)
        return 0;
    *out = (uint32_t)parsed;
    return 1;
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_speech_transcription_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_speech_transcription_request_v1 request = {0};
    trtmc_config_entry_v1 entry = {0};
    trtmc_config_view_v1 config = {0};
    trtmc_text_result_view_v1 view = {0};
    float* pcm = NULL;
    size_t sample_count = 0;
    uint64_t field_count = 0, index;
    uint32_t rate = 0, channels = 0;
    long long limit;
    char* end = NULL;
    int declares_limit = 0;
    int status = 1;

    if (argc != 7 && argc != 8) {
        fprintf(stderr,
                "Usage: %s BUNDLE RUNTIME_ROOT PCM_F32 RATE CHANNELS MAX_OUTPUT_TOKENS "
                "[SOURCE_LANGUAGE]\n",
                argv[0]);
        return 2;
    }
    errno = 0;
    limit = strtoll(argv[6], &end, 10);
    if (!parse_u32(argv[4], &rate) || !parse_u32(argv[5], &channels) || errno || end == argv[6] ||
        *end || limit < 1) {
        fprintf(stderr, "RATE, CHANNELS and MAX_OUTPUT_TOKENS must be positive integers\n");
        return 2;
    }
    pcm = read_floats(argv[3], &sample_count);
    if (!pcm) {
        fprintf(stderr, "PCM_F32 must be a non-empty float32 file\n");
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
    if (core->model_get_task_api(model, text(TRTMC_TASK_SPEECH_TRANSCRIPTION), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*task)) {
        fprintf(stderr, "incomplete speech_transcription C table\n");
        goto cleanup;
    }
    task = (const trtmc_speech_transcription_api_v1*)header;

    if (core->config_field_count(model, text(TRTMC_TASK_SPEECH_TRANSCRIPTION), 1, 0, &field_count,
                                 &error) != TRTMC_OK)
        goto cleanup;
    for (index = 0; index < field_count; ++index) {
        trtmc_config_field_v1 field;
        if (core->config_field_info(model, text(TRTMC_TASK_SPEECH_TRANSCRIPTION), 1, 0, index,
                                    &field, &error) != TRTMC_OK)
            goto cleanup;
        if (same(field.name, "max_output_tokens") && field.kind == TRTMC_CONFIG_I64)
            declares_limit = 1;
    }
    if (!declares_limit) {
        fprintf(stderr, "Qwen3-ASR must declare the integer Config field max_output_tokens\n");
        goto cleanup;
    }

    entry.name = text("max_output_tokens");
    entry.value.kind = TRTMC_CONFIG_I64;
    entry.value.as.i64 = (int64_t)limit;
    config.entries = &entry;
    config.count = 1;
    request.audio.samples = pcm;
    request.audio.sample_count = (uint64_t)sample_count;
    request.audio.has_sample_rate = 1;
    request.audio.sample_rate = rate;
    request.audio.channels = channels;
    if (argc == 8) {
        request.has_source_language = 1;
        request.source_language = text(argv[7]);
    }
    if (task->run(model, &request, &config, &result, &error) != TRTMC_OK)
        goto cleanup;
    /* The result must stay readable after its inputs and model are gone. */
    free(pcm);
    pcm = NULL;
    core->model_release(model);
    model = NULL;

    if (task->result_view(result, &view, &error) != TRTMC_OK)
        goto cleanup;
    printf("{\"task\":\"speech_transcription\",\"text\":");
    print_json_string(view.text);
    printf(",\"token_ids\":[");
    for (index = 0; index < view.token_ids.size; ++index)
        printf("%s%" PRId32, index ? "," : "", view.token_ids.data[index]);
    printf("]}\n");
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
    free(pcm);
    return status;
}

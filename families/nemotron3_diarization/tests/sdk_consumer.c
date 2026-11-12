/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C11 SDK consumer for the family-owned Nemotron 3 Diarization E2E.
 * Reads mono float32 PCM, runs audio_to_timed_labels with the family defaults,
 * releases the model, then prints the owned result in the CLI's detect --audio
 * JSON layout. */
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/audio.h>
#include <trtmc/trtmc.h>

_Static_assert(sizeof(float) == 4, "input uses float32");

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}

static uint32_t positive(const char* value) {
    char* end = NULL;
    uintmax_t parsed;
    errno = 0;
    parsed = strtoumax(value, &end, 10);
    if (errno || end == value || *end || !parsed || parsed > UINT32_MAX)
        return 0;
    return (uint32_t)parsed;
}

static float* read_pcm(const char* path, size_t* count) {
    FILE* file = fopen(path, "rb");
    float* samples = NULL;
    long bytes;
    if (!file)
        return NULL;
    if (fseek(file, 0, SEEK_END) == 0 && (bytes = ftell(file)) > 0 &&
        bytes % (long)sizeof(float) == 0 && fseek(file, 0, SEEK_SET) == 0) {
        *count = (size_t)bytes / sizeof(float);
        samples = (float*)malloc((size_t)bytes);
        if (samples && fread(samples, sizeof(float), *count, file) != *count) {
            free(samples);
            samples = NULL;
        }
    }
    fclose(file);
    return samples;
}

static int valid_view(const trtmc_timed_labels_view_v1* view) {
    uint64_t i;
    if (!view->labels.size || (view->interval_count && !view->intervals))
        return 0;
    if (view->frame_scores.rows * view->frame_scores.columns != view->frame_scores.count ||
        view->frame_scores.columns != view->labels.size || view->score_kind != TRTMC_SCORE_LOGIT ||
        !(view->frame_seconds > 0))
        return 0;
    for (i = 0; i < view->interval_count; ++i) {
        const trtmc_timed_label_v1* item = &view->intervals[i];
        if (!(item->end_seconds > item->start_seconds) || item->label >= view->labels.size)
            return 0;
    }
    for (i = 0; i < view->frame_scores.count; ++i)
        if (!isfinite(view->frame_scores.data[i]))
            return 0;
    return 1;
}

static void print_json(const trtmc_timed_labels_view_v1* view) {
    uint64_t i, j;
    fputs("{\"task\":\"audio_to_timed_labels\",\"labels\":[", stdout);
    for (i = 0; i < view->labels.size; ++i)
        printf("%s\"%.*s\"", i ? "," : "", (int)view->labels.data[i].size,
               view->labels.data[i].data);
    printf("],\"vocabulary_id\":\"%.*s\",\"intervals\":[", (int)view->vocabulary_id.size,
           view->vocabulary_id.data ? view->vocabulary_id.data : "");
    for (i = 0; i < view->interval_count; ++i)
        printf("%s{\"start_seconds\":%.17g,\"end_seconds\":%.17g,\"label\":%" PRIu32 "}",
               i ? "," : "", view->intervals[i].start_seconds, view->intervals[i].end_seconds,
               view->intervals[i].label);
    printf("],\"frame_seconds\":%.17g,\"score_kind\":\"logit\",\"frame_scores\":[",
           view->frame_seconds);
    for (i = 0; i < view->frame_scores.rows; ++i) {
        fputs(i ? ",[" : "[", stdout);
        for (j = 0; j < view->frame_scores.columns; ++j)
            printf("%s%.*g", j ? "," : "", FLT_DECIMAL_DIG,
                   (double)view->frame_scores.data[i * view->frame_scores.columns + j]);
        putchar(']');
    }
    puts("]}");
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_audio_to_timed_labels_api_v1* task = NULL;
    trtmc_model* model = NULL;
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_audio_to_timed_labels_request_v1 request = {0};
    trtmc_timed_labels_view_v1 view = {0};
    float* samples = NULL;
    size_t count = 0;
    uint32_t sample_rate;
    uint64_t field_count = 0;
    int status = 1;

    if (argc != 5) {
        fprintf(stderr, "Usage: %s BUNDLE RUNTIME_ROOT PCM_F32 SAMPLE_RATE\n", argv[0]);
        return 2;
    }
    sample_rate = positive(argv[4]);
    samples = read_pcm(argv[3], &count);
    if (!sample_rate || !samples) {
        fprintf(stderr, "input must be nonempty mono float32 PCM with a positive sample rate\n");
        goto cleanup;
    }
    if (trtmc_get_api(1, 0, &core) != TRTMC_OK || !core) {
        fprintf(stderr, "unable to obtain the v1 C API\n");
        goto cleanup;
    }
    options.struct_size = sizeof(options);
    options.runtime_root = text(argv[2]);
    if (core->model_load(text(argv[1]), &options, &model, &error) != TRTMC_OK)
        goto cleanup;
    if (core->model_get_task_api(model, text(TRTMC_TASK_AUDIO_TO_TIMED_LABELS), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*task)) {
        fprintf(stderr, "incomplete audio_to_timed_labels C table\n");
        goto cleanup;
    }
    task = (const trtmc_audio_to_timed_labels_api_v1*)header;
    if (core->config_field_count(model, text(TRTMC_TASK_AUDIO_TO_TIMED_LABELS), 1, 0, &field_count,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (field_count != 1) {
        fprintf(stderr, "Nemotron 3 Diarization must expose exactly its threshold Config\n");
        goto cleanup;
    }

    request.audio = (trtmc_audio_view_v1){samples, (uint64_t)count, 1, sample_rate, 1};
    if (task->run(model, &request, NULL, &result, &error) != TRTMC_OK)
        goto cleanup;
    /* The result must stay readable after its input and model are gone. */
    free(samples);
    samples = NULL;
    core->model_release(model);
    model = NULL;

    if (task->result_view(result, &view, &error) != TRTMC_OK)
        goto cleanup;
    if (!valid_view(&view)) {
        fprintf(stderr, "timed labels result breaks its contract\n");
        goto cleanup;
    }
    print_json(&view);
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
    free(samples);
    return status;
}

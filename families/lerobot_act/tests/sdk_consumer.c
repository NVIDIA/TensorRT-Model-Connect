/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Public C SDK consumer for the family-owned end-to-end test. It predicts one
 * chunk from a UInt8 image, replays it through the action queue, refills and
 * resets the queue, and prints the chunk after releasing the model. */
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <trtmc/action.h>
#include <trtmc/trtmc.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../../../third_party/stb/stb_image.h"

static trtmc_string_view text(const char* value) {
    const trtmc_string_view result = {value, (uint64_t)strlen(value)};
    return result;
}

static int equals(trtmc_string_view value, const char* expected) {
    const size_t size = strlen(expected);
    return value.size == size && (size == 0 || memcmp(value.data, expected, size) == 0);
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

static float* read_floats(const char* path, uint64_t* count) {
    FILE* file = fopen(path, "rb");
    float* values = NULL;
    long bytes = 0;
    *count = 0;
    if (!file)
        return NULL;
    if (fseek(file, 0, SEEK_END) == 0)
        bytes = ftell(file);
    if (bytes > 0 && bytes % (long)sizeof(float) == 0 && fseek(file, 0, SEEK_SET) == 0) {
        values = (float*)malloc((size_t)bytes);
        if (values && fread(values, 1, (size_t)bytes, file) == (size_t)bytes) {
            *count = (uint64_t)bytes / sizeof(float);
        } else {
            free(values);
            values = NULL;
        }
    }
    fclose(file);
    return values;
}

/* Exact equality: every queued step replays the chunk the same engine produced. */
static int same_row(const trtmc_f32_matrix_view_v1* chunk, uint64_t row,
                    const trtmc_f32_matrix_view_v1* step) {
    return step->rows == 1 && step->columns == chunk->columns && step->count == chunk->columns &&
           memcmp(step->data, chunk->data + row * chunk->columns,
                  (size_t)chunk->columns * sizeof(float)) == 0;
}

static int same_matrix(const trtmc_f32_matrix_view_v1* left,
                       const trtmc_f32_matrix_view_v1* right) {
    return left->rows == right->rows && left->columns == right->columns &&
           left->count == right->count &&
           memcmp(left->data, right->data, (size_t)left->count * sizeof(float)) == 0;
}

static int valid_chunk(const trtmc_image_state_action_chunk_view_v1* view) {
    const trtmc_action_sequence_view_v1* actions = &view->actions;
    uint64_t i;
    if (!actions->values.rows || !actions->values.columns ||
        actions->values.count != actions->values.rows * actions->values.columns ||
        !(view->inference_ms > 0) || !equals(actions->schema.domain, "lerobot.act") ||
        !equals(actions->schema.normalization, "unnormalized"))
        return 0;
    for (i = 0; i < actions->values.count; ++i) {
        if (!isfinite(actions->values.data[i]))
            return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    const trtmc_core_api_v1* core = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_image_state_to_action_chunk_api_v1* chunk_api = NULL;
    const trtmc_image_state_action_queue_api_v1* queue_api = NULL;
    trtmc_model* model = NULL;
    trtmc_image_state_action_session* session = NULL;
    trtmc_result* chunk = NULL;
    trtmc_result* again = NULL;
    trtmc_result* step = NULL;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    trtmc_image_state_to_action_chunk_request_v1 request;
    trtmc_image_state_action_chunk_view_v1 view;
    trtmc_image_state_action_chunk_view_v1 again_view;
    trtmc_action_step_view_v1 step_view;
    unsigned char* pixels = NULL;
    float* state = NULL;
    uint64_t state_count = 0, field_count = 0, rows = 0, i, steps = 0;
    int width = 0, height = 0, channels = 0, status = 1;
    memset(&request, 0, sizeof(request));
    if (argc != 5) {
        fprintf(stderr, "Usage: %s BUNDLE RUNTIME_ROOT IMAGE STATE\n", argv[0]);
        return 2;
    }
    pixels = stbi_load(argv[3], &width, &height, &channels, 3);
    state = read_floats(argv[4], &state_count);
    if (!pixels || !state) {
        fprintf(stderr, "unable to read the recorded image or state\n");
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
    if (core->model_get_task_api(model, text(TRTMC_TASK_IMAGE_STATE_TO_ACTION_CHUNK), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*chunk_api)) {
        fprintf(stderr, "incomplete action chunk C table\n");
        goto cleanup;
    }
    chunk_api = (const trtmc_image_state_to_action_chunk_api_v1*)header;
    if (core->model_get_task_api(model, text(TRTMC_TASK_IMAGE_STATE_ACTION_QUEUE), 1, 0, &header,
                                 &error) != TRTMC_OK)
        goto cleanup;
    if (!header || header->byte_size < sizeof(*queue_api)) {
        fprintf(stderr, "incomplete action queue C table\n");
        goto cleanup;
    }
    queue_api = (const trtmc_image_state_action_queue_api_v1*)header;
    if (core->config_field_count(model, text(TRTMC_TASK_IMAGE_STATE_TO_ACTION_CHUNK), 1, 0,
                                 &field_count, &error) != TRTMC_OK)
        goto cleanup;
    if (field_count == 0 &&
        core->config_field_count(model, text(TRTMC_TASK_IMAGE_STATE_ACTION_QUEUE), 1, 0,
                                 &field_count, &error) != TRTMC_OK)
        goto cleanup;
    if (field_count != 0) {
        fprintf(stderr, "LeRobot ACT must expose no runtime Config fields\n");
        goto cleanup;
    }

    request.observation.image.data = pixels;
    request.observation.image.byte_size = (uint64_t)width * (uint64_t)height * 3U;
    request.observation.image.height = (uint32_t)height;
    request.observation.image.width = (uint32_t)width;
    request.observation.image.channels = 3;
    request.observation.image.format = TRTMC_IMAGE_UINT8;
    request.observation.state = state;
    request.observation.state_count = state_count;
    if (chunk_api->run(model, &request, NULL, &chunk, &error) != TRTMC_OK ||
        chunk_api->result_view(chunk, &view, &error) != TRTMC_OK)
        goto cleanup;
    if (!valid_chunk(&view)) {
        fprintf(stderr, "action chunk is incomplete, non-finite or mislabeled\n");
        goto cleanup;
    }
    rows = view.actions.values.rows;

    /* Every chunk step, one refill after the chunk is spent, and one step after reset. */
    if (queue_api->create(model, NULL, &session, &error) != TRTMC_OK)
        goto cleanup;
    for (i = 0; i <= rows + 1; ++i) {
        const int started = i == 0 || i >= rows;
        if (i == rows + 1 && queue_api->reset(session, &error) != TRTMC_OK)
            goto cleanup;
        if (queue_api->act(session, &request.observation, NULL, &step, &error) != TRTMC_OK ||
            queue_api->result_view(step, &step_view, &error) != TRTMC_OK)
            goto cleanup;
        if (step_view.started_new_chunk != (uint32_t)started ||
            !same_row(&view.actions.values, i < rows ? i : 0, &step_view.action.values) ||
            (started ? !(step_view.inference_ms > 0) : step_view.inference_ms != 0)) {
            fprintf(stderr, "queued step %" PRIu64 " does not replay the action chunk\n", i);
            goto cleanup;
        }
        core->result_release(step);
        step = NULL;
        ++steps;
        if (i != 0)
            continue;
        /* An independent chunk neither consumes nor replaces the live queue. */
        if (chunk_api->run(model, &request, NULL, &again, &error) != TRTMC_OK ||
            chunk_api->result_view(again, &again_view, &error) != TRTMC_OK)
            goto cleanup;
        if (!same_matrix(&view.actions.values, &again_view.actions.values)) {
            fprintf(stderr, "repeated action chunk differs\n");
            goto cleanup;
        }
        core->result_release(again);
        again = NULL;
    }
    queue_api->release(session);
    session = NULL;
    core->model_release(model);
    model = NULL;

    /* The chunk result stays readable after the session and model are released. */
    if (chunk_api->result_view(chunk, &view, &error) != TRTMC_OK)
        goto cleanup;
    printf("{\"task\":\"image_state_to_action_chunk\",\"image_format\":\"uint8\"");
    printf(",\"rows\":%" PRIu64 ",\"columns\":%" PRIu64, view.actions.values.rows,
           view.actions.values.columns);
    printf(",\"within_training_bounds\":%s", view.within_training_bounds ? "true" : "false");
    printf(",\"queue_steps\":%" PRIu64 ",\"schema\":{\"domain\":", steps);
    json_string(view.actions.schema.domain);
    printf(",\"normalization\":");
    json_string(view.actions.schema.normalization);
    printf(",\"component_names\":%" PRIu64 ",\"units\":%" PRIu64 "}",
           view.actions.schema.component_names.size, view.actions.schema.units.size);
    printf(",\"timestamps\":%" PRIu64 ",\"frame_spans\":%" PRIu64 ",\"actions\":[",
           view.actions.timestamps_seconds.size, view.actions.frame_span_count);
    for (i = 0; i < view.actions.values.count; ++i) {
        if (i)
            putchar(',');
        printf("%.*g", FLT_DECIMAL_DIG, (double)view.actions.values.data[i]);
    }
    puts("]}");
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
        core->result_release(step);
        core->result_release(again);
        if (queue_api)
            queue_api->release(session);
        core->result_release(chunk);
        core->model_release(model);
    }
    stbi_image_free(pixels);
    free(state);
    return status;
}

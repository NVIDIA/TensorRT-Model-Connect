/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/action.h"
#include "trtmc/trtmc.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures;
static void check(int condition, const char* message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}
static trtmc_string_view str(const char* value) {
    trtmc_string_view out = {value, strlen(value)};
    return out;
}
static int equal(trtmc_string_view value, const char* expected) {
    return value.size == strlen(expected) && memcmp(value.data, expected, (size_t)value.size) == 0;
}
static void clear_error(const trtmc_core_api_v1* core, trtmc_error** error) {
    core->error_release(*error);
    *error = NULL;
}
static int bundle(const char* path, const char* task) {
    char header[512];
    const int count =
        snprintf(header, sizeof(header),
                 "{\"format\":1,\"family\":\"action_fixture\",\"task\":\"%s\",\"backend\":\"fake\","
                 "\"sections\":{\"engine.plan\":{\"offset\":0,\"length\":4}}}",
                 task);
    if (count < 0 || (size_t)count >= sizeof(header))
        return 0;
    FILE* out = fopen(path, "wb");
    if (!out)
        return 0;
    int ok = fwrite("BUNDLE\x01\x00", 1, 8, out) == 8;
    for (unsigned shift = 0; shift < 64; shift += 8)
        ok = (fputc((int)(((uint64_t)count >> shift) & 255U), out) != EOF) && ok;
    ok = (fwrite(header, 1, (size_t)count, out) == (size_t)count) && ok;
    ok = (fwrite("PLAN", 1, 4, out) == 4) && ok;
    return fclose(out) == 0 && ok;
}
int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    char path[4096], disabled_path[4096];
    if (snprintf(path, sizeof(path), "%s/action_c_all.bundle", argv[1]) >= (int)sizeof(path) ||
        snprintf(disabled_path, sizeof(disabled_path), "%s/action_c_none.bundle", argv[1]) >=
            (int)sizeof(disabled_path))
        return 2;
    if (!bundle(path, "all") || !bundle(disabled_path, "none"))
        return 2;
    const trtmc_core_api_v1* core = NULL;
    if (trtmc_get_api(1, 0, &core) != TRTMC_OK)
        return 2;
    trtmc_error* error = NULL;
    trtmc_load_options_v1 options = {0};
    options.struct_size = sizeof(options);
    options.runtime_root = str(argv[1]);
    trtmc_model *model = NULL, *disabled = NULL;
    if (core->model_load(str(path), &options, &model, &error) != TRTMC_OK ||
        core->model_load(str(disabled_path), &options, &disabled, &error) != TRTMC_OK)
        return 2;
    const trtmc_api_header* header = NULL;
    if (core->model_get_task_api(model, str(TRTMC_TASK_IMAGE_STATE_TO_ACTION_CHUNK), 1, 0, &header,
                                 &error) != TRTMC_OK)
        return 2;
    const trtmc_image_state_to_action_chunk_api_v1* chunks =
        (const trtmc_image_state_to_action_chunk_api_v1*)header;
    if (core->model_get_task_api(model, str(TRTMC_TASK_IMAGE_STATE_ACTION_QUEUE), 1, 0, &header,
                                 &error) != TRTMC_OK)
        return 2;
    const trtmc_image_state_action_queue_api_v1* queue =
        (const trtmc_image_state_action_queue_api_v1*)header;
    const float pixels[] = {0.5F, 0.25F, 0.125F};
    float state[] = {2, 4};
    trtmc_image_state_observation_v1 observation = {
        {pixels, sizeof(pixels), 1, 1, 3, TRTMC_IMAGE_FLOAT32}, state, 2};
    trtmc_image_state_to_action_chunk_request_v1 request = {observation};
    trtmc_result *chunk = NULL, *rejected = NULL;
    check(chunks->run(model, &request, NULL, &chunk, &error) == TRTMC_OK,
          "C stateless chunk succeeds");
    trtmc_image_state_action_chunk_view_v1 chunk_view = {0};
    check(chunks->result_view(chunk, &chunk_view, &error) == TRTMC_OK &&
              chunk_view.actions.values.rows == 2 && chunk_view.actions.values.columns == 2 &&
              chunk_view.actions.values.data[0] == 2.5F && chunk_view.actions.values.data[3] == 8 &&
              chunk_view.within_training_bounds == 0 && chunk_view.inference_ms == 101,
          "C chunk preserves raw out-of-bound action values and ordered shape");
    check(chunks->run(disabled, &request, NULL, &rejected, &error) == TRTMC_UNSUPPORTED &&
              rejected == NULL,
          "chunk table cannot bypass per-model capability");
    clear_error(core, &error);
    trtmc_image_state_action_session* session = NULL;
    check(queue->create(disabled, NULL, &session, &error) == TRTMC_UNSUPPORTED && session == NULL,
          "queue factory table cannot bypass per-model capability");
    clear_error(core, &error);
    request.observation.state = NULL;
    check(chunks->run(model, &request, NULL, &rejected, &error) == TRTMC_INVALID_ARGUMENT &&
              rejected == NULL,
          "state pointer/count contract is checked");
    clear_error(core, &error);
    request.observation = observation;
    request.observation.image.byte_size = 1;
    check(chunks->run(model, &request, NULL, &rejected, &error) == TRTMC_INVALID_ARGUMENT,
          "action uses the existing image transport validation");
    clear_error(core, &error);
    request.observation = observation;

    char tag[] = "persisted";
    trtmc_config_entry_v1 entry = {0};
    entry.name = str("tag");
    entry.value.kind = TRTMC_CONFIG_STRING;
    entry.value.as.string = str(tag);
    trtmc_config_view_v1 config = {&entry, 1};
    check(queue->create(model, &config, &session, &error) == TRTMC_OK && session != NULL,
          "C queue creation succeeds");
    tag[0] = 'X';
    check(chunks->run(model, &request, NULL, &rejected, &error) == TRTMC_BUSY && rejected == NULL,
          "stateless calls cannot consume a live queue's execution");
    clear_error(core, &error);
    trtmc_image_state_action_session* second = NULL;
    check(queue->create(model, NULL, &second, &error) == TRTMC_BUSY && second == NULL,
          "model reservation prevents a second session");
    clear_error(core, &error);
    trtmc_result* first = NULL;
    check(queue->act(session, &observation, NULL, &first, &error) == TRTMC_OK,
          "C act returns the first queued step");
    trtmc_action_step_view_v1 step_view = {0};
    check(queue->result_view(first, &step_view, &error) == TRTMC_OK &&
              step_view.action.values.rows == 1 && step_view.action.values.count == 2 &&
              step_view.action.values.data[0] == 2.5F && step_view.action.values.data[1] == -3 &&
              step_view.started_new_chunk == 1 && step_view.within_training_bounds == 1 &&
              step_view.inference_ms == 1 &&
              equal(step_view.action.schema.domain, "fixture.persisted") &&
              equal(step_view.action.schema.normalization, "unnormalized") &&
              step_view.action.schema.units.size == 0,
          "step owns unnormalized values, copied creation config and truthful schema metadata");
    check(queue->result_view(chunk, &step_view, &error) == TRTMC_INVALID_ARGUMENT &&
              step_view.action.values.count == 0,
          "chunk cannot masquerade as a single queued step");
    clear_error(core, &error);
    check(chunks->result_view(first, &chunk_view, &error) == TRTMC_INVALID_ARGUMENT &&
              chunk_view.actions.values.count == 0,
          "queued step cannot masquerade as a stateless chunk");
    clear_error(core, &error);
    state[0] = NAN;
    check(queue->act(session, &observation, NULL, &rejected, &error) == TRTMC_INVALID_ARGUMENT,
          "family checks every action input even while queue contains values");
    clear_error(core, &error);
    state[0] = 20;
    state[1] = 40;
    trtmc_result* queued = NULL;
    check(queue->act(session, &observation, NULL, &queued, &error) == TRTMC_OK,
          "queued action survives prior argument rejection");
    check(queue->result_view(queued, &step_view, &error) == TRTMC_OK &&
              step_view.action.values.data[0] == 4 && step_view.action.values.data[1] == 8 &&
              step_view.started_new_chunk == 0 && step_view.within_training_bounds == 0 &&
              step_view.inference_ms == 0,
          "family queue consumes original chunk in order with zero new inference");

    core->model_release(model);
    model = NULL;
    check(queue->reset(session, &error) == TRTMC_OK,
          "session retains model and can reset after caller releases its model handle");
    trtmc_result* fresh = NULL;
    check(queue->act(session, &observation, NULL, &fresh, &error) == TRTMC_OK,
          "session remains executable after model release");
    check(queue->result_view(fresh, &step_view, &error) == TRTMC_OK &&
              step_view.action.values.data[0] == 20.5F && step_view.started_new_chunk == 1 &&
              step_view.inference_ms == 2,
          "reset delegates a new family chunk from current observation");
    queue->release(session);
    session = NULL;
    queue->release(NULL);
    check(queue->result_view(first, &step_view, &error) == TRTMC_OK &&
              step_view.action.values.data[0] == 2.5F &&
              equal(step_view.action.schema.domain, "fixture.persisted"),
          "owned first-step result survives reset and all handle releases");
    check(chunks->result_view(chunk, &chunk_view, &error) == TRTMC_OK &&
              chunk_view.actions.values.data[3] == 8,
          "owned stateless chunk also survives model release");
    check(queue->act(NULL, &observation, NULL, &rejected, &error) == TRTMC_INVALID_ARGUMENT &&
              rejected == NULL,
          "null session fails without touching an implementation");
    clear_error(core, &error);
    core->result_release(chunk);
    core->result_release(first);
    core->result_release(queued);
    core->result_release(fresh);
    core->model_release(disabled);
    clear_error(core, &error);
    return failures ? 1 : 0;
}

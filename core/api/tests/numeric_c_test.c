/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/numeric.h"
#include "trtmc/trtmc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const trtmc_core_api_v1* core;
static int failures;
static trtmc_string_view string(const char* value) {
    trtmc_string_view out = {value, (uint64_t)strlen(value)};
    return out;
}
static int same(trtmc_string_view value, const char* expected) {
    return value.size == strlen(expected) && memcmp(value.data, expected, (size_t)value.size) == 0;
}
static void check(int ok, const char* label) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}
static void checked(trtmc_status status, trtmc_status expected, trtmc_error** error,
                    const char* label) {
    check(status == expected, label);
    check(status == TRTMC_OK ? *error == NULL : *error != NULL,
          "error ownership agrees with status");
    core->error_release(*error);
    *error = NULL;
}
static void write_bundle(const char* path, const char* mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    char header[512];
    unsigned shift;
    FILE* file;
    int length = snprintf(header, sizeof(header),
                          "{\"format\":1,\"family\":\"numeric_fixture\",\"task\":\"%s\","
                          "\"backend\":\"fake\",\"sections\":{}}",
                          mode);
    if (length < 0 || (size_t)length >= sizeof(header) || (file = fopen(path, "wb")) == NULL)
        exit(2);
    fwrite(magic, 1, 8, file);
    for (shift = 0; shift < 64; shift += 8)
        fputc((int)(((uint64_t)length >> shift) & 255), file);
    fwrite(header, 1, (size_t)length, file);
    if (fclose(file))
        exit(2);
}
static const trtmc_api_header* get(trtmc_model* model, const char* task) {
    const trtmc_api_header* table = NULL;
    trtmc_error* error = NULL;
    checked(core->model_get_task_api(model, string(task), 1, 0, &table, &error), TRTMC_OK, &error,
            "numeric C table lookup");
    return table;
}
static void tests(trtmc_model* model, trtmc_model* restricted, trtmc_result** retained,
                  trtmc_result** retained_joint) {
    const trtmc_series_to_point_forecast_api_v1* point =
        (const trtmc_series_to_point_forecast_api_v1*)get(model,
                                                          TRTMC_TASK_SERIES_TO_POINT_FORECAST);
    const trtmc_series_to_quantile_forecast_api_v1* quantile =
        (const trtmc_series_to_quantile_forecast_api_v1*)get(
            model, TRTMC_TASK_SERIES_TO_QUANTILE_FORECAST);
    const trtmc_series_to_regression_distribution_api_v1* regression =
        (const trtmc_series_to_regression_distribution_api_v1*)get(
            model, TRTMC_TASK_SERIES_TO_REGRESSION_DISTRIBUTION);
    const trtmc_latent_conditioned_text_generation_api_v1* condition =
        (const trtmc_latent_conditioned_text_generation_api_v1*)get(
            model, TRTMC_TASK_LATENT_CONDITIONED_TEXT_GENERATION);
    const trtmc_latent_replay_to_text_api_v1* replay =
        (const trtmc_latent_replay_to_text_api_v1*)get(model, TRTMC_TASK_LATENT_REPLAY_TO_TEXT);
    const trtmc_latent_denoising_step_api_v1* denoise =
        (const trtmc_latent_denoising_step_api_v1*)get(model, TRTMC_TASK_LATENT_DENOISING_STEP);
    const trtmc_latent_to_token_logits_api_v1* logits =
        (const trtmc_latent_to_token_logits_api_v1*)get(model, TRTMC_TASK_LATENT_TO_TOKEN_LOGITS);
    const float history[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const uint8_t observed[] = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    trtmc_series_request_v1 request = {{history, 12, 4, 3}, observed, 12};
    trtmc_error* error = NULL;
    trtmc_result* output = NULL;
    trtmc_point_forecast_view_v1 p = {0};
    trtmc_quantile_forecast_view_v1 q = {0};
    trtmc_regression_distribution_view_v1 r = {0};
    trtmc_text_result_view_v1 text = {0};
    trtmc_denoised_latents_view_v1 d = {0};
    trtmc_latent_token_logits_view_v1 l = {0};
    const float latent[] = {1, 2, 3, 4}, initial[] = {9, 8, 7, 6}, mask[] = {1, 0.5F},
                noise[] = {10, 11, 12, 13};
    trtmc_latent_conditioned_text_request_v1 condition_request = {
        {latent, 4}, {mask, 2}, {initial, 4}, {noise, 4}, {"preserved", 9}};
    trtmc_latent_replay_text_request_v1 replay_request = {{initial, 4}, {"source", 6}, {noise, 4}};
    trtmc_latent_step_request_v1 step = {{latent, 4, 2, 2}, {initial, 4, 2, 2}, 0.25};
    if (!point || !quantile || !regression || !condition || !replay || !denoise || !logits)
        return;
    const trtmc_series_to_point_and_quantile_forecast_api_v1* joint =
        (const trtmc_series_to_point_and_quantile_forecast_api_v1*)get(
            model, TRTMC_TASK_SERIES_TO_POINT_AND_QUANTILE_FORECAST);
    if (!joint)
        return;
    trtmc_point_and_quantile_forecast_view_v1 both = {0};
    checked(joint->run(model, &request, NULL, retained_joint, &error), TRTMC_OK, &error,
            "C joint forecast uses one family evaluation");
    checked(joint->result_view(*retained_joint, &both, &error), TRTMC_OK, &error,
            "C joint forecast has two typed views");
    check(both.point.values.data[0] == 990 && both.quantiles.quantile_levels.size == 3 &&
              both.quantiles.quantile_levels.data[1] == 0.5 && both.quantiles.values[6] == 995 &&
              both.point.values.columns == 3 && both.quantiles.channels == 3,
          "C point is not synthesized from the median and channel axis is retained");
    checked(point->result_view(*retained_joint, &p, &error), TRTMC_INVALID_ARGUMENT, &error,
            "joint result cannot be reinterpreted as a standalone point result");
    checked(joint->run(restricted, &request, NULL, &output, &error), TRTMC_UNSUPPORTED, &error,
            "joint table cannot bypass another model's Task declaration");
    checked(point->run(model, &request, NULL, &output, &error), TRTMC_OK, &error,
            "C point forecast executes");
    checked(point->result_view(output, &p, &error), TRTMC_OK, &error, "C point result readable");
    check(p.values.rows == 2 && p.values.columns == 3 && p.values.data[0] == -10 &&
              p.axes.horizon_steps.data[1] == 3,
          "C point axes and mask are preserved");
    checked(regression->result_view(output, &r, &error), TRTMC_INVALID_ARGUMENT, &error,
            "point result cannot be reinterpreted as regression");
    core->result_release(output);
    output = NULL;
    checked(quantile->run(model, &request, NULL, &output, &error), TRTMC_OK, &error,
            "C quantile executes");
    checked(quantile->result_view(output, &q, &error), TRTMC_OK, &error,
            "C quantile result readable");
    check(q.value_count == 12 && q.horizon == 2 && q.channels == 3 &&
              q.quantile_levels.data[0] == 0.1 && q.values[6] == 7,
          "C quantile output exposes Q,H,C semantics");
    core->result_release(output);
    output = NULL;
    checked(regression->run(model, &request, NULL, retained, &error), TRTMC_OK, &error,
            "C regression executes");
    checked(regression->result_view(*retained, &r, &error), TRTMC_OK, &error,
            "C regression readable");
    check(r.target_count == 2 && r.parameter_count == 2 && same(r.parameters[0].name, "scale") &&
              r.parameters[0].values[0] == 0.5F && r.parameters[0].target_count == 2 &&
              r.target_names.size == 0,
          "C regression parameters are named target arrays with unknown target metadata");
    checked(point->run(restricted, &request, NULL, &output, &error), TRTMC_UNSUPPORTED, &error,
            "model A table cannot bypass model B task support");
    request.observed_count = 1;
    checked(point->run(model, &request, NULL, &output, &error), TRTMC_INVALID_ARGUMENT, &error,
            "observed mask count must match history");
    request.observed_count = 12;
    request.past_values.rows = UINT64_MAX;
    checked(point->run(model, &request, NULL, &output, &error), TRTMC_INVALID_ARGUMENT, &error,
            "overflowed matrix shape fails before dereference");
    checked(condition->run(model, &condition_request, NULL, &output, &error), TRTMC_OK, &error,
            "typed latent conditioning executes");
    checked(condition->result_view(output, &text, &error), TRTMC_OK, &error,
            "latent text result readable");
    check(text.token_ids.size == 6 && text.token_ids.data[1] == 50 && text.token_ids.data[2] == 9 &&
              text.token_ids.data[3] == 13 && text.token_ids.data[5] == 9,
          "C condition mask, initial latent and SDE replay are preserved");
    core->result_release(output);
    output = NULL;
    checked(replay->run(model, &replay_request, NULL, &output, &error), TRTMC_OK, &error,
            "initial-only replay has separate table");
    checked(replay->result_view(output, &text, &error), TRTMC_OK, &error, "replay text readable");
    check(same(text.text, "source|replayed"), "replay prompt is not discarded");
    core->result_release(output);
    output = NULL;
    checked(denoise->run(model, &step, NULL, &output, &error), TRTMC_OK, &error,
            "denoising component executes");
    checked(denoise->result_view(output, &d, &error), TRTMC_OK, &error, "denoised matrix readable");
    check(d.latents.rows == 2 && d.latents.columns == 2 && d.latents.data[0] == 10.25F,
          "timestep and self-condition reach denoiser");
    checked(logits->result_view(output, &l, &error), TRTMC_INVALID_ARGUMENT, &error,
            "denoised result is not vocabulary logits");
    core->result_release(output);
    output = NULL;
    checked(logits->run(model, &step, NULL, &output, &error), TRTMC_OK, &error,
            "latent decoder executes");
    checked(logits->result_view(output, &l, &error), TRTMC_OK, &error,
            "vocabulary logits readable");
    check(l.logits.rows == 2 && l.logits.columns == 3 &&
              same(l.vocabulary_id, "fixture-vocabulary"),
          "position/vocabulary axes and identity retained");
    core->result_release(output);
    output = NULL;
    condition_request.condition_mask.size = 0;
    checked(condition->run(model, &condition_request, NULL, &output, &error),
            TRTMC_INVALID_ARGUMENT, &error, "condition requires one mask per position");
    replay_request.initial_latents.size = 0;
    checked(replay->run(model, &replay_request, NULL, &output, &error), TRTMC_OK, &error,
            "C SDE-only replay uses family initial-state default");
    checked(replay->result_view(output, &text, &error), TRTMC_OK, &error,
            "C noise-only result readable");
    check(text.token_ids.data[0] == -1 && text.token_ids.data[1] == 13 &&
              same(text.text, "source|replayed"),
          "C noise-only preserves noise and prompt without a fabricated shape");
    core->result_release(output);
    output = NULL;
    replay_request.sde_noises.size = 0;
    checked(replay->run(model, &replay_request, NULL, &output, &error), TRTMC_INVALID_ARGUMENT,
            &error, "replay requires initial latents or noise");
    replay_request.initial_latents = (trtmc_f32_view){NULL, 4};
    checked(replay->run(model, &replay_request, NULL, &output, &error), TRTMC_INVALID_ARGUMENT,
            &error, "NULL positive-length initial buffer fails before family execution");
    replay_request.initial_latents = (trtmc_f32_view){initial, UINT64_MAX};
    checked(replay->run(model, &replay_request, NULL, &output, &error), TRTMC_INVALID_ARGUMENT,
            &error, "overflowed initial buffer length fails before dereference");
    replay_request.initial_latents = (trtmc_f32_view){NULL, 0};
    replay_request.sde_noises = (trtmc_f32_view){NULL, 4};
    checked(replay->run(model, &replay_request, NULL, &output, &error), TRTMC_INVALID_ARGUMENT,
            &error, "NULL positive-length SDE noise fails before family execution");
    replay_request.sde_noises = (trtmc_f32_view){noise, UINT64_MAX};
    checked(replay->run(model, &replay_request, NULL, &output, &error), TRTMC_INVALID_ARGUMENT,
            &error, "overflowed SDE noise length fails before dereference");
    condition_request.condition_mask = (trtmc_f32_view){NULL, 2};
    checked(condition->run(model, &condition_request, NULL, &output, &error),
            TRTMC_INVALID_ARGUMENT, &error,
            "NULL positive-length condition mask fails before family execution");
    condition_request.condition_mask = (trtmc_f32_view){mask, 2};
    condition_request.condition_latents = (trtmc_f32_view){NULL, 0};
    checked(condition->run(model, &condition_request, NULL, &output, &error),
            TRTMC_INVALID_ARGUMENT, &error, "mask cannot be supplied without a raw condition");
    checked(denoise->run(model, NULL, NULL, &output, &error), TRTMC_INVALID_ARGUMENT, &error,
            "null numeric request fails");
    const float packed[] = {1, 2, 9, 8, 3, 4, 7, 6};
    trtmc_latent_step_request_v1 native_step = {{packed, 8, 0, 0}, {0}, 0.25};
    trtmc_config_entry_v1 guidance = {0};
    guidance.name = string("self_cond_cfg_scale");
    guidance.value.kind = TRTMC_CONFIG_F64;
    guidance.value.as.f64 = 3;
    const trtmc_config_view_v1 native_config = {&guidance, 1};
    checked(denoise->run(model, &native_step, &native_config, &output, &error), TRTMC_OK, &error,
            "C native packed denoising input executes");
    checked(denoise->result_view(output, &d, &error), TRTMC_OK, &error,
            "C native denoised output readable");
    check(d.latents.rows == 2 && d.latents.columns == 2 && d.latents.data[0] == 12.25F &&
              d.latents.data[2] == 3,
          "C packed input preserves its embedded self-conditioning without double packing");
    core->result_release(output);
    output = NULL;
    checked(logits->run(model, &native_step, &native_config, &output, &error), TRTMC_OK, &error,
            "C native packed decoder executes");
    checked(logits->result_view(output, &l, &error), TRTMC_OK, &error,
            "C native decoder logits readable");
    check(l.logits.rows == 2 && l.logits.columns == 3 && l.logits.data[0] == 12.25F,
          "C decoder keeps its own output width and guidance");
    core->result_release(output);
    output = NULL;
    native_step.self_condition = step.self_condition;
    checked(denoise->run(model, &native_step, NULL, &output, &error), TRTMC_INVALID_ARGUMENT,
            &error, "C native packed input rejects separate self-conditioning");
    native_step.self_condition = (trtmc_f32_matrix_view_v1){0};
    native_step.latents.count = 7;
    checked(logits->run(model, &native_step, NULL, &output, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C family validates packed input length");
}

static trtmc_result* batch_forecasts(trtmc_model* model, trtmc_model* restricted) {
    const trtmc_batch_series_to_point_forecast_api_v1* point =
        (const trtmc_batch_series_to_point_forecast_api_v1*)get(
            model, TRTMC_TASK_BATCH_SERIES_TO_POINT_FORECAST);
    const trtmc_batch_series_to_quantile_forecast_api_v1* quantile =
        (const trtmc_batch_series_to_quantile_forecast_api_v1*)get(
            model, TRTMC_TASK_BATCH_SERIES_TO_QUANTILE_FORECAST);
    const trtmc_batch_series_to_point_and_quantile_forecast_api_v1* joint =
        (const trtmc_batch_series_to_point_and_quantile_forecast_api_v1*)get(
            model, TRTMC_TASK_BATCH_SERIES_TO_POINT_AND_QUANTILE_FORECAST);
    if (!point || !quantile || !joint)
        return NULL;
    const float first[] = {3, 4}, second[] = {8, 9, 10, 11};
    const uint8_t observed[] = {0, 1, 1, 1};
    trtmc_config_entry_v1 frequency = {0};
    frequency.name = string("frequency");
    frequency.value.kind = TRTMC_CONFIG_I64;
    frequency.value.as.i64 = 2;
    const trtmc_batch_series_item_v1 items[] = {
        {{{first, 2, 2, 1}, NULL, 0}, {NULL, 0}},
        {{{second, 4, 4, 1}, observed, 4}, {&frequency, 1}}};
    const trtmc_batch_series_request_v1 request = {items, 2};
    trtmc_error* error = NULL;
    trtmc_result *points = NULL, *quantiles = NULL, *joined = NULL, *rejected = NULL;
    uint64_t count = 0;
    trtmc_point_forecast_view_v1 p = {0};
    trtmc_quantile_forecast_view_v1 q = {0};
    trtmc_point_and_quantile_forecast_view_v1 both = {0};
    checked(point->run(model, &request, &points, &error), TRTMC_OK, &error,
            "C native point batch executes");
    checked(point->result_count(points, &count, &error), TRTMC_OK, &error,
            "C point batch count readable");
    check(count == 2, "C batch count is independent of history/channel counts");
    checked(point->result_item_view(points, 1, &p, &error), TRTMC_OK, &error,
            "C later point forecast readable");
    check(p.values.columns == 1 && p.values.rows == 2 && p.values.data[0] == 2190 &&
              p.values.data[1] == 2204,
          "C ragged history length, mask and per-item frequency reach the family");
    checked(quantile->result_item_view(points, 0, &q, &error), TRTMC_INVALID_ARGUMENT, &error,
            "point batch cannot masquerade as quantile batch");
    checked(quantile->run(model, &request, &quantiles, &error), TRTMC_OK, &error,
            "C native quantile batch executes");
    checked(quantile->result_item_view(quantiles, 1, &q, &error), TRTMC_OK, &error,
            "C quantile batch item readable");
    check(q.channels == 1 && q.horizon == 2 && q.value_count == 6 && q.quantile_levels.size == 3 &&
              q.quantile_levels.data[0] == 0.1 && q.quantile_levels.data[2] == 0.9 &&
              q.values[2] == 3195,
          "C quantile levels and Q,H,C remain explicit");
    checked(joint->run(model, &request, &joined, &error), TRTMC_OK, &error,
            "C native joint batch executes");
    checked(joint->result_item_view(joined, 1, &both, &error), TRTMC_OK, &error,
            "C joint batch item readable");
    check(both.point.values.data[0] == 4190 && both.quantiles.values[2] == 4195 &&
              both.point.axes.horizon_steps.data[1] == both.quantiles.axes.horizon_steps.data[1],
          "C joint batch preserves independent point/median and matching axes");
    checked(joint->result_item_view(joined, 2, &both, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C batch result item index is bounded");
    check(both.point.values.count == 0 && both.quantiles.value_count == 0,
          "failed item read clears output views");
    const float multi[] = {1, 2, 3, 4};
    const trtmc_batch_series_item_v1 one_item = {{{multi, 4, 2, 2}, NULL, 0}, {NULL, 0}};
    const trtmc_batch_series_request_v1 multi_request = {&one_item, 1};
    trtmc_result* multi_result = NULL;
    checked(point->run(model, &multi_request, &multi_result, &error), TRTMC_OK, &error,
            "C one multichannel series executes as one batch item");
    checked(point->result_count(multi_result, &count, &error), TRTMC_OK, &error,
            "C multichannel batch count readable");
    checked(point->result_item_view(multi_result, 0, &p, &error), TRTMC_OK, &error,
            "C multichannel point forecast readable");
    check(count == 1 && p.values.columns == 2, "input_channel is never used as a batch axis");
    core->result_release(multi_result);
    checked(point->run(restricted, &request, &rejected, &error), TRTMC_UNSUPPORTED, &error,
            "batch table cannot bypass another model's declaration");
    trtmc_batch_series_item_v1 invalid_items[] = {items[0], items[1]};
    trtmc_batch_series_request_v1 invalid = {invalid_items, 2};
    trtmc_config_entry_v1 wrong = frequency;
    wrong.value.kind = TRTMC_CONFIG_F64;
    wrong.value.as.f64 = 2;
    invalid_items[1].config = (trtmc_config_view_v1){&wrong, 1};
    trtmc_status status = point->run(model, &invalid, &rejected, &error);
    check(status == TRTMC_INVALID_CONFIG && rejected == NULL && error != NULL &&
              core->error_message(error).size >= 13 &&
              memcmp(core->error_message(error).data, "batch item[1]", 13) == 0,
          "C family frequency error identifies the failing item without a partial handle");
    checked(status, TRTMC_INVALID_CONFIG, &error, "C item config error is translated");
    checked(point->run(model, &request, &rejected, &error), TRTMC_OK, &error,
            "C batch runs after config rejection");
    checked(point->result_item_view(rejected, 0, &p, &error), TRTMC_OK, &error,
            "C post-error result readable");
    check(p.values.data[0] == 6003,
          "failed per-item config preflight did not execute a native family batch");
    core->result_release(rejected);
    rejected = NULL;
    invalid_items[1] = items[1];
    invalid_items[1].input.observed_count = 1;
    status = point->run(model, &invalid, &rejected, &error);
    check(status == TRTMC_INVALID_ARGUMENT && rejected == NULL && error != NULL &&
              core->error_message(error).size >= 13 &&
              memcmp(core->error_message(error).data, "batch item[1]", 13) == 0,
          "C malformed second mask retains an indexed transport error");
    checked(status, TRTMC_INVALID_ARGUMENT, &error, "C mask error is translated");
    const trtmc_batch_series_request_v1 empty = {NULL, 0};
    checked(point->run(model, &empty, &rejected, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C native forecast batch must be nonempty");
    core->result_release(points);
    core->result_release(quantiles);
    return joined;
}

static void flat_history_tests(const char* root) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/numeric-c-flat.bundle", root) >= (int)sizeof(path))
        exit(2);
    write_bundle(path, "flat_channels_two");
    trtmc_load_options_v1 options = {0};
    options.struct_size = sizeof(options);
    options.runtime_root = string(root);
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    checked(core->model_load(string(path), &options, &model, &error), TRTMC_OK, &error,
            "C loads provider with bundle-owned channel count");
    if (!model)
        return;
    const trtmc_series_to_point_forecast_api_v1* point =
        (const trtmc_series_to_point_forecast_api_v1*)get(model,
                                                          TRTMC_TASK_SERIES_TO_POINT_FORECAST);
    if (!point) {
        core->model_release(model);
        return;
    }
    const float values[] = {1, 2, 3, 4, 5, 6};
    const uint8_t observed[] = {0, 1, 1, 1, 1, 1};
    trtmc_series_request_v1 request = {{values, 6, 0, 0}, observed, 6};
    trtmc_result* result = NULL;
    trtmc_point_forecast_view_v1 view = {0};
    checked(point->run(model, &request, NULL, &result, &error), TRTMC_OK, &error,
            "C nonempty history can omit both shape dimensions");
    checked(point->result_view(result, &view, &error), TRTMC_OK, &error,
            "C family-resolved output is fully shaped");
    check(view.values.columns == 2 && view.values.rows == 2 && view.values.data[0] == -10,
          "C shared layer preserved zero dimensions and flat mask for family resolution");
    core->result_release(result);
    result = NULL;
    request.past_values.rows = 2;
    request.past_values.columns = 3;
    checked(point->run(model, &request, NULL, &result, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C explicit incompatible channels are not silently reinterpreted");
    request.past_values.rows = 0;
    request.past_values.columns = 2;
    checked(point->run(model, &request, NULL, &result, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C rejects partially specified series dimensions");
    request.past_values.columns = 0;
    request.past_values.count = 5;
    request.observed_count = 0;
    checked(point->run(model, &request, NULL, &result, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C family rejects indivisible flat element count");
    request.past_values.count = 6;
    request.observed_count = 2;
    checked(point->run(model, &request, NULL, &result, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C flat observation mask must match element count");
    request.past_values.count = 0;
    request.observed_count = 0;
    checked(point->run(model, &request, NULL, &result, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C does not treat empty history as a shape default");
    core->model_release(model);
}

static void regression_values_tests(const char* root) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/targets-c.bundle", root);
    write_bundle(path, "regression_values_named");
    trtmc_load_options_v1 options = {0};
    options.struct_size = sizeof(options);
    options.runtime_root = string(root);
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    checked(core->model_load(string(path), &options, &model, &error), TRTMC_OK, &error,
            "C loads deterministic target regression");
    if (!model)
        return;
    const trtmc_series_to_regression_values_api_v1* task =
        (const trtmc_series_to_regression_values_api_v1*)get(
            model, TRTMC_TASK_SERIES_TO_REGRESSION_VALUES);
    if (!task) {
        core->model_release(model);
        return;
    }
    const float values[] = {1, 2, 3, 4};
    const uint8_t observed[] = {1, 0, 1, 1};
    trtmc_series_request_v1 input = {{values, 4, 2, 2}, observed, 4};
    trtmc_result* result = NULL;
    trtmc_config_entry_v1 scale = {0};
    scale.name = string("scale");
    scale.value.kind = TRTMC_CONFIG_F64;
    scale.value.as.f64 = 2;
    const trtmc_config_view_v1 config = {&scale, 1};
    checked(task->run(model, &input, &config, &result, &error), TRTMC_OK, &error,
            "C target regression receives typed history and config");
    const trtmc_api_header* unsupported = NULL;
    checked(core->model_get_task_api(model, string(TRTMC_TASK_SERIES_TO_POINT_FORECAST), 1, 0,
                                     &unsupported, &error),
            TRTMC_UNSUPPORTED, &error, "target regression does not advertise a forecast");
    input.observed_count = 1;
    trtmc_result* rejected = NULL;
    checked(task->run(model, &input, NULL, &rejected, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C rejects a malformed target-regression mask");
    check(rejected == NULL, "invalid target input yields no owned result");
    core->model_release(model);
    trtmc_regression_values_view_v1 view = {0};
    checked(task->result_view(result, &view, &error), TRTMC_OK, &error,
            "C target-regression view survives actual model release");
    check(view.values.size == 2 && view.values.data[0] == 16 && view.values.data[1] == 1 &&
              view.target_names.size == 2 && same(view.target_names.data[0], "total") &&
              view.target_units.size == 2 && same(view.target_units.data[1], "count"),
          "C complete target arrays and metadata retain order and ownership");
    core->result_release(result);
}

int main(int argc, char** argv) {
    char full[4096], restricted_path[4096];
    trtmc_load_options_v1 options = {0};
    trtmc_error* error = NULL;
    trtmc_model* model = NULL;
    trtmc_model* restricted = NULL;
    trtmc_result* retained = NULL;
    trtmc_result* retained_joint = NULL;
    trtmc_result* retained_batch = NULL;
    const trtmc_batch_series_to_point_and_quantile_forecast_api_v1* batch_joint = NULL;
    const trtmc_series_to_point_and_quantile_forecast_api_v1* joint = NULL;
    const trtmc_series_to_regression_distribution_api_v1* regression = NULL;
    trtmc_regression_distribution_view_v1 view = {0};
    if (argc != 2 || strlen(argv[1]) + 40 >= sizeof(full) || trtmc_get_api(1, 0, &core) != TRTMC_OK)
        return 2;
    snprintf(full, sizeof(full), "%s/numeric-c-all.bundle", argv[1]);
    snprintf(restricted_path, sizeof(restricted_path), "%s/numeric-c-regression.bundle", argv[1]);
    write_bundle(full, "all_numeric");
    write_bundle(restricted_path, "regression_only");
    options.struct_size = sizeof(options);
    options.runtime_root = string(argv[1]);
    checked(core->model_load(string(full), &options, &model, &error), TRTMC_OK, &error,
            "load numeric model");
    checked(core->model_load(string(restricted_path), &options, &restricted, &error), TRTMC_OK,
            &error, "load restricted numeric model");
    if (model && restricted) {
        tests(model, restricted, &retained, &retained_joint);
        retained_batch = batch_forecasts(model, restricted);
        batch_joint = (const trtmc_batch_series_to_point_and_quantile_forecast_api_v1*)get(
            model, TRTMC_TASK_BATCH_SERIES_TO_POINT_AND_QUANTILE_FORECAST);
        joint = (const trtmc_series_to_point_and_quantile_forecast_api_v1*)get(
            model, TRTMC_TASK_SERIES_TO_POINT_AND_QUANTILE_FORECAST);
        regression = (const trtmc_series_to_regression_distribution_api_v1*)get(
            model, TRTMC_TASK_SERIES_TO_REGRESSION_DISTRIBUTION);
    }
    core->model_release(model);
    core->model_release(restricted);
    if (retained && regression) {
        checked(regression->result_view(retained, &view, &error), TRTMC_OK, &error,
                "regression snapshot survives model release");
        check(view.parameters[1].values[1] == 20,
              "owned named parameter values survive model release");
    }
    core->result_release(retained);
    if (retained_joint && joint) {
        trtmc_point_and_quantile_forecast_view_v1 both = {0};
        checked(joint->result_view(retained_joint, &both, &error), TRTMC_OK, &error,
                "joint snapshot survives model release");
        check(both.point.values.data[0] == 990 && both.quantiles.values[6] == 995,
              "both components own their values after all model handles are released");
    }
    core->result_release(retained_joint);
    if (retained_batch && batch_joint) {
        trtmc_point_and_quantile_forecast_view_v1 both = {0};
        checked(batch_joint->result_item_view(retained_batch, 1, &both, &error), TRTMC_OK, &error,
                "C owned batch remains readable after model release");
        check(both.point.values.data[0] == 4190 && both.quantiles.values[2] == 4195,
              "C complete joint batch owns both components after all input/model lifetimes end");
    }
    core->result_release(retained_batch);
    flat_history_tests(argv[1]);
    regression_values_tests(argv[1]);
    fprintf(stderr, "%s\n", failures ? "SOME FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}

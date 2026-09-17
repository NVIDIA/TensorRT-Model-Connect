/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/trtmc.h"
#include "trtmc/video.h"

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
    trtmc_string_view result = {value, strlen(value)};
    return result;
}
static int bundle(const char* path, const char* mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    char header[512];
    const int count =
        snprintf(header, sizeof(header),
                 "{\"format\":1,\"family\":\"video_fixture\",\"task\":\"%s\",\"backend\":\"fake\","
                 "\"sections\":{\"engine.plan\":{\"offset\":0,\"length\":4}}}",
                 mode);
    if (count < 0 || (size_t)count >= sizeof(header))
        return 0;
    FILE* out = fopen(path, "wb");
    if (!out)
        return 0;
    int ok = fwrite(magic, 1, sizeof(magic), out) == sizeof(magic);
    for (unsigned shift = 0; shift < 64; shift += 8)
        ok = (fputc((int)(((uint64_t)count >> shift) & 255U), out) != EOF) && ok;
    ok = (fwrite(header, 1, (size_t)count, out) == (size_t)count) && ok;
    ok = (fwrite("PLAN", 1, 4, out) == 4) && ok;
    return fclose(out) == 0 && ok;
}
static void release_error(const trtmc_core_api_v1* core, trtmc_error** error) {
    core->error_release(*error);
    *error = NULL;
}
static void other_batches(const trtmc_core_api_v1* core, trtmc_model* model) {
    const float pixels[] = {0.25F, 0, 0}, values[] = {0.25F, 0.5F, 0.75F, 1};
    const trtmc_image_input_v1 image = {pixels, sizeof(pixels), 1, 1, 3, TRTMC_IMAGE_FLOAT32};
    const trtmc_image_input_v1 frames[] = {image, image};
    const double times[] = {0, 0.1};
    const trtmc_video_view_v1 clip = {frames, 2, {times, 2}};
    trtmc_action_schema_view_v1 schema = {0};
    schema.domain = str("fixture.motion");
    const trtmc_action_output_spec_v1 spec = {schema, 2};
    const trtmc_action_frame_span_v1 spans[] = {{1, 2}, {2, 3}};
    const trtmc_action_sequence_view_v1 actions = {{values, 4, 2, 2}, schema, {times, 2}, spans, 2};
    trtmc_batch_initial_image_text_to_video_item_v1 initial[2] = {0};
    trtmc_batch_video_text_to_future_video_item_v1 future[2] = {0};
    trtmc_batch_image_action_to_future_video_item_v1 forward[2] = {0};
    trtmc_batch_video_to_action_sequence_item_v1 inverse[2] = {0};
    trtmc_batch_image_to_action_and_video_item_v1 joint[2] = {0};
    trtmc_batch_text_to_audio_video_item_v1 av[2] = {0};
    trtmc_batch_initial_image_text_to_audio_video_item_v1 iav[2] = {0};
    for (int i = 0; i < 2; ++i) {
        const trtmc_string_view prompt = str(i ? "bb" : "a");
        initial[i].input.initial_image = image;
        initial[i].input.prompt = prompt;
        future[i].input.history = clip;
        future[i].input.prompt = prompt;
        forward[i].input.observation = image;
        forward[i].input.actions = actions;
        forward[i].input.has_prompt = 1;
        forward[i].input.prompt = prompt;
        inverse[i].input.observations = clip;
        inverse[i].input.action_spec = spec;
        joint[i].input.observation = image;
        joint[i].input.action_spec = spec;
        av[i].input.prompt = prompt;
        iav[i].input.initial_image = image;
        iav[i].input.prompt = prompt;
    }
#define RUN_BATCH(Id, Name, Items, View, Condition)                                                \
    do {                                                                                           \
        trtmc_error* error = NULL;                                                                 \
        const trtmc_api_header* header = NULL;                                                     \
        trtmc_result* result = NULL;                                                               \
        uint64_t count = 0;                                                                        \
        View view = {0};                                                                           \
        const trtmc_batch_##Name##_request_v1 request = {Items, 2};                                \
        if (core->model_get_task_api(model, str(Id), 1, 0, &header, &error) != TRTMC_OK) {         \
            check(0, "C new batch table discovery");                                               \
            release_error(core, &error);                                                           \
            break;                                                                                 \
        }                                                                                          \
        const trtmc_batch_##Name##_api_v1* task = (const trtmc_batch_##Name##_api_v1*)header;      \
        check(task->run(model, &request, &result, &error) == TRTMC_OK, "C typed batch dispatch");  \
        check(task->result_count(result, &count, &error) == TRTMC_OK && count == 2 &&              \
                  task->result_item_view(result, 1, &view, &error) == TRTMC_OK && (Condition),     \
              "C typed batch output and per-item input fidelity");                                 \
        check(task->result_item_view(result, 2, &view, &error) == TRTMC_INVALID_ARGUMENT,          \
              "C batch bounds checked");                                                           \
        release_error(core, &error);                                                               \
        core->result_release(result);                                                              \
    } while (0)
    RUN_BATCH(TRTMC_TASK_BATCH_INITIAL_IMAGE_TEXT_TO_VIDEO, initial_image_text_to_video, initial,
              trtmc_video_result_view_v1, fabs(view.frames[0].pixels[1] - 0.27) < 1e-6);
    RUN_BATCH(TRTMC_TASK_BATCH_VIDEO_TEXT_TO_FUTURE_VIDEO, video_text_to_future_video, future,
              trtmc_video_result_view_v1,
              view.conditioned_prefix_frames == 1 &&
                  fabs(view.timestamps_seconds.data[0] - 0.1) < 1e-9);
    RUN_BATCH(TRTMC_TASK_BATCH_IMAGE_ACTION_TO_FUTURE_VIDEO, image_action_to_future_video, forward,
              trtmc_video_result_view_v1,
              view.conditioned_prefix_frames == 1 && fabs(view.frames[1].pixels[0] - 0.24) < 1e-6);
    RUN_BATCH(TRTMC_TASK_BATCH_VIDEO_TO_ACTION_SEQUENCE, video_to_action_sequence, inverse,
              trtmc_action_sequence_view_v1,
              view.values.rows == 2 && fabs(view.values.data[0] - 0.25) < 1e-6 &&
                  view.frame_spans[1].end == 2);
    RUN_BATCH(TRTMC_TASK_BATCH_IMAGE_TO_ACTION_AND_VIDEO, image_to_action_and_video, joint,
              trtmc_action_video_result_view_v1,
              view.actions.frame_spans[1].end == 3 &&
                  fabs(view.video.frames[1].pixels[0] - view.actions.values.data[0]) < 1e-6);
    RUN_BATCH(TRTMC_TASK_BATCH_TEXT_TO_AUDIO_VIDEO, text_to_audio_video, av,
              trtmc_audio_video_result_view_v1,
              view.audio.audio.sample_count == 8 && fabs(view.audio_start_seconds + 0.05) < 1e-9);
    RUN_BATCH(TRTMC_TASK_BATCH_INITIAL_IMAGE_TEXT_TO_AUDIO_VIDEO, initial_image_text_to_audio_video,
              iav, trtmc_audio_video_result_view_v1,
              view.audio.audio.sample_count == 4 &&
                  fabs(view.video.frames[0].pixels[1] - 0.27) < 1e-6);
#undef RUN_BATCH
}

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    char path[4096], disabled_path[4096];
    if (snprintf(path, sizeof(path), "%s/video_c_all.bundle", argv[1]) >= (int)sizeof(path) ||
        snprintf(disabled_path, sizeof(disabled_path), "%s/video_c_none.bundle", argv[1]) >=
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
    {
        char batch_path[4096];
        trtmc_model* batch_model = NULL;
        const trtmc_api_header* batch_header = NULL;
        trtmc_result* batch_result = NULL;
        uint64_t count = 0;
        trtmc_video_result_view_v1 item = {0};
        trtmc_batch_text_to_video_item_v1 items[2] = {0};
        items[0].input.prompt = str("a");
        items[1].input.prompt = str("bb");
        const trtmc_batch_text_to_video_request_v1 request = {items, 2};
        if (snprintf(batch_path, sizeof(batch_path), "%s/video_c_batch.bundle", argv[1]) >=
                (int)sizeof(batch_path) ||
            !bundle(batch_path, "batch"))
            return 2;
        if (core->model_load(str(batch_path), &options, &batch_model, &error) != TRTMC_OK ||
            core->model_get_task_api(batch_model, str(TRTMC_TASK_BATCH_TEXT_TO_VIDEO), 1, 0,
                                     &batch_header, &error) != TRTMC_OK)
            return 2;
        const trtmc_batch_text_to_video_api_v1* task =
            (const trtmc_batch_text_to_video_api_v1*)batch_header;
        check(task->run(batch_model, &request, &batch_result, &error) == TRTMC_OK,
              "C executes a real B2 native video batch");
        other_batches(core, batch_model);
        trtmc_result* rejected_batch = NULL;
        trtmc_batch_text_to_video_request_v1 malformed = {NULL, 2};
        check(task->run(batch_model, &malformed, &rejected_batch, &error) ==
                      TRTMC_INVALID_ARGUMENT &&
                  rejected_batch == NULL,
              "C NULL batch items fail before family access");
        release_error(core, &error);
        malformed.items = items;
        malformed.count = UINT64_MAX;
        check(task->run(batch_model, &malformed, &rejected_batch, &error) ==
                      TRTMC_INVALID_ARGUMENT &&
                  rejected_batch == NULL,
              "C overflowed batch count fails before dereference");
        release_error(core, &error);
        check(task->result_count(batch_result, NULL, &error) == TRTMC_INVALID_ARGUMENT,
              "C null batch count output is rejected");
        release_error(core, &error);
        core->model_release(batch_model);
        check(task->result_count(batch_result, &count, &error) == TRTMC_OK && count == 2 &&
                  task->result_item_view(batch_result, 1, &item, &error) == TRTMC_OK &&
                  item.frame_count == 4 && fabs(item.frames[0].pixels[1] - 0.02) < 1e-6,
              "C batch retains ordered per-request video storage after model release");
        core->result_release(batch_result);
    }
    trtmc_model *model = NULL, *disabled = NULL;
    if (core->model_load(str(path), &options, &model, &error) != TRTMC_OK)
        return 2;
    if (core->model_load(str(disabled_path), &options, &disabled, &error) != TRTMC_OK)
        return 2;
    const trtmc_api_header* header = NULL;
    if (core->model_get_task_api(model, str(TRTMC_TASK_TEXT_TO_VIDEO), 1, 0, &header, &error) !=
        TRTMC_OK)
        return 2;
    const trtmc_text_to_video_api_v1* text = (const trtmc_text_to_video_api_v1*)header;
    trtmc_text_to_video_request_v1 input = {str("p"), {NULL, 0}};
    trtmc_result* result = NULL;
    check(text->run(model, &input, NULL, &result, &error) == TRTMC_OK,
          "C text-to-video call succeeds");
    trtmc_video_result_view_v1 view = {0};
    check(text->result_view(result, &view, &error) == TRTMC_OK && view.frame_count == 3 &&
              fabs(view.frames[0].pixels[0] - 0.01) < 1e-6 &&
              view.frames[1].pixels == view.frames[0].pixels + 3 &&
              view.timestamps_seconds.size == 3 &&
              fabs(view.timestamps_seconds.data[2] - 0.3) < 1e-9,
          "C sees ordered frame views into the owned contiguous buffer and actual timestamps");
    trtmc_result* rejected = NULL;
    check(text->run(disabled, &input, NULL, &rejected, &error) == TRTMC_UNSUPPORTED &&
              rejected == NULL,
          "video table cannot bypass another model's support declaration");
    release_error(core, &error);

    {
        const float replay_values[] = {0, -0.25F, 0.5F, -0.5F, 0.75F, 1};
        trtmc_result* replayed = NULL;
        trtmc_video_result_view_v1 replay_view = {0};
        input.initial_latents = (trtmc_f32_view){replay_values, 6};
        check(text->run(model, &input, NULL, &replayed, &error) == TRTMC_OK,
              "C video initial-latent replay executes");
        check(text->result_view(replayed, &replay_view, &error) == TRTMC_OK &&
                  memcmp(replay_values, replay_view.frames[1].pixels, sizeof(replay_values)) == 0,
              "C video preserves all raw float32 values");
        core->result_release(replayed);
        input.initial_latents.data = NULL;
        check(text->run(model, &input, NULL, &rejected, &error) == TRTMC_INVALID_ARGUMENT &&
                  rejected == NULL,
              "NULL video replay buffer fails before family execution");
        release_error(core, &error);
        input.initial_latents = (trtmc_f32_view){replay_values, UINT64_MAX};
        check(text->run(model, &input, NULL, &rejected, &error) == TRTMC_INVALID_ARGUMENT &&
                  rejected == NULL,
              "overflowed video replay length fails before dereference");
        release_error(core, &error);
        input.initial_latents.size = 3;
        check(text->run(model, &input, NULL, &rejected, &error) == TRTMC_INVALID_ARGUMENT &&
                  rejected == NULL,
              "family validates video replay count");
        release_error(core, &error);
        input.initial_latents = (trtmc_f32_view){NULL, 0};
    }

    if (core->model_get_task_api(model, str(TRTMC_TASK_TEXT_TO_AUDIO_VIDEO), 1, 0, &header,
                                 &error) != TRTMC_OK)
        return 2;
    const trtmc_text_to_audio_video_api_v1* av = (const trtmc_text_to_audio_video_api_v1*)header;
    trtmc_audio_video_result_view_v1 av_view = {0};
    check(av->result_view(result, &av_view, &error) == TRTMC_INVALID_ARGUMENT &&
              av_view.video.frame_count == 0,
          "video-only result cannot masquerade as synchronized audio/video");
    release_error(core, &error);
    const trtmc_text_to_audio_video_request_v1 av_input = {str("p")};
    trtmc_result* av_result = NULL;
    check(av->run(model, &av_input, NULL, &av_result, &error) == TRTMC_OK,
          "C synchronized AV call succeeds");
    check(av->result_view(av_result, &av_view, &error) == TRTMC_OK &&
              av_view.audio.audio.has_sample_rate == 1 && av_view.audio.audio.sample_rate == 10 &&
              av_view.audio.audio.channels == 2 && av_view.audio.audio.sample_count == 8 &&
              fabs(av_view.audio_start_seconds + 0.05) < 1e-9,
          "C AV result retains actual sample rate, stereo layout and a nonzero clock offset");

    if (core->model_get_task_api(model, str(TRTMC_TASK_TIMED_FRAMES_TEXT_TO_VIDEO), 1, 0, &header,
                                 &error) != TRTMC_OK)
        return 2;
    const trtmc_timed_frames_text_to_video_api_v1* timed =
        (const trtmc_timed_frames_text_to_video_api_v1*)header;
    trtmc_timed_video_anchor_v1 bad_anchor = {0};
    bad_anchor.kind = 99;
    trtmc_timed_frames_text_to_video_request_v1 bad_timed = {&bad_anchor, 1, str("p")};
    check(timed->run(model, &bad_timed, NULL, &rejected, &error) == TRTMC_INVALID_ARGUMENT &&
              rejected == NULL,
          "unknown temporal anchor discriminator is rejected");
    release_error(core, &error);

    const float pixels[] = {0.25F, 0, 0};
    const trtmc_image_input_v1 frames[] = {{pixels, sizeof(pixels), 1, 1, 3, TRTMC_IMAGE_FLOAT32},
                                           {pixels, sizeof(pixels), 1, 1, 3, TRTMC_IMAGE_FLOAT32}};
    const double times[] = {0, 0.1};
    const trtmc_video_view_v1 history = {frames, 2, {times, 2}};
    if (core->model_get_task_api(model, str(TRTMC_TASK_VIDEO_TO_ACTION_SEQUENCE), 1, 0, &header,
                                 &error) != TRTMC_OK)
        return 2;
    const trtmc_video_to_action_sequence_api_v1* inverse =
        (const trtmc_video_to_action_sequence_api_v1*)header;
    trtmc_video_to_action_sequence_request_v1 inverse_input = {0};
    inverse_input.observations = history;
    inverse_input.action_spec.schema.domain = str("fixture.motion");
    const trtmc_string_view names[] = {{"x", 1}, {"y", 1}};
    inverse_input.action_spec.schema.component_names = (trtmc_strings_view){names, 2};
    inverse_input.action_spec.dimensions = 2;
    trtmc_result* action_result = NULL;
    check(inverse->run(model, &inverse_input, NULL, &action_result, &error) == TRTMC_OK,
          "C inverse dynamics accepts an explicit output action schema");
    trtmc_action_sequence_view_v1 action_view = {0};
    check(inverse->result_view(action_result, &action_view, &error) == TRTMC_OK &&
              action_view.values.rows == 2 && action_view.values.columns == 2 &&
              action_view.schema.units.size == 0 && action_view.frame_span_count == 2 &&
              action_view.frame_spans[1].end == 2,
          "C action axes and frame association survive without invented units");
    inverse_input.observations.timestamps_seconds.size = 1;
    check(inverse->run(model, &inverse_input, NULL, &rejected, &error) == TRTMC_INVALID_ARGUMENT &&
              rejected == NULL,
          "mismatched video timeline is rejected before family call");
    release_error(core, &error);
    core->model_release(model);
    core->model_release(disabled);
    check(text->result_view(result, &view, &error) == TRTMC_OK && view.frames[2].pixel_count == 3,
          "C video frame storage survives model release");
    check(inverse->result_view(action_result, &action_view, &error) == TRTMC_OK &&
              action_view.schema.domain.size == 14 &&
              memcmp(action_view.schema.domain.data, "fixture.motion", 14) == 0,
          "C action schema strings survive model release");
    check(av->result_view(av_result, &av_view, &error) == TRTMC_OK &&
              av_view.audio.audio.sample_count == 8,
          "C paired audio/video resources survive model release");
    core->result_release(result);
    core->result_release(action_result);
    core->result_release(av_result);
    release_error(core, &error);
    return failures ? 1 : 0;
}

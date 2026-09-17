/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/tracking.h"
#include "trtmc/trtmc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const trtmc_core_api_v1* core;
static int failures;
static trtmc_string_view str(const char* s) {
    trtmc_string_view value = {s, (uint64_t)strlen(s)};
    return value;
}
static void check(int ok, const char* label) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", label);
        ++failures;
    }
}
static void status(trtmc_status value, trtmc_status expected, trtmc_error** error,
                   const char* label) {
    check(value == expected, label);
    core->error_release(*error);
    *error = NULL;
}
static void bundle(const char* file, const char* mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    char header[512];
    unsigned shift;
    FILE* out;
    int length = snprintf(header, sizeof(header),
                          "{\"format\":1,\"family\":\"tracking_fixture\",\"task\":\"%s\","
                          "\"backend\":\"fake\",\"sections\":{}}",
                          mode);
    if (length < 0 || (size_t)length >= sizeof(header) || (out = fopen(file, "wb")) == NULL)
        exit(2);
    fwrite(magic, 1, 8, out);
    for (shift = 0; shift < 64; shift += 8)
        fputc((int)(((uint64_t)length >> shift) & 255), out);
    fwrite(header, 1, (size_t)length, out);
    if (fclose(out))
        exit(2);
}
static void text_tracking(const char* path, const trtmc_load_options_v1* options,
                          const trtmc_video_view_v1* clip) {
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_frames_text_to_mask_tracks_api_v1* text = NULL;
    const trtmc_prompt_frame_text_to_mask_tracks_api_v1* prompt_api = NULL;
    trtmc_text_mask_clip_session* text_session = NULL;
    trtmc_text_prompt_frame_session* prompt_session = NULL;
    trtmc_result* complete = NULL;
    trtmc_result* prompt = NULL;
    trtmc_result* continuation = NULL;
    trtmc_result* rejected = NULL;
    trtmc_track_clip_view_v1 view = {0};
    status(core->model_load(str(path), options, &model, &error), TRTMC_OK, &error,
           "C text tracking model loads");
    if (!model)
        goto cleanup;
    status(core->model_get_task_api(model, str(TRTMC_TASK_FRAMES_TEXT_TO_MASK_TRACKS), 1, 0,
                                    &header, &error),
           TRTMC_OK, &error, "C text clip table");
    text = (const trtmc_frames_text_to_mask_tracks_api_v1*)header;
    if (!text)
        goto cleanup;
    status(text->create(model, NULL, &text_session, &error), TRTMC_OK, &error,
           "C text clip session creates");
    if (!text_session)
        goto cleanup;
    status(text->segment(text_session, clip, str("bird"), NULL, &complete, &error), TRTMC_OK,
           &error, "C text clip delegates exact prompt");
    status(text->result_view(complete, &view, &error), TRTMC_OK, &error, "C float mask clip view");
    check(view.frame_count == 5 && view.frames[0].element_type == TRTMC_TRACK_FLOAT32 &&
              view.frames[0].removed_object_ids.data[0] == 19 &&
              view.frames[0].suppressed_object_ids.data[0] == 23,
          "C text result retains binary float masks and removal metadata");
    text->release(text_session);
    text_session = NULL;
    status(core->model_get_task_api(model, str(TRTMC_TASK_PROMPT_FRAME_TEXT_TO_MASK_TRACKS), 1, 0,
                                    &header, &error),
           TRTMC_OK, &error, "C two-phase protocol has its own table");
    prompt_api = (const trtmc_prompt_frame_text_to_mask_tracks_api_v1*)header;
    if (!prompt_api)
        goto cleanup;
    status(prompt_api->create(model, str("bird"), NULL, &prompt_session, &error), TRTMC_OK, &error,
           "C two-phase session creates");
    if (!prompt_session)
        goto cleanup;
    status(prompt_api->continue_borrowed(prompt_session, complete, clip, &rejected, &error),
           TRTMC_INVALID_ARGUMENT, &error, "C rejects unrelated result as a prompt snapshot");
    check(!rejected, "invalid snapshot clears output");
    status(prompt_api->accept_prompt_frame(prompt_session, &clip->frames[0], &prompt, &error),
           TRTMC_OK, &error, "C prompt frame emits an owned snapshot");
    status(prompt_api->continue_borrowed(prompt_session, prompt, clip, &continuation, &error),
           TRTMC_OK, &error, "C continuation borrows caller-owned prompt snapshot");
    status(prompt_api->result_view(continuation, &view, &error), TRTMC_OK, &error,
           "C continuation output view");
    check(view.frame_count == 5 && ((const float*)view.frames[0].masks)[0] == 0.0F,
          "C consolidation replaces frame zero; shared ABI does not prepend snapshot");
    status(prompt_api->result_view(prompt, &view, &error), TRTMC_OK, &error,
           "C prompt snapshot remains owned");
    check(view.frame_count == 1 && ((const float*)view.frames[0].masks)[0] == 1.0F,
          "C continuation does not consume or mutate prompt snapshot");
    status(prompt_api->continue_borrowed(prompt_session, prompt, clip, &rejected, &error),
           TRTMC_INVALID_ARGUMENT, &error, "family enforces one native continuation");
    prompt_api->release(prompt_session);
    prompt_session = NULL;
    status(prompt_api->result_view(prompt, &view, &error), TRTMC_OK, &error,
           "owned prompt outlives native session");
cleanup:
    if (text)
        text->release(text_session);
    if (prompt_api)
        prompt_api->release(prompt_session);
    core->result_release(complete);
    core->result_release(prompt);
    core->result_release(continuation);
    core->result_release(rejected);
    core->model_release(model);
}
static void image_contexts(const char* path, const char* limited_path,
                           const trtmc_load_options_v1* options) {
    trtmc_model* model = NULL;
    trtmc_model* limited = NULL;
    trtmc_error* error = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_interactive_image_masks_api_v1* api = NULL;
    trtmc_image_mask_context* context = NULL;
    trtmc_image_mask_context* restricted = NULL;
    trtmc_image_mask_editors_v1 editors = {0}, hidden = {0};
    trtmc_result* first = NULL;
    trtmc_result* result = NULL;
    trtmc_masks_view_v1 view = {0};
    float pixels[18] = {0.25F}, logits[4] = {-3, 1, 2, 4};
    trtmc_image_input_v1 image = {pixels, sizeof(pixels), 2, 3, 3, TRTMC_IMAGE_FLOAT32};
    trtmc_point_prompt_v1 point = {{1, 1}, 1};
    trtmc_image_prior_prompt_v1 prior = {{logits, 4, 2, 2}, NULL, 0, 0, {0, 0, 0, 0}};
    status(core->model_load(str(path), options, &model, &error), TRTMC_OK, &error,
           "C image context model");
    status(core->model_load(str(limited_path), options, &limited, &error), TRTMC_OK, &error,
           "C limited context model");
    if (!model || !limited)
        goto cleanup;
    status(core->model_get_task_api(model, str(TRTMC_TASK_INTERACTIVE_IMAGE_MASKS), 1, 0, &header,
                                    &error),
           TRTMC_OK, &error, "C image context Task");
    api = (const trtmc_interactive_image_masks_api_v1*)header;
    if (!api)
        goto cleanup;
    status(api->create(model, &image, NULL, &context, &error), TRTMC_OK, &error,
           "C image encoding creates context");
    status(api->create(limited, &image, NULL, &restricted, &error), TRTMC_OK, &error,
           "C limited image context");
    if (!context || !restricted)
        goto cleanup;
    pixels[0] = 0.75F;
    status(api->get_editors(context, &editors, &error), TRTMC_OK, &error, "C typed image editors");
    status(api->get_editors(restricted, &hidden, &error), TRTMC_OK, &error,
           "C limited typed image editors");
    check(editors.points && editors.box && editors.prior && hidden.points && !hidden.prior,
          "C editor pointers are projected from native getters");
    if (!editors.points || !editors.prior)
        goto cleanup;
    status(editors.points->run(context, &point, 1, NULL, &first, &error), TRTMC_OK, &error,
           "C point editor");
    status(api->result_view(first, &view, &error), TRTMC_OK, &error, "C image result view");
    check(view.masks[0] == 0.25F && view.predicted_iou[0] == 1,
          "C family owns encoded input before create returns");
    point.foreground = 0;
    status(editors.points->run(context, &point, 1, NULL, &result, &error), TRTMC_OK, &error,
           "C background point editor");
    status(api->result_view(result, &view, &error), TRTMC_OK, &error, "C second image result");
    check(view.masks[0] == -0.25F && view.predicted_iou[0] == 1,
          "C repeated prompts reuse one encoding");
    core->result_release(result);
    result = NULL;
    status(editors.prior->run(restricted, &prior, NULL, &result, &error), TRTMC_UNSUPPORTED, &error,
           "C foreign editor table cannot bypass target context getter");
    check(!result, "C unsupported editor leaves no result");
    status(editors.prior->run(context, &prior, NULL, &result, &error), TRTMC_OK, &error,
           "C decoder prior editor");
    status(api->result_view(result, &view, &error), TRTMC_OK, &error, "C prior result");
    check(view.masks[0] == -3, "C prior logits remain signed decoder-space values");
    api->release(context);
    context = NULL;
    status(api->result_view(first, &view, &error), TRTMC_OK, &error,
           "C mask result outlives image context");
cleanup:
    if (api) {
        api->release(context);
        api->release(restricted);
    }
    core->result_release(first);
    core->result_release(result);
    core->model_release(model);
    core->model_release(limited);
}
static void interactive_tracking(const char* path, const char* limited_path,
                                 const trtmc_load_options_v1* options,
                                 const trtmc_video_view_v1* clip) {
    trtmc_model* model = NULL;
    trtmc_model* limited = NULL;
    trtmc_error* error = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_frames_points_to_mask_tracks_api_v1* factory = NULL;
    const trtmc_mask_track_session_api_v1* api = NULL;
    trtmc_mask_track_session* session = NULL;
    trtmc_mask_track_session* restricted = NULL;
    trtmc_track_propagation* propagation = NULL;
    trtmc_mask_track_editors_v1 editors = {0}, hidden = {0};
    trtmc_result* initial = NULL;
    trtmc_result* limited_initial = NULL;
    trtmc_result* result = NULL;
    trtmc_track_clip_view_v1 view = {0};
    trtmc_point_prompt_v1 point = {{1, 1}, 1};
    trtmc_frame_object_points_v1 prompt = {2, 41, &point, 1, TRTMC_POINTS_REPLACE};
    trtmc_frame_text_v1 text_prompt = {0, {"bird", 4}};
    trtmc_propagation_range_v1 range = {4, 3, TRTMC_PROPAGATE_BACKWARD};
    int i;
    status(core->model_load(str(path), options, &model, &error), TRTMC_OK, &error,
           "C interactive model");
    status(core->model_load(str(limited_path), options, &limited, &error), TRTMC_OK, &error,
           "C limited tracking model");
    if (!model || !limited)
        goto cleanup;
    status(core->model_get_task_api(model, str(TRTMC_TASK_FRAMES_POINTS_TO_MASK_TRACKS), 1, 0,
                                    &header, &error),
           TRTMC_OK, &error, "C typed points tracking factory");
    factory = (const trtmc_frames_points_to_mask_tracks_api_v1*)header;
    if (!factory)
        goto cleanup;
    api = factory->session_api;
    status(factory->create(model, clip, &prompt, NULL, &session, &initial, &error), TRTMC_OK,
           &error, "C interactive points initialization");
    status(factory->create(limited, clip, &prompt, NULL, &restricted, &limited_initial, &error),
           TRTMC_OK, &error, "C limited interactive initialization");
    if (!session || !restricted)
        goto cleanup;
    status(api->result_view(initial, &view, &error), TRTMC_OK, &error, "C prompted-frame snapshot");
    check(view.frame_count == 1 && view.frames[0].frame_index == 2 &&
              view.frames[0].object_ids.data[0] == 41,
          "C typed initial prompt preserves frame and object IDs");
    status(api->get_editors(session, &editors, &error), TRTMC_OK, &error,
           "C interactive typed getters");
    status(api->get_editors(restricted, &hidden, &error), TRTMC_OK, &error,
           "C limited typed getters");
    check(editors.points && editors.box && editors.mask && editors.text && editors.exemplar &&
              editors.objects && editors.reset && hidden.points && !hidden.text,
          "C getters project only actual native editor pointers");
    if (!editors.points || !editors.text || !editors.reset)
        goto cleanup;
    status(editors.text->run(restricted, &text_prompt, NULL, &result, &error), TRTMC_UNSUPPORTED,
           &error, "C editor table reuse cannot bypass target session getter");
    point.foreground = 0;
    prompt.update = TRTMC_POINTS_APPEND;
    status(editors.points->run(session, &prompt, NULL, &result, &error), TRTMC_OK, &error,
           "C point append");
    status(api->result_view(result, &view, &error), TRTMC_OK, &error, "C appended prompt output");
    check(view.frames[0].tracker_scores[0] == 2 && ((const float*)view.frames[0].masks)[0] == 0,
          "C append and foreground polarity reach family state");
    core->result_release(result);
    result = NULL;
    status(api->start_propagation(session, &range, &propagation, &error), TRTMC_OK, &error,
           "C reverse traversal starts");
    status(editors.reset->reset(session, &error), TRTMC_BUSY, &error,
           "C traversal excludes parent reset");
    for (i = 0; i < 3; ++i) {
        status(api->next(propagation, &result, &error), TRTMC_OK, &error,
               "C traversal native next");
        status(api->result_view(result, &view, &error), TRTMC_OK, &error, "C traversal frame view");
        check(view.frames[0].frame_index == (uint64_t)(4 - i),
              "C reverse preserves original frame IDs");
        core->result_release(result);
        result = NULL;
    }
    status(api->next(propagation, &result, &error), TRTMC_END, &error, "C traversal END");
    check(!result && !error, "C END clears output and has no owned error");
    api->release_propagation(propagation);
    propagation = NULL;
    status(editors.reset->reset(session, &error), TRTMC_OK, &error,
           "C reset delegates after traversal release");
    status(api->start_propagation(session, &range, &propagation, &error), TRTMC_INVALID_ARGUMENT,
           &error, "C family requires prompt after reset");
    status(editors.text->run(session, &text_prompt, NULL, &result, &error), TRTMC_OK, &error,
           "C text replacement after reset");
    core->result_release(result);
    result = NULL;
    range = (trtmc_propagation_range_v1){0, 2, TRTMC_PROPAGATE_FORWARD};
    status(api->start_propagation(session, &range, &propagation, &error), TRTMC_OK, &error,
           "C retained traversal starts");
    api->release(session);
    session = NULL;
    core->model_release(model);
    model = NULL;
    status(api->next(propagation, &result, &error), TRTMC_OK, &error,
           "C traversal retains parent native model after handle releases");
    core->result_release(result);
    result = NULL;
    status(api->cancel(propagation, &error), TRTMC_OK, &error, "C traversal cancel");
    status(api->next(propagation, &result, &error), TRTMC_END, &error, "C cancel ends traversal");
    api->release_propagation(propagation);
    propagation = NULL;
    status(api->result_view(initial, &view, &error), TRTMC_OK, &error,
           "C initial snapshot outlives edits reset and parents");
    check(view.frames[0].object_ids.data[0] == 41, "C owned snapshot is immutable");
cleanup:
    if (api) {
        api->release_propagation(propagation);
        api->release(session);
        api->release(restricted);
    }
    core->result_release(initial);
    core->result_release(limited_initial);
    core->result_release(result);
    core->model_release(model);
    core->model_release(limited);
}
typedef struct {
    const trtmc_crop_pose_tracking_api_v1* api;
    trtmc_crop_pose_session* session;
    int released, refinement_calls, scoring_calls, fail;
} pose_callback_state;
typedef struct {
    pose_callback_state* state;
    float data[24];
} pose_crop_lease;
static void TRTMC_CALL release_test_pose_crops(void* owner) {
    pose_crop_lease* lease = (pose_crop_lease*)owner;
    ++lease->state->released;
    free(lease);
}
static trtmc_status TRTMC_CALL pose_crops(void* context, const trtmc_pose_crop_request_v1* query,
                                          trtmc_pose_crop_response_v1* out) {
    pose_callback_state* state = (pose_callback_state*)context;
    pose_crop_lease* lease;
    trtmc_error* error = NULL;
    uint64_t i, count;
    *out = (trtmc_pose_crop_response_v1){0};
    if (query->poses.count > 2)
        return TRTMC_INVALID_ARGUMENT;
    status(state->api->reset(state->session, &error), TRTMC_BUSY, &error,
           "C crop callback same-session reentry is BUSY, not deadlocked");
    if (query->stage == TRTMC_POSE_CROP_REFINEMENT)
        ++state->refinement_calls;
    else if (query->stage == TRTMC_POSE_CROP_SCORING)
        ++state->scoring_calls;
    else
        check(0, "C pose crop stage remains typed");
    lease = (pose_crop_lease*)malloc(sizeof(*lease));
    if (!lease)
        return TRTMC_OUT_OF_MEMORY;
    lease->state = state;
    count = query->poses.count * 6;
    for (i = 0; i < count; ++i) {
        lease->data[i] = 0.1F;
        lease->data[count + i] = 0.25F;
    }
    out->rendered = lease->data;
    out->rendered_count = count;
    out->observed = lease->data + count;
    out->observed_count = count;
    out->hypothesis_count = query->poses.count;
    out->height = 1;
    out->width = 1;
    out->channels = 6;
    out->owner = lease;
    out->release = release_test_pose_crops;
    if (state->fail) {
        out->error_message = str("test crop failure");
        return TRTMC_INVALID_CONFIG;
    }
    return TRTMC_OK;
}
static void pose_tracking(const char* path, const trtmc_load_options_v1* options,
                          const trtmc_image_input_v1* image) {
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_crop_pose_tracking_api_v1* crop = NULL;
    const trtmc_rgbd_initialized_pose_to_tracked_pose_api_v1* rgbd = NULL;
    trtmc_crop_pose_session* session = NULL;
    trtmc_rgbd_pose_session* rgbd_session = NULL;
    trtmc_result* initialized = NULL;
    trtmc_result* result = NULL;
    trtmc_refined_poses_view_v1 poses = {0};
    trtmc_object_pose_view_v1 object = {0};
    pose_callback_state callbacks = {0};
    trtmc_pose_refinement_request_v1 request = {0};
    trtmc_triangle_mesh_v1 mesh = {0};
    trtmc_object_pose_matrix_v1 original = {0};
    trtmc_rgbd_observation_v1 observation = {0};
    float candidates[32] = {0}, vertices[9] = {2, 0, 0, 0, 1, 0, 0, 0, 1},
          depth[6] = {0.5F, 1, 1, 1, 1, 1};
    uint32_t triangles[3] = {0, 1, 2};
    int i, released;
    status(core->model_load(str(path), options, &model, &error), TRTMC_OK, &error,
           "C pose tracking model");
    if (!model)
        goto cleanup;
    status(
        core->model_get_task_api(model, str(TRTMC_TASK_CROP_POSE_TRACKING), 1, 0, &header, &error),
        TRTMC_OK, &error, "C native crop pose Task");
    crop = (const trtmc_crop_pose_tracking_api_v1*)header;
    if (!crop)
        goto cleanup;
    status(crop->create(model, NULL, &session, &error), TRTMC_OK, &error,
           "C crop pose session creates");
    if (!session)
        goto cleanup;
    callbacks.api = crop;
    callbacks.session = session;
    status(crop->track(session, &callbacks, pose_crops, NULL, &result, &error),
           TRTMC_INVALID_ARGUMENT, &error, "C family track requires stored initial pose");
    for (i = 0; i < 4; ++i) {
        candidates[i * 5] = 1;
        candidates[16 + i * 5] = 1;
        original.object_to_camera[i * 5] = 1;
    }
    candidates[3] = 0.5F;
    candidates[19] = 0.75F;
    request.candidates = (trtmc_pose_matrices_v1){candidates, 32, 2};
    request.mesh_diameter_meters = 2;
    request.context = &callbacks;
    request.provide_crops = pose_crops;
    status(crop->initialize(session, &request, NULL, &initialized, &error), TRTMC_OK, &error,
           "C pose initializes using caller crop leases");
    status(crop->result_view(initialized, &poses, &error), TRTMC_OK, &error,
           "C initial refined poses");
    check(poses.best_index == 1 && callbacks.refinement_calls == 1 &&
              callbacks.scoring_calls == 1 && callbacks.released == 2,
          "C both crop stages release each provided lease exactly once");
    status(crop->track(session, &callbacks, pose_crops, NULL, &result, &error), TRTMC_OK, &error,
           "C saved selected pose tracking");
    status(crop->result_view(result, &poses, &error), TRTMC_OK, &error, "C tracked pose output");
    check(poses.refined_poses.count == 1 && poses.refined_poses.values[3] > 1.149F &&
              poses.refined_poses.values[3] < 1.151F,
          "C family owns selected pose and mesh diameter");
    core->result_release(result);
    result = NULL;
    callbacks.fail = 1;
    released = callbacks.released;
    status(crop->track(session, &callbacks, pose_crops, NULL, &result, &error),
           TRTMC_INVALID_CONFIG, &error, "C crop callback error preserves owned status");
    check(!result && callbacks.released == released + 1,
          "C callback failure still releases provided lease exactly once");
    callbacks.fail = 0;
    status(crop->track(session, &callbacks, pose_crops, NULL, &result, &error), TRTMC_OK, &error,
           "C callback failure does not strand session operation lock");
    core->result_release(result);
    result = NULL;
    status(crop->reset(session, &error), TRTMC_OK, &error, "C crop pose reset");
    status(crop->track(session, &callbacks, pose_crops, NULL, &result, &error),
           TRTMC_INVALID_ARGUMENT, &error, "C crop reset clears stored native pose");
    crop->release(session);
    session = NULL;
    status(crop->result_view(initialized, &poses, &error), TRTMC_OK, &error,
           "C pose snapshot outlives reset/release");
    status(core->model_get_task_api(model, str(TRTMC_TASK_RGBD_INITIALIZED_POSE_TO_TRACKED_POSE), 1,
                                    0, &header, &error),
           TRTMC_OK, &error, "C raw RGBD contract is separate from crop tracking");
    rgbd = (const trtmc_rgbd_initialized_pose_to_tracked_pose_api_v1*)header;
    if (!rgbd)
        goto cleanup;
    mesh.vertices = (trtmc_f32_matrix_view_v1){vertices, 9, 3, 3};
    mesh.triangles = triangles;
    mesh.triangle_index_count = 3;
    original.object_to_camera[3] = 7;
    original.object_to_camera[11] = 3;
    status(rgbd->create(model, &mesh, &original, NULL, &rgbd_session, &error), TRTMC_OK, &error,
           "C RGBD mesh and initialized pose session");
    if (!rgbd_session)
        goto cleanup;
    vertices[0] = 99;
    observation.rgb = *image;
    observation.depth_meters = (trtmc_f32_matrix_view_v1){depth, 6, 2, 3};
    observation.pixel_intrinsics[0] = 100;
    observation.pixel_intrinsics[4] = 100;
    observation.pixel_intrinsics[8] = 1;
    status(rgbd->track(rgbd_session, &observation, NULL, &result, &error), TRTMC_OK, &error,
           "C aligned meter-depth pose tracking");
    status(rgbd->result_view(result, &object, &error), TRTMC_OK, &error,
           "C original-object pose output");
    check(object.object_to_camera[3] == 7 && object.object_to_camera[11] == 3.5F &&
              object.score == 2,
          "C family owns retained mesh and preserves original-object frame");
    core->result_release(result);
    result = NULL;
    original.object_to_camera[3] = 9;
    original.object_to_camera[11] = 4;
    status(rgbd->reset(rgbd_session, &original, &error), TRTMC_OK, &error,
           "C RGBD explicit pose reset");
    observation.pixel_intrinsics[0] = 0;
    status(rgbd->track(rgbd_session, &observation, NULL, &result, &error), TRTMC_INVALID_ARGUMENT,
           &error, "C pixel intrinsics reject zero focal length before native state mutation");
    observation.pixel_intrinsics[0] = 100;
    status(rgbd->track(rgbd_session, &observation, NULL, &result, &error), TRTMC_OK, &error,
           "C RGBD resumes from explicit pose");
    status(rgbd->result_view(result, &object, &error), TRTMC_OK, &error, "C reset RGBD output");
    check(object.object_to_camera[3] == 9 && object.object_to_camera[11] == 4.5F,
          "C RGBD reset replaces state without hidden coordinate conversion");
cleanup:
    if (crop)
        crop->release(session);
    if (rgbd)
        rgbd->release(rgbd_session);
    core->result_release(initialized);
    core->result_release(result);
    core->model_release(model);
}
int main(int argc, char** argv) {
    char enabled_path[4096], host_path[4096];
    trtmc_load_options_v1 options = {0};
    trtmc_model* model = NULL;
    trtmc_model* host_model = NULL;
    trtmc_error* error = NULL;
    const trtmc_api_header* header = NULL;
    const trtmc_frames_to_detected_mask_tracks_api_v1* task = NULL;
    const trtmc_detected_device_masks_api_v1* device = NULL;
    const trtmc_detected_device_masks_api_v1* rejected = NULL;
    trtmc_detected_mask_session* session = NULL;
    trtmc_detected_mask_session* host_session = NULL;
    trtmc_detected_mask_session* competing = NULL;
    float pixels[18] = {0};
    trtmc_image_input_v1 frames[5];
    trtmc_video_view_v1 clip = {0};
    trtmc_result* host_result = NULL;
    trtmc_result* borrowed = NULL;
    trtmc_result* result = NULL;
    trtmc_track_clip_view_v1 view = {0};
    size_t i;
    if (argc != 2 || strlen(argv[1]) + 40 >= sizeof(enabled_path) ||
        trtmc_get_api(1, 0, &core) != TRTMC_OK)
        return 2;
    snprintf(enabled_path, sizeof(enabled_path), "%s/tracking-c-device.bundle", argv[1]);
    snprintf(host_path, sizeof(host_path), "%s/tracking-c-host.bundle", argv[1]);
    bundle(enabled_path, "device");
    bundle(host_path, "host_only");
    options.struct_size = sizeof(options);
    options.runtime_root = str(argv[1]);
    status(core->model_load(str(enabled_path), &options, &model, &error), TRTMC_OK, &error,
           "C detector model loads");
    status(core->model_load(str(host_path), &options, &host_model, &error), TRTMC_OK, &error,
           "C host-only model loads");
    if (!model || !host_model)
        goto cleanup;
    status(core->model_get_task_api(model, str(TRTMC_TASK_FRAMES_TO_DETECTED_MASK_TRACKS), 1, 0,
                                    &header, &error),
           TRTMC_OK, &error, "C tracking Task table");
    task = (const trtmc_frames_to_detected_mask_tracks_api_v1*)header;
    if (!task)
        goto cleanup;
    status(task->create(model, NULL, &session, &error), TRTMC_OK, &error,
           "C native detector session creates");
    status(task->create(model, NULL, &competing, &error), TRTMC_BUSY, &error,
           "live session owns model execution");
    check(!competing, "BUSY leaves no session");
    status(task->create(host_model, NULL, &host_session, &error), TRTMC_OK, &error,
           "host-only session creates");
    if (!session || !host_session)
        goto cleanup;
    for (i = 0; i < 5; ++i)
        frames[i] = (trtmc_image_input_v1){pixels, sizeof(pixels), 2, 3, 3, TRTMC_IMAGE_FLOAT32};
    clip.frames = frames;
    clip.frame_count = 5;
    status(task->segment(session, &clip, NULL, &host_result, &error), TRTMC_OK, &error,
           "C host clip segment");
    status(task->result_view(host_result, &view, &error), TRTMC_OK, &error, "C owned host result");
    check(view.frame_count == 5 && view.frames[4].frame_index == 4 &&
              view.frames[0].mask_byte_size == 6 &&
              view.frames[0].element_type == TRTMC_TRACK_UINT8,
          "fixed clip retains uint8 masks");
    check(((const unsigned char*)view.frames[0].masks)[0] == 1 &&
              view.frames[0].object_ids.data[0] == 7 && view.initial_detections[0].class_id == 2,
          "host bytes and detector label remain distinct from object ID");
    status(task->get_device_api(session, 1, 0, &device, &error), TRTMC_OK, &error,
           "C typed device getter");
    status(task->get_device_api(host_session, 1, 0, &rejected, &error), TRTMC_UNSUPPORTED, &error,
           "typed getter alone declares device absence");
    check(!rejected, "unavailable device table cleared");
    if (!device)
        goto cleanup;
    status(device->segment_device(host_session, &clip, NULL, &result, &error), TRTMC_UNSUPPORTED,
           &error, "foreign table does not bypass session device getter");
    status(device->segment_device(session, &clip, NULL, &borrowed, &error), TRTMC_OK, &error,
           "C borrowed device clip");
    status(device->result_view(borrowed, &view, &error), TRTMC_OK, &error, "C device descriptor");
    check(view.frames[0].memory_kind == TRTMC_TRACK_CUDA && view.frames[0].device_ordinal == 2 &&
              view.frames[0].masks == (const void*)(uintptr_t)0x10000,
          "device address is exposed without CPU dereference");
    status(task->segment(session, NULL, NULL, &result, &error), TRTMC_INVALID_ARGUMENT, &error,
           "pre-call invalid clip rejected");
    status(device->result_view(borrowed, &view, &error), TRTMC_OK, &error,
           "pre-call error does not invalidate earlier device view");
    clip.frame_count = 4;
    status(task->segment(session, &clip, NULL, &result, &error), TRTMC_INVALID_ARGUMENT, &error,
           "family fixed-clip validation");
    status(device->result_view(borrowed, &view, &error), TRTMC_INVALID_ARGUMENT, &error,
           "delegated failing call invalidates earlier device view");
    core->result_release(borrowed);
    borrowed = NULL;
    clip.frame_count = 5;
    core->model_release(model);
    model = NULL;
    status(device->segment_device(session, &clip, NULL, &borrowed, &error), TRTMC_OK, &error,
           "session retains model after C model handle release");
    task->release(session);
    session = NULL;
    status(device->result_view(borrowed, &view, &error), TRTMC_INVALID_ARGUMENT, &error,
           "session close invalidates borrowed addresses");
    status(task->result_view(host_result, &view, &error), TRTMC_OK, &error,
           "host result outlives model/session");
    text_tracking(enabled_path, &options, &clip);
    image_contexts(enabled_path, host_path, &options);
    interactive_tracking(enabled_path, host_path, &options, &clip);
    pose_tracking(enabled_path, &options, &frames[0]);
cleanup:
    if (task) {
        task->release(session);
        task->release(host_session);
        task->release(competing);
        task->release(NULL);
    }
    core->model_release(model);
    core->model_release(host_model);
    core->result_release(host_result);
    core->result_release(borrowed);
    core->result_release(result);
    fprintf(stderr, "%s\n", failures ? "SOME FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}

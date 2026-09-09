/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/perception.h"
#include "trtmc/trtmc.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const trtmc_core_api_v1* api;
static int failures;
static trtmc_string_view str(const char* value) {
    trtmc_string_view out = {value, (uint64_t)strlen(value)};
    return out;
}
static int same(trtmc_string_view value, const char* expected) {
    return value.size == strlen(expected) &&
           (value.size == 0 || memcmp(value.data, expected, (size_t)value.size) == 0);
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
    check(status == TRTMC_OK ? *error == NULL : *error != NULL, "C error ownership");
    api->error_release(*error);
    *error = NULL;
}
static const trtmc_api_header* table(trtmc_model* model, const char* id) {
    const trtmc_api_header* out = NULL;
    trtmc_error* error = NULL;
    checked(api->model_get_task_api(model, str(id), 1, 0, &out, &error), TRTMC_OK, &error,
            "C perception table lookup");
    return out;
}
static void bundle(const char* path, const char* mode) {
    const unsigned char magic[] = {'B', 'U', 'N', 'D', 'L', 'E', 1, 0};
    char header[512];
    unsigned shift;
    FILE* output;
    const int length = snprintf(header, sizeof(header),
                                "{\"format\":1,\"family\":\"perception_fixture\",\"task\":\"%s\","
                                "\"backend\":\"fake\",\"sections\":{}}",
                                mode);
    if (length < 0 || (size_t)length >= sizeof(header) || (output = fopen(path, "wb")) == NULL)
        exit(2);
    fwrite(magic, 1, 8, output);
    for (shift = 0; shift < 64; shift += 8)
        fputc((int)(((uint64_t)length >> shift) & 255), output);
    fwrite(header, 1, (size_t)length, output);
    if (fclose(output))
        exit(2);
}

static void boxes_batch(const char* root, const trtmc_load_options_v1* options,
                        trtmc_model* single_model) {
    char path[4096];
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    trtmc_result *result = NULL, *rejected = NULL;
    trtmc_grounded_boxes_view_v1 view = {0};
    float small[18] = {0}, large[60] = {0};
    const trtmc_image_input_v1 a = {small, sizeof(small), 2, 3, 3, TRTMC_IMAGE_FLOAT32};
    const trtmc_image_input_v1 b = {large, sizeof(large), 4, 5, 3, TRTMC_IMAGE_FLOAT32};
    trtmc_batch_image_text_to_boxes_item_v1 items[2] = {0};
    trtmc_batch_image_text_to_boxes_request_v1 request = {items, 2};
    trtmc_config_entry_v1 score = {0};
    uint64_t count = 0;
    if (snprintf(path, sizeof(path), "%s/perception-c-batch.bundle", root) >= (int)sizeof(path))
        exit(2);
    bundle(path, "batch");
    checked(api->model_load(str(path), options, &model, &error), TRTMC_OK, &error,
            "C batch-only boxes model loads");
    if (!model)
        return;
    const trtmc_batch_image_text_to_boxes_api_v1* batch =
        (const trtmc_batch_image_text_to_boxes_api_v1*)table(model,
                                                             TRTMC_TASK_BATCH_IMAGE_TEXT_TO_BOXES);
    if (!batch) {
        api->model_release(model);
        return;
    }
    score.name = str("score");
    score.value.kind = TRTMC_CONFIG_F64;
    score.value.as.f64 = 0;
    items[0].input = (trtmc_image_query_request_v1){a, {"cup\0mug", 7}};
    items[0].config = (trtmc_config_view_v1){&score, 1};
    items[1].input = (trtmc_image_query_request_v1){b, str("two")};
    checked(batch->run(model, &request, &result, &error), TRTMC_OK, &error,
            "C real B2 typed boxes call");
    checked(batch->result_count(result, &count, &error), TRTMC_OK, &error, "C boxes batch count");
    checked(batch->result_item_view(result, 0, &view, &error), TRTMC_OK, &error,
            "C first boxes batch view");
    check(count == 2 && view.count == 1 && view.boxes[0].label.size == 7 &&
              memcmp(view.boxes[0].label.data, "cup\0mug", 7) == 0 &&
              view.boxes[0].has_confidence && view.boxes[0].confidence == 0 &&
              fabs(view.boxes[0].box.x_min - 0.3) < 1e-6,
          "C first item preserves NUL label and explicit zero");
    checked(batch->result_item_view(result, 1, &view, &error), TRTMC_OK, &error,
            "C second boxes batch view");
    check(view.count == 2 && fabs(view.boxes[0].box.x_max - 4) < 1e-6 &&
              fabs(view.boxes[0].box.y_max - 3.6) < 1e-6,
          "C per-image pixel scale and ragged detections");
    checked(batch->run(single_model, &request, &rejected, &error), TRTMC_UNSUPPORTED, &error,
            "C batch table cannot grant batch support to single-only model");
    check(!rejected, "C unsupported native batch has no fallback result");
    checked(batch->result_item_view(result, 2, &view, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C boxes batch item bounds");
    checked(batch->result_count(result, NULL, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C boxes batch count null output");
    trtmc_batch_image_text_to_boxes_request_v1 malformed = {NULL, 2};
    checked(batch->run(model, &malformed, &rejected, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C null boxes batch array");
    malformed.items = items;
    malformed.count = UINT64_MAX;
    checked(batch->run(model, &malformed, &rejected, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C overflowed batch count rejected before dereference");
    score.name = str("unknown");
    checked(batch->run(model, &request, &rejected, &error), TRTMC_INVALID_CONFIG, &error,
            "C family rejects undeclared batch Config");
    score.name = str("score");
    items[1].input.text = (trtmc_string_view){NULL, 2};
    checked(batch->run(model, &request, &rejected, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C second-item null text rejected before native batch");
    items[1].input = (trtmc_image_query_request_v1){
        {small, 6 * sizeof(float), 2, 3, 1, TRTMC_IMAGE_FLOAT32}, str("gray")};
    checked(batch->run(model, &request, &rejected, &error), TRTMC_UNSUPPORTED, &error,
            "C family preflight owns native image profile restrictions");
    items[0].input.text = str("empty");
    items[1].input = (trtmc_image_query_request_v1){b, str("malformed")};
    checked(batch->run(model, &request, &rejected, &error), TRTMC_OK, &error,
            "C empty and unparsed typed batch items");
    checked(batch->result_item_view(rejected, 0, &view, &error), TRTMC_OK, &error,
            "C empty detection view");
    check(view.count == 0 && view.parse_complete, "C empty detection is a real successful result");
    checked(batch->result_item_view(rejected, 1, &view, &error), TRTMC_OK, &error,
            "C incomplete parse view");
    check(view.count == 0 && !view.parse_complete && same(view.raw_response, "batch:malformed"),
          "C incomplete parse retains raw model output");
    api->result_release(rejected);
    rejected = NULL;
    const trtmc_image_text_to_boxes_api_v1* single = (const trtmc_image_text_to_boxes_api_v1*)table(
        single_model, TRTMC_TASK_IMAGE_TEXT_TO_BOXES);
    trtmc_image_query_request_v1 one = {a, str("single")};
    checked(single->run(single_model, &one, NULL, &rejected, &error), TRTMC_OK, &error,
            "C single boxes control");
    checked(batch->result_item_view(rejected, 0, &view, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C single result cannot masquerade as batch storage");
    api->result_release(rejected);
    rejected = NULL;
    api->model_release(model);
    checked(batch->result_item_view(result, 0, &view, &error), TRTMC_OK, &error,
            "C boxes batch nested storage survives model release");
    check(view.boxes[0].label.size == 7 && view.raw_response.size == 13,
          "C owned batch strings remain length-preserving");
    api->result_release(result);
    for (int fault = 0; fault < 2; ++fault) {
        bundle(path, fault ? "batch_bad_count" : "batch_fail");
        model = NULL;
        checked(api->model_load(str(path), options, &model, &error), TRTMC_OK, &error,
                "C failed batch fixture load");
        checked(batch->run(model, &request, &rejected, &error), TRTMC_INTERNAL_ERROR, &error,
                "C execution or wrong-count batch returns call-level failure");
        check(!rejected, "C no partial result escapes failed native batch");
        api->model_release(model);
    }
}

static trtmc_model* load_required_mode(const char* root, const char* mode,
                                       const trtmc_load_options_v1* options) {
    char path[4096];
    trtmc_model* model = NULL;
    trtmc_error* error = NULL;
    if (snprintf(path, sizeof(path), "%s/perception-c-required-%s.bundle", root, mode) >=
        (int)sizeof(path))
        exit(2);
    bundle(path, mode);
    checked(api->model_load(str(path), options, &model, &error), TRTMC_OK, &error,
            "C required-output fixture loads");
    return model;
}

static void required_mask_outputs(const char* root, const trtmc_load_options_v1* options) {
    const float pixels[18] = {0}, prior_values[2] = {-1, 2};
    const trtmc_image_input_v1 image = {pixels, sizeof(pixels), 2, 3, 3, TRTMC_IMAGE_FLOAT32};
    const trtmc_point_prompt_v1 point = {{1, 1}, 1};
    const trtmc_box_exemplar_v1 exemplar = {{0, 0, 1, 1}, 1};
    const trtmc_image_points_request_v1 points = {image, &point, 1};
    const trtmc_image_box_request_v1 box = {image, {0, 0, 1, 1}};
    const trtmc_image_prior_mask_request_v1 prior = {
        image, {prior_values, 2, 1, 2}, NULL, 0, 0, {0, 0, 0, 0}};
    const trtmc_image_query_request_v1 text_input = {image, {"object", 6}};
    const trtmc_image_exemplars_request_v1 examples = {image, &exemplar, 1, {"object", 6}};
    const trtmc_perception_image_request_v1 automatic = {image};
    const char* ids[] = {TRTMC_TASK_IMAGE_POINTS_TO_MASKS,
                         TRTMC_TASK_IMAGE_BOX_TO_MASKS,
                         TRTMC_TASK_IMAGE_MASK_TO_MASKS,
                         TRTMC_TASK_IMAGE_TEXT_TO_INSTANCE_MASKS,
                         TRTMC_TASK_IMAGE_BOX_EXEMPLARS_TO_INSTANCE_MASKS,
                         TRTMC_TASK_IMAGE_TO_MASK_PROPOSALS};
    const struct {
        const char* mode;
        int kind;
    } cases[] = {{"missing_point_iou", 0},       {"missing_box_iou", 1},
                 {"missing_prior_iou", 2},       {"missing_instance_confidence", 3},
                 {"missing_instance_boxes", 3},  {"missing_instance_confidence", 4},
                 {"missing_instance_boxes", 4},  {"missing_proposal_metadata", 5},
                 {"missing_proposal_scores", 5}, {"wrong_optional_count", 0}};
    for (size_t iteration = 0; iteration < sizeof(cases) / sizeof(cases[0]) + 6; ++iteration) {
        const int negative = iteration < sizeof(cases) / sizeof(cases[0]);
        const int kind =
            negative ? cases[iteration].kind : (int)(iteration - sizeof(cases) / sizeof(cases[0]));
        const char* mode = negative ? cases[iteration].mode : "empty_contract_masks";
        trtmc_model* model = load_required_mode(root, mode, options);
        const trtmc_api_header* header = table(model, ids[kind]);
        trtmc_result* result = (trtmc_result*)(uintptr_t)1;
        trtmc_error* error = NULL;
        trtmc_status status;
        trtmc_status(TRTMC_CALL * read)(const trtmc_result*, trtmc_masks_view_v1*, trtmc_error**);
#define MASK_CALL(Type, Input)                                                                     \
    do {                                                                                           \
        const Type* task = (const Type*)header;                                                    \
        status = task->run(model, &(Input), NULL, &result, &error);                                \
        read = task->result_view;                                                                  \
    } while (0)
        switch (kind) {
        case 0:
            MASK_CALL(trtmc_image_points_to_masks_api_v1, points);
            break;
        case 1:
            MASK_CALL(trtmc_image_box_to_masks_api_v1, box);
            break;
        case 2:
            MASK_CALL(trtmc_image_mask_to_masks_api_v1, prior);
            break;
        case 3:
            MASK_CALL(trtmc_image_text_to_instance_masks_api_v1, text_input);
            break;
        case 4:
            MASK_CALL(trtmc_image_box_exemplars_to_instance_masks_api_v1, examples);
            break;
        default:
            MASK_CALL(trtmc_image_to_mask_proposals_api_v1, automatic);
            break;
        }
#undef MASK_CALL
        checked(status, negative ? TRTMC_INTERNAL_ERROR : TRTMC_OK, &error,
                "C required mask metadata or valid zero detections");
        if (negative)
            check(result == NULL, "C missing required output clears result with a valid error");
        else {
            trtmc_masks_view_v1 view = {0};
            checked(read(result, &view, &error), TRTMC_OK, &error, "C empty mask result readable");
            check(view.mask_count == 0 && view.iou_count == 0 && view.confidence_count == 0 &&
                      view.box_count == 0 && view.proposal_count == 0,
                  "C all six Task contracts permit N=0");
            api->result_release(result);
        }
        api->model_release(model);
    }
    const char* alternatives[] = {"proposal_iou_only", "proposal_confidence_only",
                                  "proposal_stability_only"};
    for (int i = 0; i < 3; ++i) {
        trtmc_model* model = load_required_mode(root, alternatives[i], options);
        const trtmc_image_to_mask_proposals_api_v1* task =
            (const trtmc_image_to_mask_proposals_api_v1*)table(model,
                                                               TRTMC_TASK_IMAGE_TO_MASK_PROPOSALS);
        trtmc_result* result = NULL;
        trtmc_error* error = NULL;
        trtmc_masks_view_v1 view = {0};
        checked(task->run(model, &automatic, NULL, &result, &error), TRTMC_OK, &error,
                "C automatic proposals accept each genuine typed quality alternative");
        checked(task->result_view(result, &view, &error), TRTMC_OK, &error,
                "C proposal quality result readable");
        check(view.proposal_count == 2 &&
                  view.iou_count + view.confidence_count + view.stability_count == 2,
              "C absent score kinds are not synthesized");
        api->result_release(result);
        api->model_release(model);
    }
}

static trtmc_status TRTMC_CALL simple_pose_crop(void* raw, const trtmc_pose_crop_request_v1* input,
                                                trtmc_pose_crop_response_v1* out) {
    static const float values[12] = {0.25F};
    int* counts = (int*)raw;
    ++counts[input->stage == TRTMC_POSE_CROP_SCORING ? 1 : 0];
    *out = (trtmc_pose_crop_response_v1){0};
    out->rendered = values;
    out->observed = values;
    out->rendered_count = out->observed_count = input->poses.count * 6;
    out->hypothesis_count = input->poses.count;
    out->height = out->width = 1;
    out->channels = 6;
    return TRTMC_OK;
}
static void required_pose_outputs(const char* root, const trtmc_load_options_v1* options,
                                  trtmc_model* complete) {
    float poses[32] = {0};
    for (size_t n = 0; n < 2; ++n)
        for (size_t axis = 0; axis < 4; ++axis)
            poses[n * 16 + axis * 5] = 1;
    int counts[2] = {0};
    trtmc_pose_refinement_request_v1 input = {{poses, 32, 2}, 1.0F, counts, simple_pose_crop};
    const trtmc_pose_hypotheses_crops_to_refined_poses_api_v1* task =
        (const trtmc_pose_hypotheses_crops_to_refined_poses_api_v1*)table(
            complete, TRTMC_TASK_POSE_HYPOTHESES_CROPS_TO_REFINED_POSES);
    trtmc_model* incomplete = load_required_mode(root, "example_pose_missing_score", options);
    trtmc_result* result = (trtmc_result*)(uintptr_t)1;
    trtmc_error* error = NULL;
    checked(task->run(incomplete, &input, NULL, &result, &error), TRTMC_INTERNAL_ERROR, &error,
            "C multiple returned poses require scores even with a legal selection");
    check(!result && counts[0] == 2 && counts[1] == 1,
          "C rejects missing output after actual family scoring, not by reading Config");
    api->model_release(incomplete);
    input.candidates = (trtmc_pose_matrices_v1){poses, 16, 1};
    counts[0] = counts[1] = 0;
    trtmc_config_entry_v1 option = {0};
    option.name = str("score_hypotheses");
    option.value.kind = TRTMC_CONFIG_BOOL;
    option.value.as.boolean = 0;
    const trtmc_config_view_v1 config = {&option, 1};
    checked(task->run(complete, &input, &config, &result, &error), TRTMC_OK, &error,
            "C single unscored pose is allowed");
    trtmc_refined_poses_view_v1 view = {0};
    checked(task->result_view(result, &view, &error), TRTMC_OK, &error, "C unscored pose readable");
    check(view.refined_poses.count == 1 && view.score_count == 0 && view.best_index == 0 &&
              counts[0] == 2 && counts[1] == 0,
          "C unscored single pose does not force or repeat scoring");
    api->result_release(result);
    result = NULL;
    input.candidates = (trtmc_pose_matrices_v1){poses, 32, 2};
    checked(task->run(complete, &input, &config, &result, &error), TRTMC_INVALID_CONFIG, &error,
            "C multi-pose unscored request remains rejected by family policy");
    check(!result, "C invalid family Config produces no partial pose result");
}

static void image_tasks(trtmc_model* model, trtmc_model* restricted, trtmc_result** retained) {
    const trtmc_image_to_semantic_segmentation_api_v1* semantic =
        (const trtmc_image_to_semantic_segmentation_api_v1*)table(
            model, TRTMC_TASK_IMAGE_TO_SEMANTIC_SEGMENTATION);
    const trtmc_image_points_to_masks_api_v1* points =
        (const trtmc_image_points_to_masks_api_v1*)table(model, TRTMC_TASK_IMAGE_POINTS_TO_MASKS);
    const trtmc_image_box_to_masks_api_v1* boxes =
        (const trtmc_image_box_to_masks_api_v1*)table(model, TRTMC_TASK_IMAGE_BOX_TO_MASKS);
    const trtmc_image_mask_to_masks_api_v1* prior =
        (const trtmc_image_mask_to_masks_api_v1*)table(model, TRTMC_TASK_IMAGE_MASK_TO_MASKS);
    const trtmc_image_to_mask_proposals_api_v1* proposal =
        (const trtmc_image_to_mask_proposals_api_v1*)table(model,
                                                           TRTMC_TASK_IMAGE_TO_MASK_PROPOSALS);
    const trtmc_image_text_to_instance_masks_api_v1* text_masks =
        (const trtmc_image_text_to_instance_masks_api_v1*)table(
            model, TRTMC_TASK_IMAGE_TEXT_TO_INSTANCE_MASKS);
    const trtmc_image_box_exemplars_to_instance_masks_api_v1* exemplars =
        (const trtmc_image_box_exemplars_to_instance_masks_api_v1*)table(
            model, TRTMC_TASK_IMAGE_BOX_EXEMPLARS_TO_INSTANCE_MASKS);
    const trtmc_stereo_images_to_disparity_api_v1* stereo =
        (const trtmc_stereo_images_to_disparity_api_v1*)table(
            model, TRTMC_TASK_STEREO_IMAGES_TO_DISPARITY);
    const trtmc_image_to_metric_geometry_api_v1* geometry =
        (const trtmc_image_to_metric_geometry_api_v1*)table(model,
                                                            TRTMC_TASK_IMAGE_TO_METRIC_GEOMETRY);
    const trtmc_image_text_to_boxes_api_v1* grounded_boxes =
        (const trtmc_image_text_to_boxes_api_v1*)table(model, TRTMC_TASK_IMAGE_TEXT_TO_BOXES);
    const trtmc_image_text_to_points_api_v1* grounded_points =
        (const trtmc_image_text_to_points_api_v1*)table(model, TRTMC_TASK_IMAGE_TEXT_TO_POINTS);
    const trtmc_rgbd_mesh_mask_to_object_pose_api_v1* rgbd =
        (const trtmc_rgbd_mesh_mask_to_object_pose_api_v1*)table(
            model, TRTMC_TASK_RGBD_MESH_MASK_TO_OBJECT_POSE);
    float pixels[18], right[18];
    size_t i;
    trtmc_image_input_v1 image;
    trtmc_error* error = NULL;
    trtmc_result* result = NULL;
    trtmc_semantic_segmentation_view_v1 s = {0};
    trtmc_masks_view_v1 m = {0};
    trtmc_disparity_view_v1 d = {0};
    trtmc_metric_geometry_view_v1 g = {0};
    trtmc_grounded_boxes_view_v1 b = {0};
    trtmc_grounded_points_view_v1 p = {0};
    trtmc_object_pose_view_v1 object_pose = {0};
    trtmc_perception_image_request_v1 single;
    const trtmc_point_prompt_v1 prompts[] = {{{1.5F, 0.5F}, 0}, {{2.5F, 1.5F}, 1}};
    trtmc_image_points_request_v1 point_request;
    trtmc_image_box_request_v1 box_request;
    const float logits[] = {-10, 5};
    trtmc_image_prior_mask_request_v1 prior_request = {0};
    trtmc_image_query_request_v1 query;
    const trtmc_box_exemplar_v1 examples[] = {{{0, 0, 1, 1}, 0}, {{0.25F, 0.5F, 2.75F, 1.75F}, 1}};
    trtmc_image_exemplars_request_v1 exemplar_request;
    trtmc_stereo_images_request_v1 stereo_request;
    const float vertices[] = {0.25F, 0, 0, 1, 0, 0, 0, 1, 0}, depth[] = {0, 2, 2, 2, 2, 2};
    const uint32_t indices[] = {0, 1, 2};
    const uint8_t mask[] = {0, 1, 1, 1, 1, 1};
    trtmc_rgbd_mesh_mask_request_v1 mesh_request = {0};
    if (!semantic || !points || !boxes || !prior || !proposal || !text_masks || !exemplars ||
        !stereo || !geometry || !grounded_boxes || !grounded_points || !rgbd)
        return;
    for (i = 0; i < 18; ++i) {
        pixels[i] = 0.75F;
        right[i] = 0.25F;
    }
    image = (trtmc_image_input_v1){pixels, sizeof(pixels), 2, 3, 3, TRTMC_IMAGE_FLOAT32};
    single.image = image;
    point_request = (trtmc_image_points_request_v1){image, prompts, 2};
    box_request = (trtmc_image_box_request_v1){image, {0.25F, 0.5F, 2.75F, 1.75F}};
    prior_request.image = image;
    prior_request.prior_logits = (trtmc_f32_matrix_view_v1){logits, 2, 1, 2};
    query = (trtmc_image_query_request_v1){image, {"target", 6}};
    exemplar_request = (trtmc_image_exemplars_request_v1){image, examples, 2, {"target", 6}};
    stereo_request = (trtmc_stereo_images_request_v1){
        image, {right, sizeof(right), 2, 3, 3, TRTMC_IMAGE_FLOAT32}};
#define CALL(Table, Request, View)                                                                 \
    do {                                                                                           \
        checked((Table)->run(model, &(Request), NULL, &result, &error), TRTMC_OK, &error,          \
                "C perception run");                                                               \
        checked((Table)->result_view(result, &(View), &error), TRTMC_OK, &error,                   \
                "C perception view");                                                              \
    } while (0)
#define RELEASE()                                                                                  \
    do {                                                                                           \
        api->result_release(result);                                                               \
        result = NULL;                                                                             \
    } while (0)
    CALL(semantic, single, s);
    check(s.class_ids.size == 2 && s.class_ids.data[1] == 5 && s.labels[0] == 255 &&
              s.score_kind == TRTMC_SCORE_LOGIT,
          "C semantic labels and logit kind");
    checked(points->result_view(result, &m, &error), TRTMC_INVALID_ARGUMENT, &error,
            "semantic labels cannot be read as instance masks");
    RELEASE();
    CALL(points, point_request, m);
    check(m.kind == TRTMC_MASK_LOGITS && m.low_res_logits[0] == -7 && m.predicted_iou[0] == 2,
          "C point polarity and low-res logits");
    RELEASE();
    CALL(boxes, box_request, m);
    check(m.boxes[0].x_max == 2.75F, "C original-image XYXY box");
    RELEASE();
    CALL(prior, prior_request, m);
    check(m.low_res_logits[0] == -10 && m.low_res_height == 1,
          "C prior logits are not an inpainting mask");
    RELEASE();
    CALL(proposal, single, m);
    check(m.proposal_count == 2 && m.proposals[0].seed_point_count == 2 && m.proposals[0].area == 3,
          "C proposal metadata");
    *retained = result;
    result = NULL;
    CALL(text_masks, query, m);
    check(m.iou_count == 0 && m.confidence_count == 2 && m.kind == TRTMC_MASK_BINARY,
          "C detection confidence differs from IoU");
    RELEASE();
    CALL(exemplars, exemplar_request, m);
    check(m.boxes[0].x_min == 0.25F && m.confidence[0] == 0.9F,
          "C exemplar polarity and text refinement");
    RELEASE();
    CALL(stereo, stereo_request, d);
    check(d.disparity.data[0] == 0.5F && d.disparity.rows == 2 && d.disparity.columns == 3,
          "C ordered stereo pair");
    RELEASE();
    CALL(geometry, single, g);
    check(!g.valid[0] && isinf(g.depth[0]) && g.depth[1] == 2 && g.normalized_intrinsics[2] == 0.5F,
          "C metric geometry preserves invalid infinity");
    RELEASE();
    CALL(grounded_boxes, query, b);
    check(b.count == 1 && !b.boxes[0].has_confidence && same(b.boxes[0].label, "target"),
          "C grounding does not invent confidence");
    RELEASE();
    CALL(grounded_points, query, p);
    check(p.points[0].point.x == 2.5F && p.points[0].has_confidence,
          "C grounded point coordinates");
    RELEASE();
    mesh_request.rgb = image;
    mesh_request.depth_meters = (trtmc_f32_matrix_view_v1){depth, 6, 2, 3};
    mesh_request.object_mask = mask;
    mesh_request.mask_count = 6;
    mesh_request.pixel_intrinsics[0] = 10;
    mesh_request.pixel_intrinsics[4] = 10;
    mesh_request.pixel_intrinsics[8] = 1;
    mesh_request.mesh.vertices = (trtmc_f32_matrix_view_v1){vertices, 9, 3, 3};
    mesh_request.mesh.triangles = indices;
    mesh_request.mesh.triangle_index_count = 3;
    CALL(rgbd, mesh_request, object_pose);
    check(object_pose.object_to_camera[3] == 0.25F && object_pose.object_to_camera[11] == 2,
          "C RGBD original object frame and depth units");
    RELEASE();
    checked(points->run(restricted, &point_request, NULL, &result, &error), TRTMC_UNSUPPORTED,
            &error, "same C table cannot bypass loaded model support");
    point_request.point_count = 0;
    checked(points->run(model, &point_request, NULL, &result, &error), TRTMC_INVALID_ARGUMENT,
            &error, "missing required point input");
    prior_request.prior_logits.count = 1;
    checked(prior->run(model, &prior_request, NULL, &result, &error), TRTMC_INVALID_ARGUMENT,
            &error, "invalid decoder prior shape");
    mesh_request.mask_count = 1;
    checked(rgbd->run(model, &mesh_request, NULL, &result, &error), TRTMC_INVALID_ARGUMENT, &error,
            "unaligned object mask rejected");
#undef CALL
#undef RELEASE
}

struct CropContext;
struct CropOwner {
    struct CropContext* context;
    float rendered[12], observed[12];
    char message[32];
};
struct CropContext {
    trtmc_model* model;
    const trtmc_pose_hypotheses_crops_to_refined_poses_api_v1* task;
    const trtmc_pose_refinement_request_v1* request;
    int calls, releases, mode, saw_busy;
    struct CropOwner* malformed;
    float borrowed_rendered[3][12], borrowed_observed[3][12];
};
static void release_crop(void* pointer) {
    struct CropOwner* owner = (struct CropOwner*)pointer;
    if (owner) {
        ++owner->context->releases;
        memset(owner, 0, sizeof(*owner));
        free(owner);
    }
}
static trtmc_status crop(void* pointer, const trtmc_pose_crop_request_v1* input,
                         trtmc_pose_crop_response_v1* out) {
    struct CropContext* context = (struct CropContext*)pointer;
    struct CropOwner* owner = NULL;
    size_t i;
    float* rendered;
    float* observed;
    memset(out, 0, sizeof(*out));
    ++context->calls;
    if (input->poses.count != 2 || context->calls > 3)
        return TRTMC_INVALID_ARGUMENT;
    if (context->mode == 5) {
        rendered = context->borrowed_rendered[context->calls - 1];
        observed = context->borrowed_observed[context->calls - 1];
    } else {
        owner = (struct CropOwner*)calloc(1, sizeof(*owner));
        if (!owner)
            return TRTMC_OUT_OF_MEMORY;
        owner->context = context;
        rendered = owner->rendered;
        observed = owner->observed;
        out->owner = owner;
        out->release = release_crop;
    }
    for (i = 0; i < 12; ++i) {
        rendered[i] = (float)(input->iteration + 1) * 0.25F;
        observed[i] = 0.25F;
    }
    if (input->stage == TRTMC_POSE_CROP_SCORING) {
        observed[0] = 0.25F;
        observed[6] = 0.5F;
        check(input->iteration == 2, "C final scoring iteration");
    } else
        check(input->iteration == (uint64_t)(context->calls - 1), "C refinement iteration order");
    out->rendered = rendered;
    out->rendered_count = 12;
    out->observed = observed;
    out->observed_count = 12;
    out->hypothesis_count = 2;
    out->height = 1;
    out->width = 1;
    out->channels = 6;
    if (!context->saw_busy) {
        trtmc_result* reentry = NULL;
        trtmc_error* error = NULL;
        trtmc_model_info_v1 metadata = {0};
        checked(api->model_info(context->model, &metadata, &error), TRTMC_OK, &error,
                "C callback can query metadata");
        checked(context->task->run(context->model, context->request, NULL, &reentry, &error),
                TRTMC_BUSY, &error, "C callback reentry returns BUSY");
        check(reentry == NULL, "reentry leaves no partial result");
        context->saw_busy = 1;
    }
    if (context->mode == 1 && context->calls == 2) {
        memcpy(owner->message, "owned\0failure", 13);
        out->error_message = (trtmc_string_view){owner->message, 13};
        return TRTMC_INVALID_CONFIG;
    }
    if (context->mode == 2)
        out->rendered_count = 11;
    if (context->mode == 3)
        return TRTMC_END;
    if (context->mode == 4) {
        context->malformed = owner;
        out->release = NULL;
    }
    if (context->mode == 6) {
        out->height = UINT32_MAX;
        out->width = UINT32_MAX;
    }
    if (context->mode == 7) {
        context->malformed = owner;
        out->owner = NULL;
    }
    return TRTMC_OK;
}
static void callbacks(trtmc_model* model) {
    const trtmc_pose_hypotheses_crops_to_refined_poses_api_v1* task =
        (const trtmc_pose_hypotheses_crops_to_refined_poses_api_v1*)table(
            model, TRTMC_TASK_POSE_HYPOTHESES_CROPS_TO_REFINED_POSES);
    float poses[32] = {0};
    size_t n, i;
    int mode;
    trtmc_pose_refinement_request_v1 request = {0};
    trtmc_refined_poses_view_v1 view = {0};
    trtmc_result* result = NULL;
    trtmc_error* error = NULL;
    if (!task)
        return;
    for (n = 0; n < 2; ++n)
        for (i = 0; i < 4; ++i)
            poses[n * 16 + i * 5] = 1;
    request.candidates = (trtmc_pose_matrices_v1){poses, 32, 2};
    request.mesh_diameter_meters = 2;
    request.provide_crops = crop;
    for (mode = 0; mode <= 7; ++mode) {
        struct CropContext context = {0};
        trtmc_status actual, expected;
        context.model = model;
        context.task = task;
        context.request = &request;
        context.mode = mode;
        request.context = &context;
        actual = task->run(model, &request, NULL, &result, &error);
        expected = (mode == 0 || mode == 5)
                       ? TRTMC_OK
                       : (mode == 1 ? TRTMC_INVALID_CONFIG
                                    : (mode == 3 ? TRTMC_INTERNAL_ERROR : TRTMC_INVALID_ARGUMENT));
        check(actual == expected, "C crop success/failure status");
        if (mode == 1)
            check(error && api->error_message(error).size == 13 &&
                      memcmp(api->error_message(error).data, "owned\0failure", 13) == 0,
                  "callback message is copied before lease release and keeps explicit length");
        api->error_release(error);
        error = NULL;
        if (actual == TRTMC_OK) {
            checked(task->result_view(result, &view, &error), TRTMC_OK, &error,
                    "C refined pose result readable");
            check(context.calls == 3 && view.refined_poses.values[3] == 1.5F &&
                      view.best_index == 1 && view.scores[1] == 2,
                  "C callback crops remain live through family consumption");
        } else
            check(result == NULL, "callback failure has no partial pose result");
        if (mode == 4 || mode == 7) {
            check(context.releases == 0 && context.malformed != NULL,
                  "malformed owner pair does not transfer an unknown allocator");
            release_crop(context.malformed); /* Invalid provider output stays provider-owned. */
        } else
            check(context.releases == (mode == 5 ? 0 : context.calls),
                  "valid crop lease releases exactly once on every path");
        api->result_release(result);
        result = NULL;
    }
    request.provide_crops = NULL;
    checked(task->run(model, &request, NULL, &result, &error), TRTMC_INVALID_ARGUMENT, &error,
            "C crop callback is required");
}

int main(int argc, char** argv) {
    char full[4096], restricted_path[4096];
    trtmc_load_options_v1 options = {0};
    trtmc_model* model = NULL;
    trtmc_model* restricted = NULL;
    trtmc_error* error = NULL;
    trtmc_result* retained = NULL;
    trtmc_masks_view_v1 view = {0};
    const trtmc_image_to_mask_proposals_api_v1* proposal = NULL;
    if (argc != 2 || strlen(argv[1]) + 40 >= sizeof(full) || trtmc_get_api(1, 0, &api) != TRTMC_OK)
        return 2;
    snprintf(full, sizeof(full), "%s/perception-c-all.bundle", argv[1]);
    snprintf(restricted_path, sizeof(restricted_path), "%s/perception-c-pose.bundle", argv[1]);
    bundle(full, "all");
    bundle(restricted_path, "pose_only");
    options.struct_size = sizeof(options);
    options.runtime_root = str(argv[1]);
    checked(api->model_load(str(full), &options, &model, &error), TRTMC_OK, &error,
            "C perception model load");
    checked(api->model_load(str(restricted_path), &options, &restricted, &error), TRTMC_OK, &error,
            "C restricted perception model load");
    if (model && restricted) {
        char missing_iou_path[4096];
        trtmc_model* incomplete = NULL;
        const float image_values[3] = {0};
        const trtmc_point_prompt_v1 point = {{0, 0}, 1};
        const trtmc_image_points_request_v1 input = {
            {image_values, sizeof(image_values), 1, 1, 3, TRTMC_IMAGE_FLOAT32}, &point, 1};
        trtmc_result* failed = (trtmc_result*)(uintptr_t)1;
        snprintf(missing_iou_path, sizeof(missing_iou_path), "%s/perception-c-missing-iou.bundle",
                 argv[1]);
        bundle(missing_iou_path, "missing_point_iou");
        checked(api->model_load(str(missing_iou_path), &options, &incomplete, &error), TRTMC_OK,
                &error, "C missing-quality fixture loads");
        const trtmc_image_points_to_masks_api_v1* points =
            (const trtmc_image_points_to_masks_api_v1*)table(incomplete,
                                                             TRTMC_TASK_IMAGE_POINTS_TO_MASKS);
        checked(points->run(incomplete, &input, NULL, &failed, &error), TRTMC_INTERNAL_ERROR,
                &error, "C nonempty point masks require quality scores");
        check(failed == NULL, "C required-output error clears the result and owns the error");
        api->model_release(incomplete);
        boxes_batch(argv[1], &options, model);
        required_mask_outputs(argv[1], &options);
        required_pose_outputs(argv[1], &options, model);
        image_tasks(model, restricted, &retained);
        callbacks(model);
        proposal = (const trtmc_image_to_mask_proposals_api_v1*)table(
            model, TRTMC_TASK_IMAGE_TO_MASK_PROPOSALS);
    }
    api->model_release(model);
    api->model_release(restricted);
    if (retained && proposal) {
        checked(proposal->result_view(retained, &view, &error), TRTMC_OK, &error,
                "C proposal snapshot outlives model");
        check(view.proposals[0].seed_points[0].x == 0.5F, "nested proposal metadata remains owned");
    }
    api->result_release(retained);
    fprintf(stderr, "%s\n", failures ? "SOME FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}

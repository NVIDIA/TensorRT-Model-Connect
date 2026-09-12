/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TRTMC_VIDEO_H
#define TRTMC_VIDEO_H

#include "trtmc/audio.h"
#include "trtmc/image.h"
#include "trtmc/matrix.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Each image is a frame of one clip, not an independent model request.
 * Empty timestamps mean physical timing is unspecified. Otherwise timestamps
 * are finite nondecreasing seconds, one per frame, in the request/result clock.
 * Shared code never invents FPS, decodes files or resamples media. */
typedef struct {
    const trtmc_image_input_v1* frames;
    uint64_t frame_count;
    trtmc_f64_view timestamps_seconds;
} trtmc_video_view_v1;
typedef struct {
    const trtmc_image_result_view_v1* frames;
    /* Zero frames is an explicit non-output participant completion. It has
     * no timestamps or context prefix and must not be rendered as a video. */
    uint64_t frame_count;
    trtmc_f64_view timestamps_seconds;
    /* Context occupies [0,prefix). Future Tasks guarantee a nonempty predicted
     * suffix [prefix,frame_count); context must not be reported as prediction. */
    uint64_t conditioned_prefix_frames;
    double setup_ms;
    double inference_ms;
} trtmc_video_result_view_v1;
typedef struct {
    const float* values; /* FHW, zero=conditioning, one=generation; soft masks are family-owned. */
    uint64_t count;
    uint64_t frames, height, width;
} trtmc_video_mask_view_v1;
enum { TRTMC_VIDEO_ANCHOR_IMAGE = 1, TRTMC_VIDEO_ANCHOR_CLIP = 2 };
typedef struct {
    uint32_t kind;
    union {
        trtmc_image_input_v1 image;
        trtmc_video_view_v1 clip;
    } content;
    uint64_t output_start_frame;
    uint32_t has_strength;
    double strength;
} trtmc_timed_video_anchor_v1;
typedef struct {
    trtmc_f32_matrix_view_v1 camera_to_world; /* [frame,16], row-major 4x4. */
    trtmc_f64_view timestamps_seconds;
    trtmc_string_view coordinate_convention;
    trtmc_string_view translation_units;
} trtmc_camera_trajectory_view_v1;
typedef struct {
    uint64_t begin, end;
} trtmc_action_frame_span_v1; /* [begin,end). */
typedef struct {
    trtmc_string_view domain;
    trtmc_strings_view component_names; /* Empty or one per matrix column. */
    trtmc_strings_view units;           /* Empty means unspecified, not assumed SI units. */
    trtmc_string_view coordinate_frame;
    trtmc_string_view normalization;
} trtmc_action_schema_view_v1;
typedef struct {
    trtmc_f32_matrix_view_v1 values; /* [step,raw action dimension]. */
    trtmc_action_schema_view_v1 schema;
    trtmc_f64_view timestamps_seconds;
    const trtmc_action_frame_span_v1* frame_spans;
    uint64_t frame_span_count; /* Empty means unspecified; otherwise one per step. */
} trtmc_action_sequence_view_v1;
typedef struct {
    trtmc_action_schema_view_v1 schema;
    uint64_t dimensions;
} trtmc_action_output_spec_v1;
typedef struct {
    trtmc_action_sequence_view_v1 actions;
    trtmc_video_result_view_v1 video;
} trtmc_action_video_result_view_v1;
typedef struct {
    trtmc_video_result_view_v1 video; /* Synchronized output requires actual timestamps. */
    trtmc_audio_result_view_v1 audio; /* Includes the actual sample rate and channels. */
    double audio_start_seconds;       /* Same clock as video timestamps. */
} trtmc_audio_video_result_view_v1;
typedef struct {
    trtmc_video_view_v1 video;
    uint32_t has_soundtrack;
    trtmc_audio_view_v1 soundtrack;
    uint32_t has_audio_start;
    double audio_start_seconds;
} trtmc_video_reference_v1;
enum {
    TRTMC_VIDEO_REFERENCE_IMAGE = 1,
    TRTMC_VIDEO_REFERENCE_CLIP = 2,
    TRTMC_VIDEO_REFERENCE_AUDIO = 3
};
typedef struct {
    uint32_t kind;
    union {
        trtmc_image_input_v1 image;
        trtmc_video_reference_v1 clip;
        trtmc_audio_view_v1 audio;
    } content;
} trtmc_video_reference_item_v1;

/* Intrinsics: [1 or F,9] row-major 3x3 matrices in initial-image pixels.
 * Action DSL dialect absence uses only the loaded family's declared default.
 * Optional prompt absence is passed through for family-owned resolution.
 * References are semantic references, not implicit source edits or keyframes.
 * Action frame spans index observations for inverse dynamics and the returned
 * rollout for forward/joint dynamics. No physical rate is inferred from them.
 * Masks express conditioning/generation regions, not pixel-exact preservation. */

typedef struct {
    trtmc_string_view prompt;
    /* Optional host float32 in family-defined packed/CTHW latent layout.
     * Empty selects family initialization; family validates layout and count. */
    trtmc_f32_view initial_latents;
} trtmc_text_to_video_request_v1;
typedef struct {
    trtmc_image_input_v1 initial_image;
    trtmc_string_view prompt;
} trtmc_initial_image_text_to_video_request_v1;
typedef struct {
    trtmc_image_input_v1 first_frame;
    trtmc_image_input_v1 last_frame;
    trtmc_string_view prompt;
} trtmc_boundary_frames_text_to_video_request_v1;
typedef struct {
    const trtmc_timed_video_anchor_v1* anchors;
    uint64_t anchor_count;
    trtmc_string_view prompt;
} trtmc_timed_frames_text_to_video_request_v1;
typedef struct {
    trtmc_video_view_v1 source;
    trtmc_string_view prompt;
} trtmc_video_text_to_video_edit_request_v1;
typedef struct {
    trtmc_video_view_v1 source;
    trtmc_video_mask_view_v1 mask;
    trtmc_string_view prompt;
} trtmc_masked_video_text_to_video_request_v1;
typedef struct {
    trtmc_video_view_v1 source;
    trtmc_video_mask_view_v1 mask;
    const trtmc_image_input_v1* references;
    uint64_t reference_count;
    trtmc_string_view prompt;
} trtmc_masked_video_reference_images_text_to_video_request_v1;
typedef struct {
    trtmc_image_input_v1 initial_image;
    trtmc_string_view prompt;
    trtmc_string_view action_dsl;
    trtmc_f32_matrix_view_v1 intrinsics;
    uint32_t has_dialect;
    trtmc_string_view dialect;
    /* The initial image still conditions/overwrites the family's latent frame. */
    trtmc_f32_view initial_latents;
} trtmc_image_text_action_to_video_request_v1;
typedef struct {
    trtmc_image_input_v1 initial_image;
    trtmc_string_view prompt;
    trtmc_camera_trajectory_view_v1 camera;
    trtmc_f32_matrix_view_v1 intrinsics;
    trtmc_f32_view initial_latents;
} trtmc_image_text_camera_trajectory_to_video_request_v1;
typedef struct {
    trtmc_video_view_v1 history;
    trtmc_string_view prompt;
} trtmc_video_text_to_future_video_request_v1;
typedef struct {
    trtmc_image_input_v1 observation;
    trtmc_action_sequence_view_v1 actions;
    uint32_t has_prompt;
    trtmc_string_view prompt;
} trtmc_image_action_to_future_video_request_v1;
typedef struct {
    trtmc_video_view_v1 history;
    trtmc_action_sequence_view_v1 actions;
    uint32_t has_prompt;
    trtmc_string_view prompt;
} trtmc_video_action_to_future_video_request_v1;
typedef struct {
    trtmc_video_view_v1 observations;
    trtmc_action_output_spec_v1 action_spec;
    uint32_t has_prompt;
    trtmc_string_view prompt;
} trtmc_video_to_action_sequence_request_v1;
typedef struct {
    trtmc_image_input_v1 observation;
    trtmc_action_output_spec_v1 action_spec;
    uint32_t has_prompt;
    trtmc_string_view prompt;
} trtmc_image_to_action_and_video_request_v1;
typedef struct {
    trtmc_video_view_v1 history;
    trtmc_action_output_spec_v1 action_spec;
    uint32_t has_prompt;
    trtmc_string_view prompt;
} trtmc_video_to_action_and_video_request_v1;
typedef struct {
    trtmc_string_view prompt;
} trtmc_text_to_audio_video_request_v1;
typedef struct {
    trtmc_image_input_v1 initial_image;
    trtmc_string_view prompt;
} trtmc_initial_image_text_to_audio_video_request_v1;
typedef struct {
    trtmc_image_input_v1 last_image;
    trtmc_string_view prompt;
} trtmc_last_image_text_to_audio_video_request_v1;
typedef struct {
    trtmc_image_input_v1 first_frame;
    trtmc_image_input_v1 last_frame;
    trtmc_string_view prompt;
} trtmc_boundary_frames_text_to_audio_video_request_v1;
typedef struct {
    const trtmc_video_reference_item_v1* references;
    uint64_t reference_count;
    trtmc_string_view prompt;
} trtmc_references_text_to_audio_video_request_v1;

#define TRTMC_TASK_TEXT_TO_VIDEO "text_to_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_to_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_text_to_video_api_v1;

#define TRTMC_TASK_INITIAL_IMAGE_TEXT_TO_VIDEO "initial_image_text_to_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_initial_image_text_to_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_initial_image_text_to_video_api_v1;

#define TRTMC_TASK_BOUNDARY_FRAMES_TEXT_TO_VIDEO "boundary_frames_text_to_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_boundary_frames_text_to_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_boundary_frames_text_to_video_api_v1;

#define TRTMC_TASK_TIMED_FRAMES_TEXT_TO_VIDEO "timed_frames_text_to_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_timed_frames_text_to_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_timed_frames_text_to_video_api_v1;

#define TRTMC_TASK_VIDEO_TEXT_TO_VIDEO_EDIT "video_text_to_video_edit"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_video_text_to_video_edit_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_video_text_to_video_edit_api_v1;

#define TRTMC_TASK_MASKED_VIDEO_TEXT_TO_VIDEO "masked_video_text_to_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_masked_video_text_to_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_masked_video_text_to_video_api_v1;

#define TRTMC_TASK_MASKED_VIDEO_REFERENCE_IMAGES_TEXT_TO_VIDEO                                     \
    "masked_video_reference_images_text_to_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(
        trtmc_model*, const trtmc_masked_video_reference_images_text_to_video_request_v1*,
        const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_masked_video_reference_images_text_to_video_api_v1;

#define TRTMC_TASK_IMAGE_TEXT_ACTION_TO_VIDEO "image_text_action_to_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_image_text_action_to_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_image_text_action_to_video_api_v1;

#define TRTMC_TASK_IMAGE_TEXT_CAMERA_TRAJECTORY_TO_VIDEO "image_text_camera_trajectory_to_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_image_text_camera_trajectory_to_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_image_text_camera_trajectory_to_video_api_v1;

#define TRTMC_TASK_VIDEO_TEXT_TO_FUTURE_VIDEO "video_text_to_future_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_video_text_to_future_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_video_text_to_future_video_api_v1;

#define TRTMC_TASK_IMAGE_ACTION_TO_FUTURE_VIDEO "image_action_to_future_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_image_action_to_future_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_image_action_to_future_video_api_v1;

#define TRTMC_TASK_VIDEO_ACTION_TO_FUTURE_VIDEO "video_action_to_future_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_video_action_to_future_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_video_action_to_future_video_api_v1;

#define TRTMC_TASK_VIDEO_TO_ACTION_SEQUENCE "video_to_action_sequence"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_video_to_action_sequence_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_action_sequence_view_v1*,
                                          trtmc_error**);
} trtmc_video_to_action_sequence_api_v1;

#define TRTMC_TASK_IMAGE_TO_ACTION_AND_VIDEO "image_to_action_and_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_image_to_action_and_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_action_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_image_to_action_and_video_api_v1;

#define TRTMC_TASK_VIDEO_TO_ACTION_AND_VIDEO "video_to_action_and_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_video_to_action_and_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_action_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_video_to_action_and_video_api_v1;

#define TRTMC_TASK_TEXT_TO_AUDIO_VIDEO "text_to_audio_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_text_to_audio_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_audio_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_text_to_audio_video_api_v1;

#define TRTMC_TASK_INITIAL_IMAGE_TEXT_TO_AUDIO_VIDEO "initial_image_text_to_audio_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_initial_image_text_to_audio_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_audio_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_initial_image_text_to_audio_video_api_v1;

#define TRTMC_TASK_LAST_IMAGE_TEXT_TO_AUDIO_VIDEO "last_image_text_to_audio_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_last_image_text_to_audio_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_audio_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_last_image_text_to_audio_video_api_v1;

#define TRTMC_TASK_BOUNDARY_FRAMES_TEXT_TO_AUDIO_VIDEO "boundary_frames_text_to_audio_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_boundary_frames_text_to_audio_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_audio_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_boundary_frames_text_to_audio_video_api_v1;

#define TRTMC_TASK_REFERENCES_TEXT_TO_AUDIO_VIDEO "references_text_to_audio_video"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_references_text_to_audio_video_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_audio_video_result_view_v1*,
                                          trtmc_error**);
} trtmc_references_text_to_audio_video_api_v1;

#define TRTMC_TASK_BATCH_TEXT_TO_VIDEO "batch_text_to_video"
typedef struct {
    trtmc_text_to_video_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_text_to_video_item_v1;
typedef struct {
    const trtmc_batch_text_to_video_item_v1* items;
    uint64_t count;
} trtmc_batch_text_to_video_request_v1;
/* One family-native batch; inputs are borrowed until return. Success has N
 * ordered owned results, each retaining its real frame/audio/action lengths.
 * An error returns NULL, never a partial-success array or implicit retry. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_batch_text_to_video_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_video_result_view_v1*, trtmc_error**);
} trtmc_batch_text_to_video_api_v1;
#define TRTMC_TASK_BATCH_INITIAL_IMAGE_TEXT_TO_VIDEO "batch_initial_image_text_to_video"
typedef struct {
    trtmc_initial_image_text_to_video_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_initial_image_text_to_video_item_v1;
typedef struct {
    const trtmc_batch_initial_image_text_to_video_item_v1* items;
    uint64_t count;
} trtmc_batch_initial_image_text_to_video_request_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_initial_image_text_to_video_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_video_result_view_v1*, trtmc_error**);
} trtmc_batch_initial_image_text_to_video_api_v1;

#define TRTMC_TASK_BATCH_VIDEO_TEXT_TO_FUTURE_VIDEO "batch_video_text_to_future_video"
typedef struct {
    trtmc_video_text_to_future_video_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_video_text_to_future_video_item_v1;
typedef struct {
    const trtmc_batch_video_text_to_future_video_item_v1* items;
    uint64_t count;
} trtmc_batch_video_text_to_future_video_request_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_video_text_to_future_video_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_video_result_view_v1*, trtmc_error**);
} trtmc_batch_video_text_to_future_video_api_v1;

#define TRTMC_TASK_BATCH_IMAGE_ACTION_TO_FUTURE_VIDEO "batch_image_action_to_future_video"
typedef struct {
    trtmc_image_action_to_future_video_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_image_action_to_future_video_item_v1;
typedef struct {
    const trtmc_batch_image_action_to_future_video_item_v1* items;
    uint64_t count;
} trtmc_batch_image_action_to_future_video_request_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_image_action_to_future_video_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_video_result_view_v1*, trtmc_error**);
} trtmc_batch_image_action_to_future_video_api_v1;

#define TRTMC_TASK_BATCH_VIDEO_TO_ACTION_SEQUENCE "batch_video_to_action_sequence"
typedef struct {
    trtmc_video_to_action_sequence_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_video_to_action_sequence_item_v1;
typedef struct {
    const trtmc_batch_video_to_action_sequence_item_v1* items;
    uint64_t count;
} trtmc_batch_video_to_action_sequence_request_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_video_to_action_sequence_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_action_sequence_view_v1*, trtmc_error**);
} trtmc_batch_video_to_action_sequence_api_v1;

#define TRTMC_TASK_BATCH_IMAGE_TO_ACTION_AND_VIDEO "batch_image_to_action_and_video"
typedef struct {
    trtmc_image_to_action_and_video_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_image_to_action_and_video_item_v1;
typedef struct {
    const trtmc_batch_image_to_action_and_video_item_v1* items;
    uint64_t count;
} trtmc_batch_image_to_action_and_video_request_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_image_to_action_and_video_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_action_video_result_view_v1*, trtmc_error**);
} trtmc_batch_image_to_action_and_video_api_v1;

#define TRTMC_TASK_BATCH_TEXT_TO_AUDIO_VIDEO "batch_text_to_audio_video"
typedef struct {
    trtmc_text_to_audio_video_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_text_to_audio_video_item_v1;
typedef struct {
    const trtmc_batch_text_to_audio_video_item_v1* items;
    uint64_t count;
} trtmc_batch_text_to_audio_video_request_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_batch_text_to_audio_video_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_audio_video_result_view_v1*, trtmc_error**);
} trtmc_batch_text_to_audio_video_api_v1;

#define TRTMC_TASK_BATCH_INITIAL_IMAGE_TEXT_TO_AUDIO_VIDEO "batch_initial_image_text_to_audio_video"
typedef struct {
    trtmc_initial_image_text_to_audio_video_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_initial_image_text_to_audio_video_item_v1;
typedef struct {
    const trtmc_batch_initial_image_text_to_audio_video_item_v1* items;
    uint64_t count;
} trtmc_batch_initial_image_text_to_audio_video_request_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*,
                                  const trtmc_batch_initial_image_text_to_audio_video_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_audio_video_result_view_v1*, trtmc_error**);
} trtmc_batch_initial_image_text_to_audio_video_api_v1;

#ifdef __cplusplus
}
#endif

#endif /* TRTMC_VIDEO_H */

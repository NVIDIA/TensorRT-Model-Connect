/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef TRTMC_ACTION_H
#define TRTMC_ACTION_H
#include "trtmc/video.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    trtmc_image_input_v1 image;
    const float* state;
    uint64_t state_count;
} trtmc_image_state_observation_v1;
typedef struct {
    trtmc_image_state_observation_v1 observation;
} trtmc_image_state_to_action_chunk_request_v1;
typedef struct {
    trtmc_action_sequence_view_v1 actions;
    uint32_t within_training_bounds;
    double inference_ms;
} trtmc_image_state_action_chunk_view_v1;
typedef struct {
    trtmc_action_sequence_view_v1 action; /* Exactly one [1,component] row. */
    uint32_t within_training_bounds;
    uint32_t started_new_chunk;
    double inference_ms;
} trtmc_action_step_view_v1;
typedef struct trtmc_image_state_action_session trtmc_image_state_action_session;

/* One image and one ordered state vector. Family output actions are already
 * unnormalized: no shared clipping, denormalization, rate or unit inference.
 * The training-range flag is not an assurance of physical execution safety.
 * Inputs borrow host storage through each call. Result views own snapshots
 * until result_release, including after reset/session/model release. */
#define TRTMC_TASK_IMAGE_STATE_TO_ACTION_CHUNK "image_state_to_action_chunk"
/* Independent prediction may run serially while an action queue is alive:
 * it neither consumes nor replaces queued actions. Overlapping/reentrant
 * chunk/queue operations return BUSY; other live sessions remain exclusive. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_image_state_to_action_chunk_request_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*,
                                          trtmc_image_state_action_chunk_view_v1*, trtmc_error**);
} trtmc_image_state_to_action_chunk_api_v1;

#define TRTMC_TASK_IMAGE_STATE_ACTION_QUEUE "image_state_action_queue"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* create)(trtmc_model*, const trtmc_config_view_v1*,
                                     trtmc_image_state_action_session**, trtmc_error**);
    trtmc_status(TRTMC_CALL* act)(trtmc_image_state_action_session*,
                                  const trtmc_image_state_observation_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_action_step_view_v1*,
                                          trtmc_error**);
    trtmc_status(TRTMC_CALL* reset)(trtmc_image_state_action_session*, trtmc_error**);
    /* A live queue reserves model execution except for its independent action
     * chunk Task. Queue act/reset and chunk execution are mutually exclusive;
     * overlapping/reentrant calls return BUSY. The family owns recovery/reset.
     * Release must not race a queue or chunk call. It destroys family state before
     * releasing the model reservation; NULL release is accepted. */
    void(TRTMC_CALL* release)(trtmc_image_state_action_session*);
} trtmc_image_state_action_queue_api_v1;
#ifdef __cplusplus
}
#endif
#endif /* TRTMC_ACTION_H */

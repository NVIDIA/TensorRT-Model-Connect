/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TRTMC_TRACKING_H
#define TRTMC_TRACKING_H
#include "trtmc/perception.h"
#include "trtmc/video.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct trtmc_detected_mask_session trtmc_detected_mask_session;
typedef struct trtmc_text_mask_clip_session trtmc_text_mask_clip_session;
typedef struct trtmc_text_prompt_frame_session trtmc_text_prompt_frame_session;
typedef struct trtmc_image_mask_context trtmc_image_mask_context;
enum { TRTMC_TRACK_HOST = 1, TRTMC_TRACK_CUDA = 2 };
enum { TRTMC_TRACK_UINT8 = 1, TRTMC_TRACK_FLOAT32 = 2 };
typedef struct {
    uint64_t frame_index;
    uint32_t height, width;
    trtmc_i64_view object_ids;
    const void* masks;
    uint64_t mask_byte_size;
    uint32_t memory_kind, element_type, mask_kind;
    int32_t device_ordinal; /* -1 for host; CUDA pointers are NEVER host-readable. */
    const trtmc_pixel_box_v1* boxes;
    uint64_t box_count;
    const float* detection_scores;
    uint64_t detection_score_count;
    const float* tracker_scores;
    uint64_t tracker_score_count;
    trtmc_i64_view class_ids, removed_object_ids, suppressed_object_ids;
} trtmc_track_frame_view_v1;
typedef struct {
    uint64_t frame_index;
    int64_t object_id, class_id;
    float score;
    trtmc_pixel_box_v1 prompt_box;
} trtmc_initial_track_detection_v1;
typedef struct {
    const trtmc_track_frame_view_v1* frames;
    uint64_t frame_count;
    const trtmc_initial_track_detection_v1* initial_detections;
    uint64_t detection_count;
} trtmc_track_clip_view_v1;
/* Device masks are producer-ready at return, matching current native SAM2.
 * Metadata belongs to the result; addresses borrow session memory. The next
 * delegated host/device segment (even one that fails), or release, invalidates
 * older addresses. result_view rejects invalidated descriptors. Metadata reads
 * do not invalidate them. Synchronize caller GPU readers before reuse/close.
 * Release does not race operations; no CUDA headers or shared CUDA wait exist. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* segment_device)(trtmc_detected_mask_session*,
                                             const trtmc_video_view_v1*,
                                             const trtmc_config_view_v1*, trtmc_result**,
                                             trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_track_clip_view_v1*,
                                          trtmc_error**);
} trtmc_detected_device_masks_api_v1;
#define TRTMC_TASK_FRAMES_TO_DETECTED_MASK_TRACKS "frames_to_detected_mask_tracks"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* create)(trtmc_model*, const trtmc_config_view_v1*,
                                     trtmc_detected_mask_session**, trtmc_error**);
    trtmc_status(TRTMC_CALL* segment)(trtmc_detected_mask_session*, const trtmc_video_view_v1*,
                                      const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_track_clip_view_v1*,
                                          trtmc_error**);
    trtmc_status(TRTMC_CALL* get_device_api)(trtmc_detected_mask_session*, uint32_t, uint32_t,
                                             const trtmc_detected_device_masks_api_v1**,
                                             trtmc_error**);
    void(TRTMC_CALL* release)(trtmc_detected_mask_session*);
} trtmc_frames_to_detected_mask_tracks_api_v1;

#define TRTMC_TASK_FRAMES_TEXT_TO_MASK_TRACKS "frames_text_to_mask_tracks"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* create)(trtmc_model*, const trtmc_config_view_v1*,
                                     trtmc_text_mask_clip_session**, trtmc_error**);
    trtmc_status(TRTMC_CALL* segment)(trtmc_text_mask_clip_session*, const trtmc_video_view_v1*,
                                      trtmc_string_view text, const trtmc_config_view_v1*,
                                      trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_track_clip_view_v1*,
                                          trtmc_error**);
    void(TRTMC_CALL* release)(trtmc_text_mask_clip_session*);
} trtmc_frames_text_to_mask_tracks_api_v1;

#define TRTMC_TASK_PROMPT_FRAME_TEXT_TO_MASK_TRACKS "prompt_frame_text_to_mask_tracks"
/* One prompt frame, then one complete borrowed continuation. No reset, reverse
 * propagation or arbitrary append is implied. The prompt result must originate
 * from this session and stays owned/readable after continue_borrowed. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* create)(trtmc_model*, trtmc_string_view text,
                                     const trtmc_config_view_v1*, trtmc_text_prompt_frame_session**,
                                     trtmc_error**);
    trtmc_status(TRTMC_CALL* accept_prompt_frame)(trtmc_text_prompt_frame_session*,
                                                  const trtmc_image_input_v1*, trtmc_result**,
                                                  trtmc_error**);
    trtmc_status(TRTMC_CALL* continue_borrowed)(trtmc_text_prompt_frame_session*,
                                                const trtmc_result* prompt,
                                                const trtmc_video_view_v1* complete_clip,
                                                trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_track_clip_view_v1*,
                                          trtmc_error**);
    void(TRTMC_CALL* release)(trtmc_text_prompt_frame_session*);
} trtmc_prompt_frame_text_to_mask_tracks_api_v1;

typedef struct {
    trtmc_f32_matrix_view_v1 prior_logits;
    const trtmc_point_prompt_v1* points;
    uint64_t point_count;
    uint32_t has_box;
    trtmc_pixel_box_v1 box;
} trtmc_image_prior_prompt_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_image_mask_context*, const trtmc_point_prompt_v1*, uint64_t,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
} trtmc_image_points_editor_api_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_image_mask_context*, const trtmc_pixel_box_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
} trtmc_image_box_editor_api_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_image_mask_context*, const trtmc_image_prior_prompt_v1*,
                                  const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
} trtmc_image_prior_editor_api_v1;
/* Each pointer projects one native typed getter: null means absent.
 * A borrowed table on a different context rechecks that context's getter. */
typedef struct {
    const trtmc_image_points_editor_api_v1* points;
    const trtmc_image_box_editor_api_v1* box;
    const trtmc_image_prior_editor_api_v1* prior;
} trtmc_image_mask_editors_v1;
#define TRTMC_TASK_INTERACTIVE_IMAGE_MASKS "interactive_image_masks"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* create)(trtmc_model*, const trtmc_image_input_v1*,
                                     const trtmc_config_view_v1*, trtmc_image_mask_context**,
                                     trtmc_error**);
    trtmc_status(TRTMC_CALL* get_editors)(trtmc_image_mask_context*, trtmc_image_mask_editors_v1*,
                                          trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_masks_view_v1*, trtmc_error**);
    void(TRTMC_CALL* release)(trtmc_image_mask_context*);
} trtmc_interactive_image_masks_api_v1;

typedef struct trtmc_mask_track_session trtmc_mask_track_session;
typedef struct trtmc_track_propagation trtmc_track_propagation;
enum { TRTMC_POINTS_REPLACE = 1, TRTMC_POINTS_APPEND = 2 };
enum { TRTMC_PROPAGATE_FORWARD = 1, TRTMC_PROPAGATE_BACKWARD = 2 };
typedef struct {
    uint64_t frame_index;
    int64_t object_id;
    const trtmc_point_prompt_v1* points;
    uint64_t point_count;
    uint32_t update;
} trtmc_frame_object_points_v1;
typedef struct {
    uint64_t frame_index;
    int64_t object_id;
    trtmc_pixel_box_v1 box;
    const trtmc_point_prompt_v1* correction_points;
    uint64_t correction_point_count;
} trtmc_frame_object_box_v1;
typedef struct {
    uint64_t frame_index;
    int64_t object_id;
    const uint8_t* mask;
    uint64_t mask_count;
    uint32_t height, width;
} trtmc_frame_object_mask_v1;
typedef struct {
    uint64_t frame_index;
    trtmc_string_view text;
} trtmc_frame_text_v1;
typedef struct {
    uint64_t frame_index;
    trtmc_box_exemplar_v1 exemplar;
} trtmc_frame_box_exemplar_v1;
typedef struct {
    uint64_t start_frame, frame_count;
    uint32_t direction;
} trtmc_propagation_range_v1;
#define TRTMC_TRACK_EDITOR_API(Name, Input)                                                        \
    typedef struct {                                                                               \
        trtmc_api_header header;                                                                   \
        trtmc_status(TRTMC_CALL* run)(trtmc_mask_track_session*, const Input*,                     \
                                      const trtmc_config_view_v1*, trtmc_result**, trtmc_error**); \
    } trtmc_##Name##_track_editor_api_v1;
TRTMC_TRACK_EDITOR_API(points, trtmc_frame_object_points_v1)
TRTMC_TRACK_EDITOR_API(box, trtmc_frame_object_box_v1)
TRTMC_TRACK_EDITOR_API(mask, trtmc_frame_object_mask_v1)
TRTMC_TRACK_EDITOR_API(text, trtmc_frame_text_v1)
TRTMC_TRACK_EDITOR_API(exemplar, trtmc_frame_box_exemplar_v1)
#undef TRTMC_TRACK_EDITOR_API
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* remove)(trtmc_mask_track_session*, int64_t, trtmc_result**,
                                     trtmc_error**);
} trtmc_track_object_removal_api_v1;
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* reset)(trtmc_mask_track_session*, trtmc_error**);
} trtmc_track_reset_api_v1;
typedef struct {
    const trtmc_points_track_editor_api_v1* points;
    const trtmc_box_track_editor_api_v1* box;
    const trtmc_mask_track_editor_api_v1* mask;
    const trtmc_text_track_editor_api_v1* text;
    const trtmc_exemplar_track_editor_api_v1* exemplar;
    const trtmc_track_object_removal_api_v1* objects;
    const trtmc_track_reset_api_v1* reset;
} trtmc_mask_track_editors_v1;
/* next delegates one native frame; END has null result/error. While traversal
 * lives, parent mutation returns BUSY. Cancel is synchronous between next calls.
 * A traversal retains native resources if the parent handle is released first.
 * Results preserve traversal order and original clip frame_index; never sorted. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* get_editors)(trtmc_mask_track_session*, trtmc_mask_track_editors_v1*,
                                          trtmc_error**);
    trtmc_status(TRTMC_CALL* start_propagation)(trtmc_mask_track_session*,
                                                const trtmc_propagation_range_v1*,
                                                trtmc_track_propagation**, trtmc_error**);
    trtmc_status(TRTMC_CALL* next)(trtmc_track_propagation*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* cancel)(trtmc_track_propagation*, trtmc_error**);
    void(TRTMC_CALL* release_propagation)(trtmc_track_propagation*);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_track_clip_view_v1*,
                                          trtmc_error**);
    void(TRTMC_CALL* release)(trtmc_mask_track_session*);
} trtmc_mask_track_session_api_v1;
#define TRTMC_TRACK_FACTORY_API(Name, Input)                                                       \
    typedef struct {                                                                               \
        trtmc_api_header header;                                                                   \
        trtmc_status(TRTMC_CALL* create)(trtmc_model*, const trtmc_video_view_v1*, const Input*,   \
                                         const trtmc_config_view_v1*, trtmc_mask_track_session**,  \
                                         trtmc_result**, trtmc_error**);                           \
        const trtmc_mask_track_session_api_v1* session_api;                                        \
    } trtmc_##Name##_api_v1;
#define TRTMC_TASK_FRAMES_POINTS_TO_MASK_TRACKS "frames_points_to_mask_tracks"
#define TRTMC_TASK_FRAMES_BOX_TO_MASK_TRACKS "frames_box_to_mask_tracks"
#define TRTMC_TASK_FRAMES_MASK_TO_MASK_TRACKS "frames_mask_to_mask_tracks"
#define TRTMC_TASK_INTERACTIVE_FRAMES_TEXT_TO_MASK_TRACKS "interactive_frames_text_to_mask_tracks"
#define TRTMC_TASK_FRAMES_BOX_EXEMPLAR_TO_MASK_TRACKS "frames_box_exemplar_to_mask_tracks"
TRTMC_TRACK_FACTORY_API(frames_points_to_mask_tracks, trtmc_frame_object_points_v1)
TRTMC_TRACK_FACTORY_API(frames_box_to_mask_tracks, trtmc_frame_object_box_v1)
TRTMC_TRACK_FACTORY_API(frames_mask_to_mask_tracks, trtmc_frame_object_mask_v1)
TRTMC_TRACK_FACTORY_API(interactive_frames_text_to_mask_tracks, trtmc_frame_text_v1)
TRTMC_TRACK_FACTORY_API(frames_box_exemplar_to_mask_tracks, trtmc_frame_box_exemplar_v1)
#undef TRTMC_TRACK_FACTORY_API

typedef struct trtmc_crop_pose_session trtmc_crop_pose_session;
typedef struct trtmc_rgbd_pose_session trtmc_rgbd_pose_session;
#define TRTMC_TASK_CROP_POSE_TRACKING "crop_pose_tracking"
/* Crop callbacks remain synchronous, serial and per-call, with the same owner/
 * release rules as stateless pose refinement. No callback is stored in session. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* create)(trtmc_model*, const trtmc_config_view_v1*,
                                     trtmc_crop_pose_session**, trtmc_error**);
    trtmc_status(TRTMC_CALL* initialize)(trtmc_crop_pose_session*,
                                         const trtmc_pose_refinement_request_v1*,
                                         const trtmc_config_view_v1*, trtmc_result**,
                                         trtmc_error**);
    trtmc_status(TRTMC_CALL* track)(trtmc_crop_pose_session*, void*, trtmc_pose_crop_callback_v1,
                                    const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* reset)(trtmc_crop_pose_session*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_refined_poses_view_v1*,
                                          trtmc_error**);
    void(TRTMC_CALL* release)(trtmc_crop_pose_session*);
} trtmc_crop_pose_tracking_api_v1;
typedef struct {
    float object_to_camera[16];
} trtmc_object_pose_matrix_v1;
typedef struct {
    trtmc_image_input_v1 rgb;
    trtmc_f32_matrix_view_v1 depth_meters;
    float pixel_intrinsics[9];
} trtmc_rgbd_observation_v1;
#define TRTMC_TASK_RGBD_INITIALIZED_POSE_TO_TRACKED_POSE "rgbd_initialized_pose_to_tracked_pose"
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* create)(trtmc_model*, const trtmc_triangle_mesh_v1*,
                                     const trtmc_object_pose_matrix_v1*,
                                     const trtmc_config_view_v1*, trtmc_rgbd_pose_session**,
                                     trtmc_error**);
    trtmc_status(TRTMC_CALL* track)(trtmc_rgbd_pose_session*, const trtmc_rgbd_observation_v1*,
                                    const trtmc_config_view_v1*, trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* reset)(trtmc_rgbd_pose_session*, const trtmc_object_pose_matrix_v1*,
                                    trtmc_error**);
    trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, trtmc_object_pose_view_v1*,
                                          trtmc_error**);
    void(TRTMC_CALL* release)(trtmc_rgbd_pose_session*);
} trtmc_rgbd_initialized_pose_to_tracked_pose_api_v1;

#ifdef __cplusplus
}
#endif
#endif /* TRTMC_TRACKING_H */

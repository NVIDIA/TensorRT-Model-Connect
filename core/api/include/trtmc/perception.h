/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef TRTMC_PERCEPTION_H
#define TRTMC_PERCEPTION_H
#include "trtmc/image.h"
#include "trtmc/matrix.h"
#include "trtmc/scores.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Original-image pixel coordinates: x right, y down. Boxes use XYXY corners. */
typedef struct {
    float x, y;
} trtmc_pixel_point_v1;
typedef struct {
    float x_min, y_min, x_max, y_max;
} trtmc_pixel_box_v1;
typedef struct {
    trtmc_pixel_point_v1 point;
    uint32_t foreground;
} trtmc_point_prompt_v1;
typedef struct {
    trtmc_pixel_box_v1 box;
    uint32_t positive;
} trtmc_box_exemplar_v1;
typedef struct {
    trtmc_image_input_v1 image;
} trtmc_perception_image_request_v1;
typedef struct {
    trtmc_image_input_v1 image;
    const trtmc_point_prompt_v1* points;
    uint64_t point_count;
} trtmc_image_points_request_v1;
typedef struct {
    trtmc_image_input_v1 image;
    trtmc_pixel_box_v1 box;
} trtmc_image_box_request_v1;
typedef struct {
    trtmc_image_input_v1 image;
    trtmc_f32_matrix_view_v1
        prior_logits; /* Decoder-space low-res logits, NOT an inpainting mask. */
    const trtmc_point_prompt_v1* points;
    uint64_t point_count;
    uint32_t has_box;
    trtmc_pixel_box_v1 box;
} trtmc_image_prior_mask_request_v1;
typedef struct {
    trtmc_image_input_v1 image;
    trtmc_string_view text;
} trtmc_image_query_request_v1;
typedef struct {
    trtmc_image_input_v1 image;
    const trtmc_box_exemplar_v1* exemplars;
    uint64_t exemplar_count;
    trtmc_string_view text_hint;
} trtmc_image_exemplars_request_v1;
typedef struct {
    trtmc_image_input_v1 left, right;
} trtmc_stereo_images_request_v1;

typedef struct {
    const int32_t* labels;
    uint64_t pixel_count;
    uint32_t height, width;
    trtmc_i32_view class_ids;
    trtmc_strings_view class_names;
    trtmc_string_view vocabulary_id;
    uint32_t has_ignore_label;
    int32_t ignore_label;
    uint32_t has_background_label;
    int32_t background_label;
    const float* class_scores;
    uint64_t score_count; /* Optional [class,score_height,score_width]. */
    uint32_t score_height, score_width, score_kind;
} trtmc_semantic_segmentation_view_v1;
enum { TRTMC_MASK_LOGITS = 1, TRTMC_MASK_PROBABILITY = 2, TRTMC_MASK_BINARY = 3 };
typedef struct {
    uint64_t area;
    trtmc_pixel_box_v1 crop_box;
    const trtmc_pixel_point_v1* seed_points;
    uint64_t seed_point_count;
} trtmc_mask_proposal_v1;
/* Stateless output guarantees for N=mask_count:
 * point/box/prior Tasks require iou_count=N;
 * text/exemplar instance Tasks require confidence_count=N and box_count=N;
 * automatic proposals require proposal_count=N and at least one of
 * iou_count/confidence_count/stability_count equal to N. N=0 is valid.
 * Other per-mask arrays may be omitted; nonempty arrays must have N items. */
typedef struct {
    const float* masks;
    uint64_t value_count, mask_count;
    uint32_t height, width, kind;
    const float* predicted_iou;
    uint64_t iou_count; /* Learned quality, not a calibrated probability. */
    const float* confidence;
    uint64_t confidence_count;
    const float* stability;
    uint64_t stability_count;
    const trtmc_pixel_box_v1* boxes;
    uint64_t box_count;
    trtmc_i64_view object_ids; /* Identities within this result; no tracking lifecycle. */
    const trtmc_mask_proposal_v1* proposals;
    uint64_t proposal_count;
    const float* low_res_logits;
    uint64_t low_res_count;
    uint32_t low_res_height, low_res_width;
} trtmc_masks_view_v1;
typedef struct {
    trtmc_f32_matrix_view_v1 disparity; /* Left grid [H,W], x_left-x_right in original pixels. */
} trtmc_disparity_view_v1;
typedef struct {
    const float* points;
    uint64_t point_value_count; /* [H,W,3], meters, x right/y down/z forward. */
    const float* depth;
    const uint8_t* valid;
    uint64_t pixel_count;
    float normalized_intrinsics[9]; /* K operates on normalized image coordinates (u/W,v/H). */
    uint32_t height, width;         /* Invalid entries may be +inf; valid mask is authoritative. */
} trtmc_metric_geometry_view_v1;
typedef struct {
    trtmc_pixel_box_v1 box;
    trtmc_string_view label;
    uint32_t has_confidence;
    float confidence;
} trtmc_grounded_box_v1;
typedef struct {
    trtmc_pixel_point_v1 point;
    trtmc_string_view label;
    uint32_t has_confidence;
    float confidence;
} trtmc_grounded_point_v1;
typedef struct {
    const trtmc_grounded_box_v1* boxes;
    uint64_t count;
    trtmc_string_view raw_response;
    uint32_t parse_complete;
} trtmc_grounded_boxes_view_v1;
typedef struct {
    const trtmc_grounded_point_v1* points;
    uint64_t count;
    trtmc_string_view raw_response;
    uint32_t parse_complete;
} trtmc_grounded_points_view_v1;

/* Fixed-label detection, not text grounding. Coordinates are continuous XYXY
 * in original-image pixels. Each box retains its numeric class ID and score;
 * the family owns filtering and suppression. No label lookup is implied. */
typedef struct {
    trtmc_pixel_box_v1 box;
    float score;
    int32_t class_id;
} trtmc_detected_box_v1;
/* Borrowed through result lifetime. Empty detections retain image dimensions. */
typedef struct {
    const trtmc_detected_box_v1* boxes;
    uint64_t count;
    uint32_t image_height, image_width;
} trtmc_detected_boxes_view_v1;

typedef struct {
    const float* values;
    uint64_t value_count, count;
} trtmc_pose_matrices_v1; /* [N,4,4]. */
enum { TRTMC_POSE_CROP_REFINEMENT = 1, TRTMC_POSE_CROP_SCORING = 2 };
typedef struct {
    trtmc_pose_matrices_v1 poses;
    uint32_t stage;
    uint64_t iteration;
} trtmc_pose_crop_request_v1;
/* NHWC crops: RGB[0,1], then XYZ relative to candidate translation divided by
 * half mesh diameter. Invalid/background XYZ=0. A valid owner/release pair
 * transfers one release obligation, on success OR failure. Both null means
 * arrays remain borrowed for the ENTIRE outer run. Callback-stack arrays are
 * never valid borrowed output. error_message lives through release/outer run. */
typedef struct {
    const float* rendered;
    uint64_t rendered_count;
    const float* observed;
    uint64_t observed_count;
    uint64_t hypothesis_count;
    uint32_t height, width, channels;
    void* owner;
    void(TRTMC_CALL* release)(void* owner);
    trtmc_string_view error_message;
} trtmc_pose_crop_response_v1;
/* Invoked synchronously, serially, only during run. Return OK or an error
 * status (never END/AGAIN); no exception may cross the callback. Metadata and
 * other models may be queried; same-model execution/mutation returns BUSY. */
typedef trtmc_status(TRTMC_CALL* trtmc_pose_crop_callback_v1)(void* context,
                                                              const trtmc_pose_crop_request_v1*,
                                                              trtmc_pose_crop_response_v1*);
typedef struct {
    trtmc_pose_matrices_v1 candidates;
    float mesh_diameter_meters;
    void* context;
    trtmc_pose_crop_callback_v1 provide_crops;
} trtmc_pose_refinement_request_v1;
typedef struct {
    trtmc_pose_matrices_v1 refined_poses;
    const float* scores;
    uint64_t score_count; /* N for multiple hypotheses; single hypothesis permits 0 or 1. */
    int64_t best_index;
    uint32_t all_poses_rigid;
    double refinement_ms, scoring_ms;
} trtmc_refined_poses_view_v1;

enum { TRTMC_MESH_NO_APPEARANCE = 0, TRTMC_MESH_VERTEX_RGB = 1, TRTMC_MESH_TEXTURE_UV = 2 };
typedef struct {
    trtmc_f32_matrix_view_v1 vertices; /* [vertex,3], original object frame in meters. */
    const uint32_t* triangles;
    uint64_t triangle_index_count;
    trtmc_f32_matrix_view_v1 vertex_normals; /* Optional [vertex,3]. */
    uint32_t appearance;
    trtmc_f32_matrix_view_v1 vertex_rgb; /* VertexRgb: [vertex,3], [0,1]. */
    trtmc_f32_matrix_view_v1 uv; /* TextureUv: [vertex,2], U right/V up, shared triangle indices. */
    trtmc_image_input_v1 texture;
    trtmc_pose_matrices_v1 symmetries; /* Empty means identity symmetry only. */
} trtmc_triangle_mesh_v1;
typedef struct {
    trtmc_image_input_v1 rgb;
    trtmc_f32_matrix_view_v1 depth_meters; /* Aligned HW, 0 = invalid. */
    const uint8_t* object_mask;
    uint64_t mask_count; /* Aligned binary HW. */
    float pixel_intrinsics[9];
    trtmc_triangle_mesh_v1 mesh;
} trtmc_rgbd_mesh_mask_request_v1;
typedef struct {
    float object_to_camera[16]; /* Original object -> OpenCV camera, row-major, translations meters.
                                 */
    uint32_t has_score;
    float score;
    uint32_t rigid;
} trtmc_object_pose_view_v1;

#define TRTMC_TASK_IMAGE_TO_SEMANTIC_SEGMENTATION "image_to_semantic_segmentation"
#define TRTMC_TASK_IMAGE_POINTS_TO_MASKS "image_points_to_masks"
#define TRTMC_TASK_IMAGE_BOX_TO_MASKS "image_box_to_masks"
#define TRTMC_TASK_IMAGE_MASK_TO_MASKS "image_mask_to_masks"
#define TRTMC_TASK_IMAGE_TO_MASK_PROPOSALS "image_to_mask_proposals"
#define TRTMC_TASK_IMAGE_TEXT_TO_INSTANCE_MASKS "image_text_to_instance_masks"
#define TRTMC_TASK_IMAGE_BOX_EXEMPLARS_TO_INSTANCE_MASKS "image_box_exemplars_to_instance_masks"
#define TRTMC_TASK_STEREO_IMAGES_TO_DISPARITY "stereo_images_to_disparity"
#define TRTMC_TASK_IMAGE_TO_METRIC_GEOMETRY "image_to_metric_geometry"
#define TRTMC_TASK_IMAGE_TO_BOXES "image_to_boxes"
#define TRTMC_TASK_IMAGE_TEXT_TO_BOXES "image_text_to_boxes"
#define TRTMC_TASK_IMAGE_TEXT_TO_POINTS "image_text_to_points"
#define TRTMC_TASK_POSE_HYPOTHESES_CROPS_TO_REFINED_POSES "pose_hypotheses_crops_to_refined_poses"
#define TRTMC_TASK_RGBD_MESH_MASK_TO_OBJECT_POSE "rgbd_mesh_mask_to_object_pose"

#define TRTMC_PERCEPTION_API(Name, Request, View)                                                  \
    typedef struct {                                                                               \
        trtmc_api_header header;                                                                   \
        trtmc_status(TRTMC_CALL* run)(trtmc_model*, const Request*, const trtmc_config_view_v1*,   \
                                      trtmc_result**, trtmc_error**);                              \
        trtmc_status(TRTMC_CALL* result_view)(const trtmc_result*, View*, trtmc_error**);          \
    } Name;
TRTMC_PERCEPTION_API(trtmc_image_to_semantic_segmentation_api_v1, trtmc_perception_image_request_v1,
                     trtmc_semantic_segmentation_view_v1)
TRTMC_PERCEPTION_API(trtmc_image_points_to_masks_api_v1, trtmc_image_points_request_v1,
                     trtmc_masks_view_v1)
TRTMC_PERCEPTION_API(trtmc_image_box_to_masks_api_v1, trtmc_image_box_request_v1,
                     trtmc_masks_view_v1)
TRTMC_PERCEPTION_API(trtmc_image_mask_to_masks_api_v1, trtmc_image_prior_mask_request_v1,
                     trtmc_masks_view_v1)
TRTMC_PERCEPTION_API(trtmc_image_to_mask_proposals_api_v1, trtmc_perception_image_request_v1,
                     trtmc_masks_view_v1)
TRTMC_PERCEPTION_API(trtmc_image_text_to_instance_masks_api_v1, trtmc_image_query_request_v1,
                     trtmc_masks_view_v1)
TRTMC_PERCEPTION_API(trtmc_image_box_exemplars_to_instance_masks_api_v1,
                     trtmc_image_exemplars_request_v1, trtmc_masks_view_v1)
TRTMC_PERCEPTION_API(trtmc_stereo_images_to_disparity_api_v1, trtmc_stereo_images_request_v1,
                     trtmc_disparity_view_v1)
TRTMC_PERCEPTION_API(trtmc_image_to_metric_geometry_api_v1, trtmc_perception_image_request_v1,
                     trtmc_metric_geometry_view_v1)
TRTMC_PERCEPTION_API(trtmc_image_to_boxes_api_v1, trtmc_perception_image_request_v1,
                     trtmc_detected_boxes_view_v1)
TRTMC_PERCEPTION_API(trtmc_image_text_to_boxes_api_v1, trtmc_image_query_request_v1,
                     trtmc_grounded_boxes_view_v1)
TRTMC_PERCEPTION_API(trtmc_image_text_to_points_api_v1, trtmc_image_query_request_v1,
                     trtmc_grounded_points_view_v1)
TRTMC_PERCEPTION_API(trtmc_pose_hypotheses_crops_to_refined_poses_api_v1,
                     trtmc_pose_refinement_request_v1, trtmc_refined_poses_view_v1)
TRTMC_PERCEPTION_API(trtmc_rgbd_mesh_mask_to_object_pose_api_v1, trtmc_rgbd_mesh_mask_request_v1,
                     trtmc_object_pose_view_v1)
#undef TRTMC_PERCEPTION_API

#define TRTMC_TASK_BATCH_IMAGE_TEXT_TO_BOXES "batch_image_text_to_boxes"
typedef struct {
    trtmc_image_query_request_v1 input;
    trtmc_config_view_v1 config;
} trtmc_batch_image_text_to_boxes_item_v1;
typedef struct {
    const trtmc_batch_image_text_to_boxes_item_v1* items;
    uint64_t count;
} trtmc_batch_image_text_to_boxes_request_v1;
/* Inputs borrow through synchronous run. Success owns N ordered typed results;
 * nested views borrow the result lifetime. Any call failure returns NULL,
 * not a partial-success array. Parsing completeness remains explicit per item. */
typedef struct {
    trtmc_api_header header;
    trtmc_status(TRTMC_CALL* run)(trtmc_model*, const trtmc_batch_image_text_to_boxes_request_v1*,
                                  trtmc_result**, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_count)(const trtmc_result*, uint64_t*, trtmc_error**);
    trtmc_status(TRTMC_CALL* result_item_view)(const trtmc_result*, uint64_t,
                                               trtmc_grounded_boxes_view_v1*, trtmc_error**);
} trtmc_batch_image_text_to_boxes_api_v1;

#ifdef __cplusplus
}
#endif
#endif /* TRTMC_PERCEPTION_H */

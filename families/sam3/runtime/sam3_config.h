/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <vector>

namespace trtmc {

struct Sam3Config {
    int32_t text_max_position_embeddings{32};
    int32_t text_pad_token_id{1};
    int32_t image_size{1008};
    int32_t low_res_mask_size{288};
    int32_t num_queries{200};
    float score_threshold{0.5F};
    float mask_threshold{0.5F};
    std::vector<float> image_mean{0.5F, 0.5F, 0.5F};
    std::vector<float> image_std{0.5F, 0.5F, 0.5F};
    // SAM3.0 prompted-concept video tracking policy.  These defaults are the
    // values in facebook/sam3 revision 3c879f3; bundle metadata overrides them
    // so a checkpoint with a different reviewed policy is never silently run
    // with detector/tracker association values from another variant.
    float detection_threshold{0.5F};
    float detection_nms_threshold{0.1F};
    float association_iou_threshold{0.1F};
    float tracker_association_iou_threshold{0.5F};
    float new_detection_threshold{0.7F};
    float high_confidence_threshold{0.8F};
    float high_iou_threshold{0.8F};
    float overlap_suppression_threshold{0.7F};
    int32_t hotstart_delay{15};
    int32_t hotstart_unmatch_threshold{8};
    int32_t hotstart_duplicate_threshold{8};
    bool suppress_unmatched_only_within_hotstart{true};
    int32_t initial_tracker_keep_alive{30};
    int32_t max_tracker_keep_alive{30};
    int32_t min_tracker_keep_alive{-1};
    bool decrease_keep_alive_for_empty_masks{false};
    int32_t recondition_every_nth_frame{16};
    int32_t fill_hole_area{16};
    int32_t max_tracked_objects{10000};
    int32_t num_mask_memory_frames{7};
    int32_t max_conditioning_frames{4};
    int32_t max_object_pointers{16};
    // The tracker keeps the official 16-frame pointer-position divisor and
    // selects at most four closest conditioning pointers plus fifteen
    // quality-filtered non-conditioning pointers.
    int32_t max_video_frames{1024};
    int32_t max_conditioning_pointers{4};
    int32_t max_pointer_inputs{19};
};

} // namespace trtmc

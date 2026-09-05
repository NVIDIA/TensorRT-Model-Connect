/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/sam3/runtime/sam3_config.h"
#include "families/sam3/runtime/sam3_video_session.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

struct Sam3VideoTextInput {
    std::vector<float> features;
    std::vector<int64_t> features_shape;
    std::vector<int32_t> attention_mask;
};

struct Sam3VideoVisionWorkspace;

// Native Sam3ImageProcessorFast-compatible HWC float32 RGB preprocessing.
std::vector<float> preprocess_sam3_image(const float* hwc_pixels, int32_t height, int32_t width,
                                         const Sam3Config& config);

// Bind B1 vision features directly to detector/tracker consumers and retain
// pipeline-owned snapshots, sparse output staging, and recurrent allocations.
std::shared_ptr<Sam3VideoVisionWorkspace> make_sam3_video_vision_workspace(
    ITrtModule& vision_encoder, ITrtModule& core_engine, ITrtModule& tracker_init_engine,
    ITrtModule& tracker_step_engine, ITrtModule& tracker_memory_engine,
    ITrtModule& tracker_hard_memory_engine, ITrtModule& tracker_hard_memory_batch2_engine,
    ITrtModule& tracker_step_batch2_engine, ITrtModule& tracker_memory_batch2_engine,
    ITrtModule& parallel_tracker_init_engine);

// Construct the fixed sequential-B1 customer processor. Module references must
// outlive the returned callbacks; Sam3Pipeline owns them for the session lifetime.
Sam3VideoFrameProcessor make_sam3_video_frame_processor(
    ITrtModule& vision_encoder, ITrtModule& core_engine, ITrtModule& tracker_init_engine,
    ITrtModule& tracker_step_engine, ITrtModule& tracker_memory_engine, Sam3Config config,
    Sam3VideoTextInput text_input, std::shared_ptr<Sam3VideoVisionWorkspace> vision_workspace,
    ITrtModule& tracker_hard_memory_engine, ITrtModule& tracker_hard_memory_batch2_engine,
    ITrtModule& tracker_step_batch2_engine, ITrtModule& tracker_memory_batch2_engine,
    ITrtModule& parallel_tracker_init_engine, ITrtModule& hard_mask_resize_engine,
    ITrtModule& hard_mask_resize_batch2_engine);

} // namespace trtmc

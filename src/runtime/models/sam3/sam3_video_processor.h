/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/sam3/sam3_config.h"
#include "runtime/models/sam3/sam3_video_session.h"
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
    TrtModule& vision_encoder, TrtModule& core_engine, TrtModule& tracker_init_engine,
    TrtModule& tracker_step_engine, TrtModule& tracker_memory_engine,
    TrtModule* tracker_step_batch2_engine, TrtModule* tracker_memory_batch2_engine,
    TrtModule* parallel_tracker_init_engine);

// Construct the fixed sequential-B1 customer processor. Module references must
// outlive the returned callbacks; Sam3Pipeline owns them for the session lifetime.
Sam3VideoFrameProcessor make_sam3_video_frame_processor(
    TrtModule& vision_encoder, TrtModule& core_engine, TrtModule& tracker_init_engine,
    TrtModule& tracker_step_engine, TrtModule& tracker_memory_engine, Sam3Config config,
    Sam3VideoTextInput text_input, std::shared_ptr<Sam3VideoVisionWorkspace> vision_workspace,
    TrtModule* tracker_step_batch2_engine, TrtModule* tracker_memory_batch2_engine,
    TrtModule* parallel_tracker_init_engine);

} // namespace trtmc

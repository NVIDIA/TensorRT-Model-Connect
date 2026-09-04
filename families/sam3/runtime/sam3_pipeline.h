/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Sam3Pipeline: SAM3 image PCS and prompted-concept video tracking surface.
// Image calls run text, vision, and DETR/mask/scoring plans. Video sessions add
// the checkpoint's learned tracker init, recurrent step, and post-policy memory
// plans while retaining state in the model-owned session processor.

#include "families/sam3/runtime/sam3_config.h"
#include "families/sam3/runtime/sam3_video_session.h"
#include "families/sam3/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trtmc {

struct Sam3VideoVisionWorkspace;

struct Sam3TextFeatures {
    std::vector<float> features;
    std::vector<int64_t> features_shape;
    std::vector<float> hidden_states;
    std::vector<int64_t> hidden_states_shape;
    std::vector<int32_t> attention_mask;
};

class Sam3Pipeline final : public IPointPromptedSegmentation,
                           public ITextPromptedSegmentation,
                           public IVideoSegmentation {
  public:
    const char* task() const noexcept override { return ITextPromptedSegmentation::kTask; }

    Sam3Pipeline(std::unique_ptr<ITrtModule> text_encoder,
                 std::unique_ptr<ITrtModule> vision_encoder,
                 std::unique_ptr<ITrtModule> core_engine, std::shared_ptr<ITokenizer> tokenizer,
                 Sam3Config config, std::string model_id_str,
                 std::unique_ptr<ITrtModule> tracker_init_engine,
                 std::unique_ptr<ITrtModule> tracker_step_engine,
                 std::unique_ptr<ITrtModule> tracker_memory_engine,
                 std::unique_ptr<ITrtModule> tracker_step_batch2_engine,
                 std::unique_ptr<ITrtModule> tracker_memory_batch2_engine,
                 std::unique_ptr<ITrtModule> parallel_tracker_init_engine,
                 std::unique_ptr<ITrtModule> tracker_hard_memory_engine,
                 std::unique_ptr<ITrtModule> tracker_hard_memory_batch2_engine,
                 std::unique_ptr<ITrtModule> hard_mask_resize_engine,
                 std::unique_ptr<ITrtModule> hard_mask_resize_batch2_engine);

    PromptedSegmentationResult segment_prompted(const float* image_pixels, int32_t image_height,
                                                int32_t image_width, float point_x = 0.5F,
                                                float point_y = 0.5F,
                                                bool is_foreground = true) override;

    PromptedSegmentationResult segment_prompted_text(const float* image_pixels,
                                                     int32_t image_height, int32_t image_width,
                                                     const std::string& text_prompt) override;

    std::unique_ptr<Sam3VideoSegmentationSession>
    create_sam3_video_session(const std::string& text_prompt);

    std::unique_ptr<IVideoSegmentationSession> create_video_segmentation_session() override;

  private:
    Sam3TextFeatures encode_text_prompt(const std::string& text_prompt) const;

    std::unique_ptr<ITrtModule> text_encoder_;
    std::unique_ptr<ITrtModule> vision_encoder_;
    std::unique_ptr<ITrtModule> core_engine_;
    std::unique_ptr<ITrtModule> tracker_init_engine_;
    std::unique_ptr<ITrtModule> parallel_tracker_init_engine_;
    std::unique_ptr<ITrtModule> tracker_step_engine_;
    std::unique_ptr<ITrtModule> tracker_step_batch2_engine_;
    std::unique_ptr<ITrtModule> tracker_memory_engine_;
    std::unique_ptr<ITrtModule> tracker_memory_batch2_engine_;
    std::unique_ptr<ITrtModule> tracker_hard_memory_engine_;
    std::unique_ptr<ITrtModule> tracker_hard_memory_batch2_engine_;
    std::unique_ptr<ITrtModule> hard_mask_resize_engine_;
    std::unique_ptr<ITrtModule> hard_mask_resize_batch2_engine_;
    std::shared_ptr<Sam3VideoVisionWorkspace> video_vision_workspace_;
    std::shared_ptr<std::mutex> execution_mutex_{std::make_shared<std::mutex>()};
    std::shared_ptr<ITokenizer> tokenizer_;
    Sam3Config config_;
    std::string model_id_;
};

} // namespace trtmc

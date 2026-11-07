/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// SamPipeline: two-stage segmentation (SAM -- encoder + decoder).
// Uses ITrtModule(image_encoder) + ITrtModule(mask_decoder).

#include "families/sam/runtime/sam_types.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class SamPipeline final : public ISegmentation, public IPointPromptedSegmentation {
  public:
    const char* task() const noexcept override { return IPointPromptedSegmentation::kTask; }

    SamPipeline(std::unique_ptr<ITrtModule> image_encoder, std::unique_ptr<ITrtModule> mask_decoder,
                SamConfig config, std::string model_id_str = "");

    SegmentResult segment(const float* pixels, int32_t height, int32_t width) override;
    PromptedSegmentationResult segment_prompted(const float* image_pixels, int32_t image_height,
                                                int32_t image_width, float point_x = 0.5F,
                                                float point_y = 0.5F,
                                                bool is_foreground = true) override;

  private:
    // Page-locked staging for the encoder's pixel_values (3 x image_size x
    // image_size floats), which the preprocess writes in place so the upload
    // is a direct DMA instead of a pageable copy. Declared before the modules
    // so it outlives any copy they still have in flight. One buffer per
    // pipeline: a pipeline serves one request at a time, as its modules do.
    std::unique_ptr<float, void (*)(float*)> pixel_values_{nullptr, nullptr};
    std::unique_ptr<ITrtModule> image_encoder_;
    std::unique_ptr<ITrtModule> mask_decoder_;
    SamConfig config_;
    std::string model_id_;
};

} // namespace trtmc

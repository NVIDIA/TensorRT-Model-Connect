/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// SegmentPipeline: single-pass segmentation (SegFormer).
// Uses a single TrtModule for pixel_values -> logits/mask output.

#include "runtime/models/segformer/segformer_preprocess_seam.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class SegmentPipeline final : public IPipeline {
  public:
    explicit SegmentPipeline(std::unique_ptr<TrtModule> model,
                             SegformerPreprocessConfig preprocess_config = {},
                             std::string model_id_str = "");

    SegmentResult segment(const float* pixels, int32_t height, int32_t width) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "SegmentPipeline"; }

  private:
    std::unique_ptr<TrtModule> model_;
    SegformerPreprocessConfig preprocess_config_;
    std::string model_id_;
};

} // namespace trtmc

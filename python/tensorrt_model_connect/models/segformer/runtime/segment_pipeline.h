/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// SegmentPipeline: single-pass segmentation (SegFormer).
// Uses a single TrtModule for pixel_values -> logits/mask output.

#include "segformer_preprocess_seam.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/device_tensor.h"
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
    bool try_segment_logits_on_device(const Tensor& input, int32_t target_h, int32_t target_w,
                                      SegmentResult& result);

    std::unique_ptr<TrtModule> model_;
    DeviceTensor device_class_map_;
    SegformerPreprocessConfig preprocess_config_;
    std::string model_id_;
};

} // namespace trtmc

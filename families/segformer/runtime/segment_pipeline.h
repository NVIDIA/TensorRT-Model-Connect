/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// SegmentPipeline: single-pass segmentation (SegFormer).
// Uses a single ITrtModule for pixel_values -> logits/mask output.

#include "families/segformer/runtime/segformer_preprocess_seam.h"
#include "trtmc/runtime/device_tensor.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class SegmentPipeline final : public ISegmentation {
  public:
    explicit SegmentPipeline(std::unique_ptr<ITrtModule> model,
                             SegformerPreprocessConfig preprocess_config = {},
                             std::string model_id_str = "");

    SegmentResult segment(const float* pixels, int32_t height, int32_t width) override;

  private:
    bool try_segment_logits_on_device(const Tensor& input, int32_t target_h, int32_t target_w,
                                      SegmentResult& result);

    std::unique_ptr<ITrtModule> model_;
    DeviceTensor device_class_map_;
    SegformerPreprocessConfig preprocess_config_;
    std::string model_id_;
};

} // namespace trtmc

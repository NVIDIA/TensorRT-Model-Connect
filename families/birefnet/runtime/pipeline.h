/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/birefnet/runtime/image_preprocess_seam.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>

namespace trtmc {

struct BiRefNetRuntimeConfig {
    BiRefNetPreprocessConfig preprocess;
    float mask_threshold{0.5F};
};

class BiRefNetSegmentationPipeline final : public ISegmentation {
  public:
    BiRefNetSegmentationPipeline(std::unique_ptr<ITrtModule> model, BiRefNetRuntimeConfig config);

    SegmentResult segment(const float* pixels, std::int32_t height, std::int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    BiRefNetRuntimeConfig config_;
};

} // namespace trtmc

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/depth_anything/runtime/image_preprocess_seam.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>

namespace trtmc {

class DepthAnythingPipeline final : public IMonocularDepth {
  public:
    DepthAnythingPipeline(std::unique_ptr<ITrtModule> model,
                          DepthAnythingPreprocessConfig preprocess_config);

    MonocularDepthResult estimate_depth(const float* pixels, int32_t height,
                                        int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    DepthAnythingPreprocessConfig preprocess_config_;
};

} // namespace trtmc

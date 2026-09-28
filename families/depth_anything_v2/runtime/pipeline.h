/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/depth_anything_v2/runtime/image_preprocess.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>

namespace trtmc {

class DepthAnythingV2Pipeline final : public IMonocularGeometry {
  public:
    explicit DepthAnythingV2Pipeline(std::unique_ptr<ITrtModule> model,
                                     DepthAnythingV2PreprocessConfig preprocess_config = {});

    GeometryResult estimate_geometry(const float* pixels, int32_t height, int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    DepthAnythingV2PreprocessConfig preprocess_config_;
};

} // namespace trtmc

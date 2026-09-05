/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <string>

namespace trtmc {

class MogePipeline final : public IMonocularGeometry {
  public:
    explicit MogePipeline(std::unique_ptr<ITrtModule> model);

    GeometryResult estimate_geometry(const float* pixels, int32_t height, int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    int32_t min_image_height_{0};
    int32_t min_image_width_{0};
    int32_t max_image_height_{0};
    int32_t max_image_width_{0};
};

} // namespace trtmc

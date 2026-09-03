/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/moge/geometry.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>

namespace trtmc {

class MogePipeline final : public IPipeline, public moge::IGeometryEstimator {
  public:
    MogePipeline(std::unique_ptr<ITrtModule> model, std::string model_id);

    moge::GeometryResult estimate_geometry(const float* pixels, int32_t height,
                                           int32_t width) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "MogePipeline"; }

  private:
    std::unique_ptr<ITrtModule> model_;
    std::string model_id_;
    int32_t min_image_height_{0};
    int32_t min_image_width_{0};
    int32_t max_image_height_{0};
    int32_t max_image_width_{0};
};

} // namespace trtmc

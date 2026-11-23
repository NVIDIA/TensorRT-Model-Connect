/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <memory>
#include <vector>

namespace trtmc::bevfusion {

class Pipeline final : public internal::IModel, public internal::ISpatialFeaturesToClassScores {
  public:
    explicit Pipeline(std::unique_ptr<ITrtModule> module);
    const char* task() const noexcept override {
        return internal::ISpatialFeaturesToClassScores::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::ISpatialFeaturesToClassScores>(*this)};
    }
    internal::SpatialClassScoresResult run(const internal::SpatialFeaturesToClassScoresRequest&,
                                           internal::ConfigView) override;
    std::vector<float> forward(const Tensor& camera_bev, const Tensor& lidar_bev);

  private:
    std::unique_ptr<ITrtModule> module_;
};

} // namespace trtmc::bevfusion

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/model.h"
#include "trtmc/internal/point_cloud.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>

namespace trtmc {

class PointNetPipeline final : public internal::IModel,
                               public internal::IPointsToSemanticSegmentation {
  public:
    PointNetPipeline(std::unique_ptr<ITrtModule> module, std::int32_t max_points,
                     std::int32_t num_classes, std::int32_t input_dim);

    const char* task() const noexcept override {
        return internal::IPointsToSemanticSegmentation::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IPointsToSemanticSegmentation>(*this)};
    }
    internal::PointsToSemanticSegmentationResult
    run(const internal::PointsToSemanticSegmentationRequest& request,
        internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> module_;
    std::int32_t max_points_;
    std::int32_t num_classes_;
    std::int32_t input_dim_;
};

} // namespace trtmc

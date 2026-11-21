/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "trtmc/internal/features.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>

namespace trtmc {

class Uni3DPipeline final : public internal::IModel, public internal::IPointsToEmbedding {
  public:
    Uni3DPipeline(std::unique_ptr<ITrtModule> module, std::int32_t min_points,
                  std::int32_t max_points, std::int32_t input_dim, std::int32_t embed_dim);

    const char* task() const noexcept override {
        return internal::IPointsToEmbedding::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IPointsToEmbedding>(*this)};
    }
    internal::SemanticEmbeddingResult run(const internal::PointsToEmbeddingRequest& request,
                                          internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> module_;
    std::int32_t min_points_;
    std::int32_t max_points_;
    std::int32_t input_dim_;
    std::int32_t embed_dim_;
};

} // namespace trtmc

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace trtmc {

class FoundationPosePipeline final : public IPoseHypothesisRefinement {
  public:
    FoundationPosePipeline(std::unique_ptr<ITrtModule> refiner, std::unique_ptr<ITrtModule> scorer,
                           std::int32_t crop_height, std::int32_t crop_width, std::int32_t channels,
                           std::int32_t max_refiner_batch, std::int32_t max_hypotheses,
                           std::int32_t max_refinement_iterations);

    PoseEstimationResult estimate_pose_hypotheses(const PoseEstimationRequest& request) override;
    void reset_pose_tracking() override;

  private:
    PoseCropBatch request_crops(const PoseEstimationRequest& request,
                                const std::vector<float>& poses, PoseCropStage stage,
                                std::int32_t iteration) const;
    std::vector<float> initial_poses(const PoseEstimationRequest& request) const;
    void validate_request(const PoseEstimationRequest& request,
                          const std::vector<float>& poses) const;
    double refine_poses(const PoseEstimationRequest& request, std::vector<float>& poses);
    void score_poses(const PoseEstimationRequest& request, const std::vector<float>& poses,
                     PoseEstimationResult& result);
    void validate_result(const PoseEstimationResult& result) const;

    std::unique_ptr<ITrtModule> refiner_;
    std::unique_ptr<ITrtModule> scorer_;
    std::int32_t crop_height_;
    std::int32_t crop_width_;
    std::int32_t channels_;
    std::int32_t max_refiner_batch_;
    std::int32_t max_hypotheses_;
    std::int32_t max_refinement_iterations_;
    std::vector<float> tracked_pose_;
};

} // namespace trtmc

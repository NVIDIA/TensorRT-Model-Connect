/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class FoundationPosePipeline final : public IPipeline {
  public:
    FoundationPosePipeline(std::unique_ptr<ITrtModule> refiner, std::unique_ptr<ITrtModule> scorer,
                           int32_t crop_height = 160, int32_t crop_width = 160,
                           int32_t channels = 6, int32_t max_refiner_batch = 42,
                           int32_t max_hypotheses = 252, std::string model_id = "");

    PoseEstimationResult estimate_pose_hypotheses(const PoseEstimationRequest& request) override;
    void reset() override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "FoundationPosePipeline"; }

  private:
    PoseCropBatch request_crops(const PoseEstimationRequest& request,
                                const std::vector<float>& poses, PoseCropStage stage,
                                int32_t iteration) const;
    std::vector<float> initial_poses(const PoseEstimationRequest& request) const;
    void validate_request(const PoseEstimationRequest& request,
                          const std::vector<float>& poses) const;
    double refine_poses(const PoseEstimationRequest& request, std::vector<float>& poses);
    void score_poses(const PoseEstimationRequest& request, const std::vector<float>& poses,
                     PoseEstimationResult& result);
    void validate_result(const PoseEstimationResult& result) const;

    std::unique_ptr<ITrtModule> refiner_;
    std::unique_ptr<ITrtModule> scorer_;
    int32_t crop_height_{0};
    int32_t crop_width_{0};
    int32_t channels_{0};
    int32_t max_refiner_batch_{0};
    int32_t max_hypotheses_{0};
    std::string model_id_;
    std::vector<float> tracked_pose_;
};

} // namespace trtmc

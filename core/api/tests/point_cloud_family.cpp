/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "trtmc/internal/model.h"
#include "trtmc/internal/point_cloud.h"
#include "trtmc/runtime/family_factory.h"

#include <stdexcept>

namespace {
using namespace trtmc::internal;

// Labels each point with its first coordinate so tests can see exactly which
// floats Core handed to the family and how they were split into points.
class PointCloudModel final : public IModel, public IPointsToSemanticSegmentation {
  public:
    explicit PointCloudModel(std::string mode) : mode_(std::move(mode)) {}
    const char* task() const noexcept override { return mode_.c_str(); }
    std::vector<TaskInstance> task_bindings() override {
        return {bind<IPointsToSemanticSegmentation>(*this, {})};
    }
    PointsToSemanticSegmentationResult run(const PointsToSemanticSegmentationRequest& request,
                                           ConfigView) override {
        if (request.points.size() !=
            static_cast<std::size_t>(request.num_points) * request.input_dim)
            throw std::runtime_error("Core handed the family a mis-sized point span");
        PointsToSemanticSegmentationResult result;
        result.num_points = request.num_points;
        result.num_classes = 4;
        result.class_ids = {0, 1, 2, 3};
        result.class_scores.assign(static_cast<std::size_t>(request.num_points) * 4, 0.0F);
        for (std::uint32_t point = 0; point < request.num_points; ++point) {
            const auto label = static_cast<std::int32_t>(request.points[point * request.input_dim]);
            result.labels.push_back(label);
            result.class_scores[point * 4 + static_cast<std::uint32_t>(label)] = 1.0F;
        }
        if (mode_ == "bad_labels")
            result.labels.pop_back();
        return result;
    }

  private:
    std::string mode_;
};
} // namespace

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return new PointCloudModel(context.reader.info().task);
}

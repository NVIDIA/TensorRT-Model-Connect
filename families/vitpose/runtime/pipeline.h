/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/vitpose/runtime/image_ops.h"
#include "trtmc/internal/config.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/perception.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc::vitpose {

// The one query this family answers: the 17 body keypoints of one person. Empty text means it too.
constexpr std::string_view kQuery = "person";

// The bundle's runtime.json: the crop and its normalization, the heatmap size, the DARK kernel and
// one label per keypoint.
struct RuntimeMetadata {
    PreprocessConfig preprocess;
    std::int32_t heatmap_height{0};
    std::int32_t heatmap_width{0};
    std::int32_t dark_kernel{0};
    std::vector<std::string> labels;
};

// Reads runtime.json strictly: one JSON object with exactly the expected keys, none repeated;
// sizes are positive JSON integers within int32; numbers are finite and fit float where float is
// stored; labels are distinct non-empty strings. Throws std::runtime_error otherwise.
RuntimeMetadata parse_runtime_json(const std::vector<char>& data);

// The person box of a request in COCO order: the optional `person_box` (x_min, y_min, x_max,
// y_max in original-image pixels, inside the image, at least one pixel wide and tall), or the
// whole image. Throws internal::ConfigError for an invalid box.
PersonBox request_box(internal::ConfigView config, std::uint32_t height, std::uint32_t width);

class VitPosePoints final : public internal::IModel, public internal::IImageTextToPoints {
  public:
    VitPosePoints(std::unique_ptr<ITrtModule> engine, RuntimeMetadata metadata);

    const char* task() const noexcept override { return IImageTextToPoints::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::GroundedPointsResult run(const internal::ImageTextToPointsRequest& request,
                                       internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> engine_;
    RuntimeMetadata metadata_;
};

} // namespace trtmc::vitpose

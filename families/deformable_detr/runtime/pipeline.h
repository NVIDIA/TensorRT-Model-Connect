/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/deformable_detr/runtime/box_decode.h"
#include "families/deformable_detr/runtime/image_preprocess_seam.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>

namespace trtmc {

struct DeformableDetrRuntimeConfig {
    DeformableDetrPreprocessConfig preprocess;
    std::int32_t num_queries{300};
    std::int32_t num_labels{91};
    float score_threshold{0.3F};
    // The reference post-processor's own default, which is not the query count.
    std::int32_t top_k{100};
};

class DeformableDetrObjectDetectionPipeline final : public IObjectDetection {
  public:
    DeformableDetrObjectDetectionPipeline(std::unique_ptr<ITrtModule> model,
                                          DeformableDetrRuntimeConfig config);

    ObjectDetectionResult detect(const float* pixels, std::int32_t height,
                                 std::int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    DeformableDetrRuntimeConfig config_;
};

} // namespace trtmc

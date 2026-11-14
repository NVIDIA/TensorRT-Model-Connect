/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/oneformer/runtime/image_ops.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/perception.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::oneformer {

struct RuntimeConfig {
    std::int32_t shortest_edge{0};
    std::int32_t longest_edge{0};
    Normalization normalization;
    std::vector<std::string> class_names;
};

// Semantic segmentation with a OneFormer engine conditioned on the semantic
// task: fast-processor preprocessing, engine score map at stride 4, then the
// processor's antialiased bilinear resize to the input image and argmax.
class SemanticSegmentation final : public internal::IModel,
                                   public internal::IImageToSemanticSegmentation {
  public:
    SemanticSegmentation(std::unique_ptr<ITrtModule> engine, RuntimeConfig config);

    const char* task() const noexcept override {
        return internal::IImageToSemanticSegmentation::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IImageToSemanticSegmentation>(*this)};
    }
    internal::SemanticSegmentationResult
    run(const internal::ImageToSemanticSegmentationRequest& request,
        internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> engine_;
    RuntimeConfig config_;
    ImageSize min_input_;
    ImageSize max_input_;
};

} // namespace trtmc::oneformer

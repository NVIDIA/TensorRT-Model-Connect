/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/mask2former/runtime/image_ops.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/perception.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc::mask2former {

struct SemanticHead {
    std::int32_t score_height{0};
    std::int32_t score_width{0};
    std::vector<std::string> labels; // One name per class ID, in class order.
};

// The bundle's runtime.json: the image preprocessing and the semantic head.
struct RuntimeMetadata {
    PreprocessConfig preprocess;
    SemanticHead head;
};

// Reads runtime.json strictly: one JSON object without repeated keys; sizes and the class count are
// positive JSON integers within int32; numbers are finite, and those stored as float fit float;
// labels are nonempty strings, one per class. Throws std::runtime_error otherwise.
RuntimeMetadata parse_runtime_json(const std::vector<char>& data);

class Mask2FormerSemanticSegmentation final : public internal::IModel,
                                              public internal::IImageToSemanticSegmentation {
  public:
    Mask2FormerSemanticSegmentation(std::unique_ptr<ITrtModule> engine, PreprocessConfig preprocess,
                                    SemanticHead head);

    const char* task() const noexcept override {
        return IImageToSemanticSegmentation::kTask.data();
    }
    std::vector<internal::TaskInstance> task_bindings() override {
        return {internal::bind<internal::IImageToSemanticSegmentation>(*this)};
    }
    internal::SemanticSegmentationResult
    run(const internal::ImageToSemanticSegmentationRequest& request,
        internal::ConfigView config) override;

  private:
    std::unique_ptr<ITrtModule> engine_;
    PreprocessConfig preprocess_;
    SemanticHead head_;
};

} // namespace trtmc::mask2former

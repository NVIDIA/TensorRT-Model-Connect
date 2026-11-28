/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/dpt/runtime/image_ops.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>
#include <string>

namespace trtmc::dpt {

// A bundle's runtime.json: exactly the keys the build writes, each of its JSON
// type (no conversion), no key twice, and values check_preprocess_config
// accepts. Throws std::runtime_error otherwise.
PreprocessConfig parse_runtime_config(const std::string& text);

// The engine's inputs and outputs against runtime.json: one float32 input
// "pixel_values" [1, 3, H, W] and one float32 output "predicted_depth"
// [1, 1, H, W]. Throws std::runtime_error otherwise.
void check_engine_io(const ITrtModule& module, const PreprocessConfig& config);

// Resize to the engine's fixed input, run it, and bring the relative inverse
// depth back to the caller's image size the way the reference post-processes.
class DptPipeline final : public IMonocularDepth {
  public:
    DptPipeline(std::unique_ptr<ITrtModule> model, PreprocessConfig preprocess_config);

    MonocularDepthResult estimate_depth(const float* pixels, std::int32_t height,
                                        std::int32_t width) override;

  private:
    std::unique_ptr<ITrtModule> model_;
    PreprocessConfig preprocess_config_;
};

} // namespace trtmc::dpt

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// PatchTSTPipeline: numeric time-series inference pipeline.

#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class PatchTSTPipeline final : public IPipeline {
  public:
    PatchTSTPipeline(std::unique_ptr<TrtModule> module, std::string task_type,
                     int32_t context_length, int32_t num_input_channels, int32_t prediction_length,
                     int32_t num_targets, std::string model_id_str = "");

    EmbeddingResult solve(const float* branch_input, int32_t branch_len, const float* trunk_input,
                          int32_t trunk_len) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "PatchTSTPipeline"; }

  private:
    std::unique_ptr<TrtModule> module_;
    std::string task_type_;
    int32_t context_length_{0};
    int32_t num_input_channels_{1};
    int32_t prediction_length_{0};
    int32_t num_targets_{1};
    std::string model_id_;
};

} // namespace trtmc

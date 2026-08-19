/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// TimesFmPipeline: numeric forecasting pipeline for TimesFM bundles.

#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trtmc {

class TimesFmPipeline final : public IPipeline {
  public:
    TimesFmPipeline(std::unique_ptr<TrtModule> model, int32_t default_freq,
                    int32_t prediction_length, std::string model_id_str = "");

    EmbeddingResult solve(const float* branch_input, int32_t branch_len, const float* trunk_input,
                          int32_t trunk_len) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "TimesFmPipeline"; }

  private:
    static int32_t infer_input_length(const TrtModule& module, const std::string& name,
                                      int32_t fallback);
    static const Tensor* select_forecast_output(const TensorMap& outputs,
                                                std::string& selected_name);

    std::unique_ptr<TrtModule> model_;
    int32_t default_freq_{0};
    int32_t prediction_length_{0};
    std::string model_id_;
};

} // namespace trtmc

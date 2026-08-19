/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// PatchTSMixerPipeline: numeric time-series inference via a single TRT engine.

#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>

namespace trtmc {

struct PatchTSMixerConfig {
    int32_t context_length{1};
    int32_t num_input_channels{1};
    int32_t prediction_length{1};
    int32_t num_targets{1};
    std::string task_kind{"prediction"};
};

PatchTSMixerConfig parse_patchtsmixer_config(const std::string& config_json,
                                             int32_t fallback_context_length);

class PatchTSMixerPipeline final : public IPipeline {
  public:
    PatchTSMixerPipeline(std::unique_ptr<TrtModule> model, PatchTSMixerConfig config,
                         std::string model_id_str = "");

    EmbeddingResult solve(const float* branch_input, int32_t branch_len, const float* trunk_input,
                          int32_t trunk_len) override;

    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "PatchTSMixerPipeline"; }

  private:
    std::unique_ptr<TrtModule> model_;
    PatchTSMixerConfig config_;
    std::string model_id_;
};

} // namespace trtmc

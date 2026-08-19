/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/pipeline.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/tokenizer.h"

#include <cuda_runtime_api.h>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace trtmc {

using MiniMaxH3ModuleLoader =
    std::function<std::unique_ptr<ITrtModule>(const std::string&, cudaStream_t)>;

struct MiniMaxH3Schedule {
    std::vector<float> sigmas;
    std::vector<float> timesteps;
};

MiniMaxH3Schedule make_minimax_h3_schedule(int32_t grid_points, float shift);
void minimax_h3_scheduler_step(float* sample, const float* velocity, std::size_t count,
                               float timestep, float sigma, float sigma_next);

class MiniMaxH3Pipeline final : public IPipeline {
  public:
    MiniMaxH3Pipeline(MiniMaxH3ModuleLoader loader, std::unique_ptr<ITokenizer> tokenizer,
                      std::string model_id, bool first_block_cache = false,
                      float cache_threshold = 0.025F);
    ~MiniMaxH3Pipeline() override;

    bool supports_image_generation() const override { return true; }
    ImageResult generate_image(const std::string& prompt, const GenerateConfig& cfg = {}) override;
    const char* model_id() const override { return model_id_.c_str(); }
    const char* pipeline_type() const override { return "MiniMaxH3Pipeline"; }

  private:
    struct ResidentState;

    MiniMaxH3ModuleLoader loader_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    cudaStream_t stream_{nullptr};
    std::mutex generation_mutex_;
    std::unique_ptr<ResidentState> resident_;
    bool first_block_cache_{false};
    float cache_threshold_{0.025F};
};

} // namespace trtmc

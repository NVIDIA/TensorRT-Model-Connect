/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minimax_h3/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

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
std::vector<float> make_minimax_h3_position_ids(int32_t text_rows);
void minimax_h3_scheduler_step(float* sample, const float* velocity, std::size_t count,
                               float timestep, float sigma, float sigma_next);

class MiniMaxH3Pipeline final : public IImageGeneration {
  public:
    MiniMaxH3Pipeline(MiniMaxH3ModuleLoader loader, std::unique_ptr<ITokenizer> tokenizer,
                      std::string model_id, bool first_block_cache = false,
                      float cache_threshold = 0.025F,
                      std::shared_ptr<void> distributed_owner = nullptr,
                      int32_t distributed_rank = 0, int32_t distributed_world_size = 1);
    ~MiniMaxH3Pipeline() override;

    ImageResult generate_image(const std::string& prompt,
                               const ImageGenerationConfig& cfg = {}) override;

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
    // Keeps the NCCL communicator alive; every rank runs the denoiser and VAE
    // (they contain collectives) but only rank 0 returns pixels.
    std::shared_ptr<void> distributed_owner_;
    int32_t distributed_rank_{0};
    int32_t distributed_world_size_{1};
};

} // namespace trtmc

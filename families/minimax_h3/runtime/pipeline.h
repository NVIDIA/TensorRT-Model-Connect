/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/minimax_h3/runtime/tokenizer.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/video.h"
#include "trtmc/runtime/trt_module.h"

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

struct MiniMaxH3GenerationConfig {
    std::string profile{"minimax-h3-base"};
    int32_t num_inference_steps{50};
    float video_scheduler_shift{12.0F};
    float audio_scheduler_shift{3.0F};
    std::vector<int32_t> dmd_denoising_steps;
};

MiniMaxH3Schedule make_minimax_h3_schedule(int32_t grid_points, float shift);
MiniMaxH3Schedule make_minimax_h3_dmd_schedule(const std::vector<int32_t>& denoising_steps,
                                               float shift);
std::vector<float> make_minimax_h3_position_ids(int32_t text_rows);
void minimax_h3_scheduler_step(float* sample, const float* velocity, std::size_t count,
                               float timestep, float sigma, float sigma_next);

class MiniMaxH3Pipeline final : public internal::IModel, public internal::ITextToAudioVideo {
  public:
    MiniMaxH3Pipeline(MiniMaxH3ModuleLoader loader, std::unique_ptr<ITokenizer> tokenizer,
                      std::string model_id, MiniMaxH3GenerationConfig generation = {},
                      bool first_block_cache = false, float cache_threshold = 0.025F);
    ~MiniMaxH3Pipeline() override;

    const char* task() const noexcept override { return internal::ITextToAudioVideo::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;
    internal::AudioVideoResult run(const internal::TextToAudioVideoRequest& request,
                                   internal::ConfigView config) override;

  private:
    struct ResidentState;

    MiniMaxH3ModuleLoader loader_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::string model_id_;
    MiniMaxH3GenerationConfig generation_;
    cudaStream_t stream_{nullptr};
    std::mutex generation_mutex_;
    std::unique_ptr<ResidentState> resident_;
    bool first_block_cache_{false};
    float cache_threshold_{0.025F};
};

} // namespace trtmc

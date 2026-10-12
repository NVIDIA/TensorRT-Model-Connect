/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/stable_diffusion/runtime/scheduler.h"
#include "families/stable_diffusion/runtime/tokenizer.h"
#include "trtmc/runtime/trt_module.h"
#include "trtmc/task.h"

#include <memory>
#include <string>
#include <vector>

namespace trtmc {

struct StableDiffusionConfig {
    std::int32_t latent_size{64};
    std::int32_t latent_channels{4};
    std::int32_t image_size{512};
    std::int32_t context_length{77};
    std::int32_t context_width{768};
    float scaling_factor{0.18215F};
    std::int32_t num_train_timesteps{1000};
    std::int32_t steps_offset{1};
    std::int32_t default_num_steps{25};
    float default_guidance_scale{7.5F};
    std::vector<float> alphas_cumprod;
};

class StableDiffusionPipeline final : public IImageGeneration {
  public:
    StableDiffusionPipeline(std::unique_ptr<ITrtModule> text_encoder,
                            std::unique_ptr<ITrtModule> unet, std::unique_ptr<ITrtModule> vae,
                            std::unique_ptr<ITokenizer> tokenizer, StableDiffusionConfig config);

    ImageResult generate_image(const std::string& prompt,
                               const ImageGenerationConfig& config = {}) override;

  private:
    std::vector<float> encode(const std::string& text);

    std::unique_ptr<ITrtModule> text_encoder_;
    std::unique_ptr<ITrtModule> unet_;
    std::unique_ptr<ITrtModule> vae_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::int32_t bos_token_id_{0};
    std::int32_t pad_token_id_{0};
    StableDiffusionConfig config_;
    stable_diffusion::DdimScheduler scheduler_;
};

} // namespace trtmc

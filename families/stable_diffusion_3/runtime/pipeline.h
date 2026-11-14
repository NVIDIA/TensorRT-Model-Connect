/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "families/stable_diffusion_3/runtime/tokenizer.h"
#include "trtmc/internal/image.h"
#include "trtmc/internal/model.h"
#include "trtmc/runtime/trt_module.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc {

struct ClipTokens {
    std::int32_t bos{0};
    std::int32_t eos{0};
    std::int32_t pad{0};
    // The text model's configured eos_token_id, which selects the pooling rule.
    std::int32_t config_eos{2};
};

struct StableDiffusion3Config {
    std::int32_t height{1024};
    std::int32_t width{1024};
    std::int32_t latent_height{128};
    std::int32_t latent_width{128};
    std::int32_t latent_channels{16};
    std::int32_t clip_length{77};
    std::int32_t clip_l_width{768};
    std::int32_t clip_g_width{1280};
    std::int32_t clip_l_projection{768};
    std::int32_t clip_g_projection{1280};
    ClipTokens clip_l_tokens;
    ClipTokens clip_g_tokens;
    std::int32_t t5_length{256};
    std::int32_t t5_eos_id{1};
    std::int32_t t5_pad_id{0};
    std::int32_t joint_attention_dim{4096};
    std::int32_t pooled_projection_dim{2048};
    std::int32_t timestep_channels{256};
    std::int32_t num_train_timesteps{1000};
    double shift{3.0};
    float scaling_factor{1.5305F};
    float shift_factor{0.0609F};
    std::int32_t max_batch{2};
    std::int32_t default_num_steps{28};
    double default_guidance_scale{7.0};
};

struct StableDiffusion3Engines {
    std::unique_ptr<ITrtModule> clip_l;
    std::unique_ptr<ITrtModule> clip_g;
    std::unique_ptr<ITrtModule> t5;
    std::unique_ptr<ITrtModule> transformer;
    std::unique_ptr<ITrtModule> vae;
};

struct StableDiffusion3Tokenizers {
    std::unique_ptr<ITokenizer> clip_l;
    std::unique_ptr<ITokenizer> clip_g;
    std::unique_ptr<ITokenizer> t5;
};

// Text-to-image with classifier-free guidance and the flow-matching Euler
// sampler of StableDiffusion3Pipeline. One image per request, at the size
// the bundle was built for.
class StableDiffusion3Pipeline final : public internal::IModel, public internal::ITextToImage {
  public:
    StableDiffusion3Pipeline(StableDiffusion3Engines engines, StableDiffusion3Tokenizers tokenizers,
                             StableDiffusion3Config config);

    const char* task() const noexcept override { return internal::ITextToImage::kTask.data(); }
    std::vector<internal::TaskInstance> task_bindings() override;

    internal::ImageResult run(const internal::TextToImageRequest& request,
                              internal::ConfigView config) override;

  private:
    struct Conditioning {
        std::vector<float> embeds; // [clip_length + t5_length, joint_attention_dim]
        std::vector<float> pooled; // [pooled_projection_dim]
    };

    Conditioning encode(std::string_view text);
    std::vector<float> denoise(const std::vector<float>& latents, float timestep,
                               const Conditioning* unconditional, const Conditioning& conditional);
    internal::ImageResult decode(std::vector<float> latents);

    StableDiffusion3Engines engines_;
    StableDiffusion3Tokenizers tokenizers_;
    StableDiffusion3Config config_;
    std::vector<internal::ConfigField> fields_;
};

} // namespace trtmc

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
#include <optional>
#include <string>
#include <vector>

namespace trtmc {

struct StableDiffusionConfig {
    // "sd" is the single-encoder lineage; "sdxl" adds a second text encoder,
    // a pooled embedding and the micro-conditioning size vector.
    std::string variant{"sd"};
    std::string scheduler{"ddim"};
    std::int32_t latent_size{64};
    std::int32_t latent_channels{4};
    std::int32_t image_size{512};
    std::int32_t context_length{77};
    std::int32_t context_width{768};
    // SDXL only: width of the pooled CLIP-G embedding, and how many
    // micro-conditioning values the UNet expects.
    std::int32_t pooled_width{0};
    std::int32_t time_ids{0};
    std::int32_t tokenizer_2_pad_id{0};
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
                            std::unique_ptr<ITrtModule> text_encoder_2,
                            std::unique_ptr<ITrtModule> unet, std::unique_ptr<ITrtModule> vae,
                            std::unique_ptr<ITokenizer> tokenizer,
                            std::unique_ptr<ITokenizer> tokenizer_2, StableDiffusionConfig config);

    ImageResult generate_image(const std::string& prompt,
                               const ImageGenerationConfig& config = {}) override;

  private:
    std::vector<std::int32_t> tokenize(const ITokenizer& tokenizer, const std::string& text,
                                       std::int32_t bos_id, std::int32_t eos_id,
                                       std::int32_t pad_id) const;
    // Single CLIP encoder, final hidden state. Used by the SD lineage.
    std::vector<float> encode(const std::string& text);
    // Both encoders, penultimate hidden states concatenated along the width,
    // plus CLIP-G's pooled output. Used by SDXL.
    std::vector<float> encode_xl(const std::string& text, std::vector<float>& pooled);

    std::unique_ptr<ITrtModule> text_encoder_;
    std::unique_ptr<ITrtModule> text_encoder_2_;
    std::unique_ptr<ITrtModule> unet_;
    std::unique_ptr<ITrtModule> vae_;
    std::unique_ptr<ITokenizer> tokenizer_;
    std::unique_ptr<ITokenizer> tokenizer_2_;
    std::int32_t bos_token_id_{0};
    std::int32_t pad_token_id_{0};
    std::int32_t bos_token_id_2_{0};
    std::int32_t eos_token_id_2_{0};
    bool xl_{false};
    StableDiffusionConfig config_;
    stable_diffusion::DdimScheduler scheduler_;
    stable_diffusion::EulerAncestralScheduler euler_;
};

} // namespace trtmc

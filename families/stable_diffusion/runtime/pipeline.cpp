/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/stable_diffusion/runtime/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <unordered_map>

namespace trtmc {
namespace {

const Tensor& require_output(const std::unordered_map<std::string, Tensor>& outputs,
                             const char* name) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error("stable_diffusion engine did not produce " + std::string(name));
    return found->second;
}

} // namespace

StableDiffusionPipeline::StableDiffusionPipeline(std::unique_ptr<ITrtModule> text_encoder,
                                                 std::unique_ptr<ITrtModule> unet,
                                                 std::unique_ptr<ITrtModule> vae,
                                                 std::unique_ptr<ITokenizer> tokenizer,
                                                 StableDiffusionConfig config)
    : text_encoder_(std::move(text_encoder)), unet_(std::move(unet)), vae_(std::move(vae)),
      tokenizer_(std::move(tokenizer)), config_(std::move(config)),
      scheduler_(config_.alphas_cumprod, config_.num_train_timesteps, config_.steps_offset) {
    if (!text_encoder_ || !text_encoder_->ok() || !unet_ || !unet_->ok() || !vae_ || !vae_->ok())
        throw std::runtime_error("StableDiffusionPipeline: an engine failed to load");
    if (!tokenizer_)
        throw std::runtime_error("StableDiffusionPipeline: missing tokenizer");
    // CLIP brackets the prompt with <|startoftext|> and <|endoftext|> and pads
    // with the latter. These are applied here rather than left to the
    // tokenizer's post-processor, which emits nothing at all for an empty
    // string - and the empty string is exactly the unconditional prompt that
    // classifier-free guidance leans on.
    bos_token_id_ = tokenizer_->id_for_token("<|startoftext|>");
    pad_token_id_ = tokenizer_->id_for_token("<|endoftext|>");
    if (bos_token_id_ < 0 || pad_token_id_ < 0)
        throw std::runtime_error("stable_diffusion tokenizer lacks the CLIP special tokens");
}

std::vector<float> StableDiffusionPipeline::encode(const std::string& text) {
    auto content = tokenizer_->encode(text);
    const auto limit = static_cast<std::size_t>(config_.context_length);
    // Two slots are reserved for the brackets.
    if (content.size() > limit - 2U)
        content.resize(limit - 2U);
    std::vector<std::int32_t> ids;
    ids.reserve(limit);
    ids.push_back(bos_token_id_);
    ids.insert(ids.end(), content.begin(), content.end());
    ids.push_back(pad_token_id_);
    ids.resize(limit, pad_token_id_);

    Tensor input;
    input.data = ids.data();
    input.shape = {1, config_.context_length};
    input.dtype = DType::kInt32;

    auto outputs = text_encoder_->forward({{"input_ids", input}});
    const Tensor& hidden = require_output(outputs, "last_hidden_state");
    const auto* values = static_cast<const float*>(hidden.data);
    return std::vector<float>(values, values + hidden.numel());
}

ImageResult StableDiffusionPipeline::generate_image(const std::string& prompt,
                                                    const ImageGenerationConfig& request) {
    const int32_t steps = request.num_steps > 0 ? request.num_steps : config_.default_num_steps;
    const float guidance =
        request.guidance_scale > 0.0F ? request.guidance_scale : config_.default_guidance_scale;

    const auto conditional = encode(prompt);
    const auto unconditional = encode(request.negative_prompt);

    const auto latent_count = static_cast<std::size_t>(config_.latent_channels) *
                              config_.latent_size * config_.latent_size;
    std::vector<float> latents(latent_count);
    if (!request.initial_latents.empty()) {
        if (request.initial_latents.size() != latent_count)
            throw std::invalid_argument("stable_diffusion initial latents have the wrong size");
        latents = request.initial_latents;
    } else {
        std::mt19937 engine(request.seed >= 0 ? static_cast<std::uint32_t>(request.seed) : 0U);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        for (auto& value : latents)
            value = normal(engine);
    }

    const auto schedule = scheduler_.timesteps(steps);
    std::vector<float> noise_uncond(latent_count);
    std::vector<float> noise_cond(latent_count);

    for (std::size_t index = 0; index < schedule.size(); ++index) {
        const int32_t timestep = schedule[index];
        const int32_t previous = index + 1 < schedule.size() ? schedule[index + 1] : -1;

        float step_value = static_cast<float>(timestep);
        Tensor sample;
        sample.data = latents.data();
        sample.shape = {1, config_.latent_channels, config_.latent_size, config_.latent_size};
        sample.dtype = DType::kFloat32;
        Tensor step;
        step.data = &step_value;
        step.shape = {1, 1};
        step.dtype = DType::kFloat32;

        // Classifier-free guidance: the same latent is denoised twice, once
        // against the prompt and once against the negative prompt.
        Tensor context;
        context.shape = {1, config_.context_length, config_.context_width};
        context.dtype = DType::kFloat32;

        context.data = const_cast<float*>(unconditional.data());
        auto outputs = unet_->forward(
            {{"sample", sample}, {"timestep", step}, {"encoder_hidden_states", context}});
        const Tensor& u = require_output(outputs, "out_sample");
        std::copy_n(static_cast<const float*>(u.data), latent_count, noise_uncond.begin());

        context.data = const_cast<float*>(conditional.data());
        outputs = unet_->forward(
            {{"sample", sample}, {"timestep", step}, {"encoder_hidden_states", context}});
        const Tensor& c = require_output(outputs, "out_sample");
        std::copy_n(static_cast<const float*>(c.data), latent_count, noise_cond.begin());

        for (std::size_t i = 0; i < latent_count; ++i)
            noise_cond[i] = noise_uncond[i] + guidance * (noise_cond[i] - noise_uncond[i]);

        scheduler_.step(noise_cond.data(), latents.data(), latent_count, timestep, previous);
    }

    for (auto& value : latents)
        value /= config_.scaling_factor;

    Tensor decoded_input;
    decoded_input.data = latents.data();
    decoded_input.shape = {1, config_.latent_channels, config_.latent_size, config_.latent_size};
    decoded_input.dtype = DType::kFloat32;
    auto decoded = vae_->forward({{"latents", decoded_input}});
    const Tensor& image = require_output(decoded, "image");

    const int32_t size = config_.image_size;
    const auto plane = static_cast<std::size_t>(size) * size;
    const auto* pixels = static_cast<const float*>(image.data);

    ImageResult result;
    result.height = size;
    result.width = size;
    result.channels = 3;
    result.num_frames = 1;
    result.pixels.resize(plane * 3U);
    // The engine emits CHW in [-1, 1]; the task contract wants HWC in [0, 1].
    for (std::size_t p = 0; p < plane; ++p) {
        for (int32_t channel = 0; channel < 3; ++channel) {
            const float value = pixels[static_cast<std::size_t>(channel) * plane + p] * 0.5F + 0.5F;
            result.pixels[p * 3U + static_cast<std::size_t>(channel)] =
                std::min(1.0F, std::max(0.0F, value));
        }
    }
    return result;
}

} // namespace trtmc

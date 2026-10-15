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

std::vector<float> to_vector(const Tensor& tensor) {
    const auto* values = static_cast<const float*>(tensor.data);
    return std::vector<float>(values, values + tensor.numel());
}

} // namespace

StableDiffusionPipeline::StableDiffusionPipeline(std::unique_ptr<ITrtModule> text_encoder,
                                                 std::unique_ptr<ITrtModule> text_encoder_2,
                                                 std::unique_ptr<ITrtModule> unet,
                                                 std::unique_ptr<ITrtModule> vae,
                                                 std::unique_ptr<ITokenizer> tokenizer,
                                                 std::unique_ptr<ITokenizer> tokenizer_2,
                                                 StableDiffusionConfig config)
    : text_encoder_(std::move(text_encoder)), text_encoder_2_(std::move(text_encoder_2)),
      unet_(std::move(unet)), vae_(std::move(vae)), tokenizer_(std::move(tokenizer)),
      tokenizer_2_(std::move(tokenizer_2)), xl_(config.variant == "sdxl"),
      config_(std::move(config)),
      scheduler_(config_.alphas_cumprod, config_.num_train_timesteps, config_.steps_offset),
      euler_(config_.alphas_cumprod, config_.num_train_timesteps) {
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
    if (xl_) {
        if (!text_encoder_2_ || !text_encoder_2_->ok() || !tokenizer_2_)
            throw std::runtime_error("stable_diffusion sdxl is missing its second text encoder");
        if (config_.pooled_width <= 0 || config_.time_ids <= 0)
            throw std::runtime_error("stable_diffusion sdxl runtime.json lacks the added "
                                     "conditioning shape");
        bos_token_id_2_ = tokenizer_2_->id_for_token("<|startoftext|>");
        eos_token_id_2_ = tokenizer_2_->id_for_token("<|endoftext|>");
        if (bos_token_id_2_ < 0 || eos_token_id_2_ < 0)
            throw std::runtime_error("stable_diffusion sdxl tokenizer_2 lacks the CLIP special "
                                     "tokens");
    }
}

std::vector<std::int32_t> StableDiffusionPipeline::tokenize(const ITokenizer& tokenizer,
                                                            const std::string& text,
                                                            std::int32_t bos_id,
                                                            std::int32_t eos_id,
                                                            std::int32_t pad_id) const {
    auto content = tokenizer.encode(text);
    const auto limit = static_cast<std::size_t>(config_.context_length);
    // Two slots are reserved for the brackets.
    if (content.size() > limit - 2U)
        content.resize(limit - 2U);
    std::vector<std::int32_t> ids;
    ids.reserve(limit);
    ids.push_back(bos_id);
    ids.insert(ids.end(), content.begin(), content.end());
    ids.push_back(eos_id);
    ids.resize(limit, pad_id);
    return ids;
}

std::vector<float> StableDiffusionPipeline::encode(const std::string& text) {
    auto ids = tokenize(*tokenizer_, text, bos_token_id_, pad_token_id_, pad_token_id_);

    Tensor input;
    input.data = ids.data();
    input.shape = {1, config_.context_length};
    input.dtype = DType::kInt32;

    auto outputs = text_encoder_->forward({{"input_ids", input}});
    return to_vector(require_output(outputs, "last_hidden_state"));
}

std::vector<float> StableDiffusionPipeline::encode_xl(const std::string& text,
                                                      std::vector<float>& pooled) {
    // CLIP-L pads with <|endoftext|>; CLIP-G pads with its own pad id, which
    // is 0 for SDXL. Both are read at the penultimate layer, with no final
    // layer norm, and their hidden states are concatenated along the width.
    auto ids = tokenize(*tokenizer_, text, bos_token_id_, pad_token_id_, pad_token_id_);
    auto ids_2 =
        tokenize(*tokenizer_2_, text, bos_token_id_2_, eos_token_id_2_, config_.tokenizer_2_pad_id);

    Tensor input;
    input.data = ids.data();
    input.shape = {1, config_.context_length};
    input.dtype = DType::kInt32;
    auto first = text_encoder_->forward({{"input_ids", input}});
    const Tensor& hidden_1 = require_output(first, "last_hidden_state");

    // The pooled vector is read at the first <|endoftext|>, which sits one
    // slot past the content. Reading it at the last slot instead would pick a
    // pad position and silently change the conditioning.
    std::int32_t eos_value = 0;
    for (std::size_t i = 0; i < ids_2.size(); ++i) {
        if (ids_2[i] == eos_token_id_2_) {
            eos_value = static_cast<std::int32_t>(i);
            break;
        }
    }
    Tensor input_2;
    input_2.data = ids_2.data();
    input_2.shape = {1, config_.context_length};
    input_2.dtype = DType::kInt32;
    Tensor eos;
    eos.data = &eos_value;
    eos.shape = {1, 1};
    eos.dtype = DType::kInt32;
    auto second = text_encoder_2_->forward({{"input_ids", input_2}, {"eos_index", eos}});
    const Tensor& hidden_2 = require_output(second, "last_hidden_state");
    pooled = to_vector(require_output(second, "pooled_output"));

    const auto length = static_cast<std::size_t>(config_.context_length);
    const auto width_1 = hidden_1.numel() / length;
    const auto width_2 = hidden_2.numel() / length;
    if (static_cast<std::int32_t>(width_1 + width_2) != config_.context_width)
        throw std::runtime_error("stable_diffusion sdxl text encoders do not fill the context");
    const auto* values_1 = static_cast<const float*>(hidden_1.data);
    const auto* values_2 = static_cast<const float*>(hidden_2.data);

    std::vector<float> context(length * (width_1 + width_2));
    for (std::size_t token = 0; token < length; ++token) {
        auto* row = context.data() + token * (width_1 + width_2);
        std::copy_n(values_1 + token * width_1, width_1, row);
        std::copy_n(values_2 + token * width_2, width_2, row + width_1);
    }
    return context;
}

ImageResult StableDiffusionPipeline::generate_image(const std::string& prompt,
                                                    const ImageGenerationConfig& request) {
    const int32_t steps = request.num_steps > 0 ? request.num_steps : config_.default_num_steps;
    // 0.0 is a meaningful guidance scale for the turbo checkpoints, so the
    // fallback triggers on the struct's negative sentinel, not on zero.
    const float guidance =
        request.guidance_scale >= 0.0F ? request.guidance_scale : config_.default_guidance_scale;
    // Below 1.0 the guided combination is a no-op, so the second UNet call is
    // pure cost. Turbo runs at 0.0 and takes the single-pass branch.
    const bool guided = guidance > 1.0F;
    const bool euler = config_.scheduler == "euler_ancestral";

    std::vector<float> pooled_cond;
    std::vector<float> pooled_uncond;
    const auto conditional = xl_ ? encode_xl(prompt, pooled_cond) : encode(prompt);
    std::vector<float> unconditional;
    if (guided)
        unconditional = xl_ ? encode_xl(request.negative_prompt, pooled_uncond)
                            : encode(request.negative_prompt);

    const auto latent_count = static_cast<std::size_t>(config_.latent_channels) *
                              config_.latent_size * config_.latent_size;
    std::mt19937 engine(request.seed >= 0 ? static_cast<std::uint32_t>(request.seed) : 0U);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> latents(latent_count);
    if (!request.initial_latents.empty()) {
        if (request.initial_latents.size() != latent_count)
            throw std::invalid_argument("stable_diffusion initial latents have the wrong size");
        latents = request.initial_latents;
    } else {
        for (auto& value : latents)
            value = normal(engine);
    }

    const auto schedule = euler ? euler_.timesteps(steps) : scheduler_.timesteps(steps);
    if (euler) {
        // Unit-variance latents have to be lifted to the level the walk starts
        // at. DDIM's equivalent factor is exactly 1.0, which is why it has no
        // counterpart on that branch.
        const auto factor = static_cast<float>(euler_.init_noise_sigma(schedule));
        for (auto& value : latents)
            value *= factor;
    }

    // SDXL conditions on the source and target geometry. Nothing here is
    // cropped or resized, so the crop offset is zero and both sizes are the
    // output size.
    const auto edge = static_cast<float>(config_.image_size);
    const std::vector<float> time_ids_values{edge, edge, 0.0F, 0.0F, edge, edge};
    if (xl_ && static_cast<std::int32_t>(time_ids_values.size()) != config_.time_ids)
        throw std::runtime_error("stable_diffusion sdxl micro-conditioning width is unexpected");

    std::vector<float> scaled(latent_count);
    std::vector<float> noise_uncond(latent_count);
    std::vector<float> noise_cond(latent_count);
    std::vector<float> gaussian(latent_count);

    for (std::size_t index = 0; index < schedule.size(); ++index) {
        const int32_t timestep = schedule[index];
        const int32_t previous = index + 1 < schedule.size() ? schedule[index + 1] : -1;

        if (euler)
            euler_.scale_input(latents.data(), scaled.data(), latent_count, timestep);
        else
            scaled = latents;

        float step_value = static_cast<float>(timestep);
        Tensor sample;
        sample.data = scaled.data();
        sample.shape = {1, config_.latent_channels, config_.latent_size, config_.latent_size};
        sample.dtype = DType::kFloat32;
        Tensor step;
        step.data = &step_value;
        step.shape = {1, 1};
        step.dtype = DType::kFloat32;

        Tensor context;
        context.shape = {1, config_.context_length, config_.context_width};
        context.dtype = DType::kFloat32;
        Tensor text_embeds;
        text_embeds.shape = {1, config_.pooled_width};
        text_embeds.dtype = DType::kFloat32;
        Tensor time_ids;
        time_ids.data = const_cast<float*>(time_ids_values.data());
        time_ids.shape = {1, config_.time_ids};
        time_ids.dtype = DType::kFloat32;

        auto denoise = [&](const std::vector<float>& ctx, const std::vector<float>& pooled,
                           std::vector<float>& into) {
            context.data = const_cast<float*>(ctx.data());
            std::unordered_map<std::string, Tensor> inputs{
                {"sample", sample}, {"timestep", step}, {"encoder_hidden_states", context}};
            if (xl_) {
                text_embeds.data = const_cast<float*>(pooled.data());
                inputs.emplace("text_embeds", text_embeds);
                inputs.emplace("time_ids", time_ids);
            }
            auto outputs = unet_->forward(inputs);
            const Tensor& noise = require_output(outputs, "out_sample");
            std::copy_n(static_cast<const float*>(noise.data), latent_count, into.begin());
        };

        // Classifier-free guidance: the same latent is denoised twice, once
        // against the prompt and once against the negative prompt.
        denoise(conditional, pooled_cond, noise_cond);
        if (guided) {
            denoise(unconditional, pooled_uncond, noise_uncond);
            for (std::size_t i = 0; i < latent_count; ++i)
                noise_cond[i] = noise_uncond[i] + guidance * (noise_cond[i] - noise_uncond[i]);
        }

        if (euler) {
            // The last step has sigma_next 0, so sigma_up is 0 and this noise
            // is discarded. A one-step turbo run is therefore deterministic.
            for (auto& value : gaussian)
                value = normal(engine);
            euler_.step(noise_cond.data(), latents.data(), latent_count, timestep, previous,
                        gaussian.data());
        } else {
            scheduler_.step(noise_cond.data(), latents.data(), latent_count, timestep, previous);
        }
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

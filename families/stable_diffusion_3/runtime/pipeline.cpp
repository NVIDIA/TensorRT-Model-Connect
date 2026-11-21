/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/stable_diffusion_3/runtime/pipeline.h"

#include "families/stable_diffusion_3/runtime/sampling.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace trtmc {
namespace {

namespace sd3 = stable_diffusion_3;
using internal::ConfigField;
using internal::ConfigKind;
using internal::ConfigValue;

constexpr std::string_view kNegativePrompt = "negative_prompt";
constexpr std::string_view kNumSteps = "num_steps";
constexpr std::string_view kGuidanceScale = "guidance_scale";
constexpr std::string_view kSeed = "seed";

const Tensor& require_output(const TensorMap& outputs, const char* name, std::size_t count) {
    const auto found = outputs.find(name);
    if (found == outputs.end())
        throw std::runtime_error("stable_diffusion_3 engine did not produce " + std::string(name));
    if (found->second.dtype != DType::kFloat32 || found->second.numel() != count)
        throw std::runtime_error("stable_diffusion_3 engine output " + std::string(name) +
                                 " has an unexpected shape");
    return found->second;
}

const float* values(const Tensor& tensor) {
    return static_cast<const float*>(tensor.data);
}

Tensor host_tensor(void* data, std::vector<int64_t> shape, DType dtype) {
    Tensor tensor;
    tensor.data = data;
    tensor.shape = std::move(shape);
    tensor.dtype = dtype;
    return tensor;
}

std::size_t as_size(std::int32_t value) {
    return static_cast<std::size_t>(value);
}

} // namespace

StableDiffusion3Pipeline::StableDiffusion3Pipeline(StableDiffusion3Engines engines,
                                                   StableDiffusion3Tokenizers tokenizers,
                                                   StableDiffusion3Config config)
    : engines_(std::move(engines)), tokenizers_(std::move(tokenizers)), config_(config) {
    for (const auto* engine : {engines_.clip_l.get(), engines_.clip_g.get(), engines_.t5.get(),
                               engines_.transformer.get(), engines_.vae.get()}) {
        if (engine == nullptr || !engine->ok())
            throw std::runtime_error("stable_diffusion_3: an engine failed to load");
    }
    if (!tokenizers_.clip_l || !tokenizers_.clip_g || !tokenizers_.t5)
        throw std::runtime_error("stable_diffusion_3: a tokenizer is missing");
    if (config_.clip_l_width + config_.clip_g_width > config_.joint_attention_dim ||
        config_.clip_l_projection + config_.clip_g_projection != config_.pooled_projection_dim)
        throw std::runtime_error("stable_diffusion_3 runtime.json widths are inconsistent");
    fields_ = {
        {kNegativePrompt, ConfigKind::String, ConfigValue{std::string_view{}},
         "Prompt that classifier-free guidance steers away from"},
        {kNumSteps, ConfigKind::I64,
         ConfigValue{static_cast<std::int64_t>(config_.default_num_steps)},
         "Flow-matching Euler sampling steps"},
        {kGuidanceScale, ConfigKind::F64, ConfigValue{config_.default_guidance_scale},
         "Classifier-free guidance scale; 1 or less runs the prompt alone"},
        {kSeed, ConfigKind::I64, ConfigValue{std::int64_t{0}},
         "Seed of the initial noise when no initial latents are supplied"},
    };
}

std::vector<internal::TaskInstance> StableDiffusion3Pipeline::task_bindings() {
    return {internal::bind<internal::ITextToImage>(
        *this, Span<const ConfigField>{fields_.data(), fields_.size()})};
}

StableDiffusion3Pipeline::Conditioning StableDiffusion3Pipeline::encode(std::string_view text) {
    const std::string prompt(text);
    const auto clip_length = as_size(config_.clip_length);
    const auto t5_length = as_size(config_.t5_length);
    const auto joint = as_size(config_.joint_attention_dim);

    auto run_clip = [&](ITrtModule& engine, const ITokenizer& tokenizer, const ClipTokens& tokens,
                        std::size_t width, std::size_t projection, std::vector<float>& hidden,
                        std::vector<float>& pooled) {
        auto ids = sd3::frame_clip_ids(tokenizer.encode(prompt), tokens.bos, tokens.eos, tokens.pad,
                                       clip_length);
        std::int32_t eos_index = sd3::clip_pool_index(ids, tokens.config_eos);
        const auto outputs = engine.forward(
            {{"input_ids",
              host_tensor(ids.data(), {1, static_cast<int64_t>(clip_length)}, DType::kInt32)},
             {"eos_index", host_tensor(&eos_index, {1}, DType::kInt32)}});
        const auto* states = values(require_output(outputs, "hidden_states", clip_length * width));
        hidden.assign(states, states + clip_length * width);
        const auto* embeds = values(require_output(outputs, "text_embeds", projection));
        pooled.assign(embeds, embeds + projection);
    };

    const auto width_l = as_size(config_.clip_l_width);
    const auto width_g = as_size(config_.clip_g_width);
    std::vector<float> hidden_l, pooled_l, hidden_g, pooled_g;
    run_clip(*engines_.clip_l, *tokenizers_.clip_l, config_.clip_l_tokens, width_l,
             as_size(config_.clip_l_projection), hidden_l, pooled_l);
    run_clip(*engines_.clip_g, *tokenizers_.clip_g, config_.clip_g_tokens, width_g,
             as_size(config_.clip_g_projection), hidden_g, pooled_g);

    auto t5_ids = sd3::frame_t5_ids(tokenizers_.t5->encode(prompt), config_.t5_eos_id,
                                    config_.t5_pad_id, t5_length);
    const auto t5_outputs = engines_.t5->forward(
        {{"input_ids",
          host_tensor(t5_ids.data(), {1, static_cast<int64_t>(t5_length)}, DType::kInt32)}});
    const auto* t5_states = values(require_output(t5_outputs, "hidden_states", t5_length * joint));

    // The CLIP hidden states sit side by side in the first clip_length rows,
    // zero-padded to the T5 width; the T5 states fill the remaining rows.
    Conditioning result;
    result.embeds.assign((clip_length + t5_length) * joint, 0.0F);
    for (std::size_t token = 0; token < clip_length; ++token) {
        auto* row = result.embeds.data() + token * joint;
        std::copy_n(hidden_l.data() + token * width_l, width_l, row);
        std::copy_n(hidden_g.data() + token * width_g, width_g, row + width_l);
    }
    std::copy_n(t5_states, t5_length * joint, result.embeds.data() + clip_length * joint);
    result.pooled = std::move(pooled_l);
    result.pooled.insert(result.pooled.end(), pooled_g.begin(), pooled_g.end());
    return result;
}

std::vector<float> StableDiffusion3Pipeline::denoise(const std::vector<float>& latents,
                                                     float timestep,
                                                     const Conditioning* unconditional,
                                                     const Conditioning& conditional) {
    const std::int64_t batch = unconditional != nullptr ? 2 : 1;
    if (batch > config_.max_batch)
        throw std::runtime_error("stable_diffusion_3 transformer was built for a smaller batch");
    const auto sample = latents.size();
    const auto tokens = as_size(config_.clip_length + config_.t5_length);
    const auto joint = as_size(config_.joint_attention_dim);
    const auto pooled_width = as_size(config_.pooled_projection_dim);
    const auto channels = as_size(config_.timestep_channels);

    // Row order follows the pipeline: unconditional first, conditional second.
    std::vector<const Conditioning*> rows;
    if (unconditional != nullptr)
        rows.push_back(unconditional);
    rows.push_back(&conditional);
    std::vector<float> hidden, embeds, pooled, projection(channels);
    sd3::timestep_projection(timestep, config_.timestep_channels, projection.data());
    std::vector<float> projections;
    for (const auto* row : rows) {
        hidden.insert(hidden.end(), latents.begin(), latents.end());
        embeds.insert(embeds.end(), row->embeds.begin(), row->embeds.end());
        pooled.insert(pooled.end(), row->pooled.begin(), row->pooled.end());
        projections.insert(projections.end(), projection.begin(), projection.end());
    }

    const auto outputs = engines_.transformer->forward({
        {"hidden_states",
         host_tensor(hidden.data(),
                     {batch, config_.latent_channels, config_.latent_height, config_.latent_width},
                     DType::kFloat32)},
        {"encoder_hidden_states",
         host_tensor(embeds.data(),
                     {batch, static_cast<int64_t>(tokens), static_cast<int64_t>(joint)},
                     DType::kFloat32)},
        {"pooled_projections",
         host_tensor(pooled.data(), {batch, static_cast<int64_t>(pooled_width)}, DType::kFloat32)},
        {"timestep_proj",
         host_tensor(projections.data(), {batch, static_cast<int64_t>(channels)}, DType::kFloat32)},
    });
    const auto* velocity =
        values(require_output(outputs, "velocity", static_cast<std::size_t>(batch) * sample));
    return std::vector<float>(velocity, velocity + static_cast<std::size_t>(batch) * sample);
}

internal::ImageResult StableDiffusion3Pipeline::decode(std::vector<float> latents) {
    for (auto& value : latents)
        value = value / config_.scaling_factor + config_.shift_factor;
    const auto outputs = engines_.vae->forward(
        {{"latents",
          host_tensor(latents.data(),
                      {1, config_.latent_channels, config_.latent_height, config_.latent_width},
                      DType::kFloat32)}});
    const auto plane = as_size(config_.height) * as_size(config_.width);
    const auto* pixels = values(require_output(outputs, "image", plane * 3U));

    internal::ImageResult result;
    result.height = config_.height;
    result.width = config_.width;
    result.channels = 3;
    result.num_frames = 1;
    result.pixels.resize(plane * 3U);
    // The decoder emits CHW in [-1, 1]; the Task returns HWC in [0, 1].
    for (std::size_t position = 0; position < plane; ++position) {
        for (std::size_t channel = 0; channel < 3U; ++channel) {
            const float value = pixels[channel * plane + position] / 2.0F + 0.5F;
            result.pixels[position * 3U + channel] = std::clamp(value, 0.0F, 1.0F);
        }
    }
    return result;
}

internal::ImageResult StableDiffusion3Pipeline::run(const internal::TextToImageRequest& request,
                                                    internal::ConfigView config) {
    const Span<const ConfigField> fields{fields_.data(), fields_.size()};
    internal::validate_config(fields, config);
    const auto negative = *internal::config_get<std::string_view>(config, fields, kNegativePrompt);
    const auto steps = *internal::config_get<std::int64_t>(config, fields, kNumSteps);
    const auto guidance = *internal::config_get<double>(config, fields, kGuidanceScale);
    const auto seed = *internal::config_get<std::int64_t>(config, fields, kSeed);
    if (steps < 1 || steps > config_.num_train_timesteps)
        throw internal::ConfigError("num_steps must be in [1, " +
                                    std::to_string(config_.num_train_timesteps) + "]");
    if (!std::isfinite(guidance) || guidance < 0.0)
        throw internal::ConfigError("guidance_scale must be finite and non-negative");
    if (seed < 0 || seed > std::numeric_limits<std::uint32_t>::max())
        throw internal::ConfigError("seed must be in [0, 2^32 - 1]");

    const auto sample = as_size(config_.latent_channels) * as_size(config_.latent_height) *
                        as_size(config_.latent_width);
    std::vector<float> latents(sample);
    if (!request.initial_latents.empty()) {
        if (request.initial_latents.size() != sample)
            throw std::invalid_argument(
                "stable_diffusion_3 initial latents must hold latent_channels x latent_height x "
                "latent_width = " +
                std::to_string(sample) + " values");
        std::copy(request.initial_latents.begin(), request.initial_latents.end(), latents.begin());
        if (!std::all_of(latents.begin(), latents.end(), [](float v) { return std::isfinite(v); }))
            throw std::invalid_argument("stable_diffusion_3 initial latents must be finite");
    } else {
        std::mt19937 engine(static_cast<std::uint32_t>(seed));
        std::normal_distribution<float> normal(0.0F, 1.0F);
        for (auto& value : latents)
            value = normal(engine);
    }

    // Guidance at or below 1 is a no-op combination, so the pipeline skips
    // the unconditional branch altogether.
    const bool guided = guidance > 1.0;
    const auto conditional = encode(request.prompt);
    Conditioning unconditional;
    if (guided)
        unconditional = encode(negative);

    const auto schedule = sd3::flow_match_schedule(static_cast<std::int32_t>(steps),
                                                   config_.num_train_timesteps, config_.shift);
    const auto scale = static_cast<float>(guidance);
    for (std::size_t index = 0; index < schedule.timesteps.size(); ++index) {
        const auto velocity = denoise(latents, schedule.timesteps[index],
                                      guided ? &unconditional : nullptr, conditional);
        const float dt = schedule.sigmas[index + 1] - schedule.sigmas[index];
        if (guided) {
            const float* uncond = velocity.data();
            const float* cond = velocity.data() + sample;
            for (std::size_t i = 0; i < sample; ++i)
                latents[i] = latents[i] + dt * (uncond[i] + scale * (cond[i] - uncond[i]));
        } else {
            for (std::size_t i = 0; i < sample; ++i)
                latents[i] = latents[i] + dt * velocity[i];
        }
    }
    return decode(std::move(latents));
}

} // namespace trtmc

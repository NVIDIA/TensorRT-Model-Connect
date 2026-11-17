/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/flux3/runtime/pipeline.h"

#include "families/flux3/runtime/sampling.h"
#include "trtmc/runtime/tensor.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::flux3 {
namespace {

constexpr std::int64_t kMaxSeed = 2147483647;
constexpr std::int32_t kImageChannels = 3;

// Surfaces asynchronous engine failures, which forward() does not report.
void synchronize(const ITrtModule& engine, const char* stage) {
    const auto status = cudaStreamSynchronize(engine.stream());
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("flux3 ") + stage +
                                 " failed on the GPU: " + cudaGetErrorString(status));
}

// Borrows the engine's host output staging, valid until its next forward().
Span<const float> float_output(const TensorMap& outputs, const char* name, std::size_t count,
                               const char* stage) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || found->second.data == nullptr ||
        found->second.dtype != DType::kFloat32 || found->second.numel() != count)
        throw std::runtime_error(std::string("flux3 ") + stage + " did not return " +
                                 std::to_string(count) + " float32 values for '" + name + "'");
    return {static_cast<const float*>(found->second.data), count};
}

} // namespace

Flux3Pipeline::Flux3Pipeline(RuntimeConfig config, std::unique_ptr<ITokenizer> tokenizer,
                             Engines engines)
    : config_(std::move(config)), tokenizer_(std::move(tokenizer)), engines_(std::move(engines)) {
    if (!tokenizer_ || !engines_.text_encoder || !engines_.denoiser || !engines_.vae)
        throw std::invalid_argument("flux3 pipeline requires its tokenizer and three engines");
    using internal::ConfigKind;
    using internal::ConfigValue;
    config_fields_ = {
        {"seed", ConfigKind::I64, ConfigValue{std::int64_t{0}},
         "Seed of the initial Gaussian latents, 0 to 2147483647; unused with initial latents"},
        {"num_steps", ConfigKind::I64, ConfigValue{std::int64_t{config_.schedule.default_steps}},
         "Flow-map Euler steps, 1 to 1000"},
    };
}

std::vector<internal::TaskInstance> Flux3Pipeline::task_bindings() {
    return {internal::bind<internal::ITextToImage>(*this,
                                                   {config_fields_.data(), config_fields_.size()})};
}

internal::ImageResult Flux3Pipeline::run(const internal::TextToImageRequest& request,
                                         internal::ConfigView config) {
    const auto options = resolve_options(config);
    auto latents = initial_latents(request.initial_latents, options.seed);
    encode_prompt(request.prompt);
    denoise(latents, options.num_steps);
    return decode(latents);
}

Flux3Pipeline::Options Flux3Pipeline::resolve_options(internal::ConfigView config) const {
    const Span<const internal::ConfigField> fields{config_fields_.data(), config_fields_.size()};
    internal::validate_config(fields, config);
    const auto seed = *internal::config_get<std::int64_t>(config, fields, "seed");
    if (seed < 0 || seed > kMaxSeed)
        throw internal::ConfigError("flux3 config seed must be in [0, 2147483647], got " +
                                    std::to_string(seed));
    const auto steps = *internal::config_get<std::int64_t>(config, fields, "num_steps");
    if (steps < 1 || steps > kMaxSteps)
        throw internal::ConfigError("flux3 config num_steps must be in [1, 1000], got " +
                                    std::to_string(steps));
    return {static_cast<std::uint32_t>(seed), static_cast<std::int32_t>(steps)};
}

std::vector<float> Flux3Pipeline::initial_latents(Span<const float> supplied,
                                                  std::uint32_t seed) const {
    const LatentShape shape{config_.latent_channels, config_.latent_height, config_.latent_width};
    if (supplied.empty()) {
        const auto noise = gaussian_latents(seed, shape.count());
        return pack_latents({noise.data(), noise.size()}, shape);
    }
    if (supplied.size() != shape.count())
        throw std::invalid_argument(
            "flux3 initial_latents must hold " + std::to_string(shape.count()) +
            " float32 values in [channel, row, column] order (" + std::to_string(shape.channels) +
            " x " + std::to_string(shape.height) + " x " + std::to_string(shape.width) + "), got " +
            std::to_string(supplied.size()));
    if (!std::all_of(supplied.begin(), supplied.end(),
                     [](float value) { return std::isfinite(value); }))
        throw std::invalid_argument("flux3 initial_latents must be finite");
    return pack_latents(supplied, shape);
}

void Flux3Pipeline::encode_prompt(std::string_view prompt) {
    const auto& text = config_.text;
    auto ids = tokenizer_->encode(frame_prompt(text.prompt_prefix, prompt, text.prompt_suffix));
    const auto window = text_window(ids.size(), text.pad_multiple, text.max_tokens);
    auto valid_length = static_cast<std::int32_t>(ids.size());
    ids = pad_tokens(std::move(ids), window, text.pad_token_id);
    TensorMap inputs;
    inputs.emplace("input_ids", Tensor{ids.data(), {1, window}, DType::kInt32});
    inputs.emplace("valid_length", Tensor{&valid_length, {1}, DType::kInt32});
    auto& encoder = *engines_.text_encoder;
    encoder.forward_async(inputs);
    synchronize(encoder, "text encoder");
    // The encoder's output buffer is the denoiser input; only its token count varies.
    engines_.denoiser->bind_external("text_embeds", encoder.device_ptr("text_embeds"),
                                     {1, window, text.embed_dim});
}

Span<const float> Flux3Pipeline::predict_velocity(std::vector<float>& latents, float timestep,
                                                  float target) {
    const auto& embedding = config_.timestep_embedding;
    auto current =
        timestep_projection(timestep_feed(timestep), embedding.dim, embedding.max_period);
    auto next = timestep_projection(timestep_feed(target), embedding.dim, embedding.max_period);
    TensorMap inputs;
    inputs.emplace("latents",
                   Tensor{latents.data(), latent_tensor_shape(config_), DType::kFloat32});
    inputs.emplace("timestep_proj", Tensor{current.data(), {1, embedding.dim}, DType::kFloat32});
    inputs.emplace("target_timestep_proj",
                   Tensor{next.data(), {1, embedding.dim}, DType::kFloat32});
    const auto outputs = engines_.denoiser->forward(inputs);
    synchronize(*engines_.denoiser, "denoiser");
    return float_output(outputs, "velocity", latents.size(), "denoiser");
}

void Flux3Pipeline::denoise(std::vector<float>& latents, std::int32_t num_steps) {
    const auto schedule = make_schedule(num_steps, config_.schedule.shift);
    for (std::size_t step = 0; step < static_cast<std::size_t>(num_steps); ++step) {
        const auto velocity =
            predict_velocity(latents, schedule.timesteps[step], schedule.timesteps[step + 1]);
        euler_update(latents, velocity, schedule.sigmas[step], schedule.sigmas[step + 1]);
    }
}

internal::ImageResult Flux3Pipeline::decode(std::vector<float>& latents) {
    TensorMap inputs;
    inputs.emplace("latents",
                   Tensor{latents.data(), latent_tensor_shape(config_), DType::kFloat32});
    const auto outputs = engines_.vae->forward(inputs);
    synchronize(*engines_.vae, "VAE decoder");
    const auto plane =
        static_cast<std::size_t>(config_.height) * static_cast<std::size_t>(config_.width);
    const auto image = float_output(
        outputs, "image", static_cast<std::size_t>(kImageChannels) * plane, "VAE decoder");
    internal::ImageResult result;
    result.pixels = image_to_pixels(image, config_.height, config_.width);
    result.height = config_.height;
    result.width = config_.width;
    result.channels = kImageChannels;
    result.num_frames = 1;
    return result;
}

} // namespace trtmc::flux3

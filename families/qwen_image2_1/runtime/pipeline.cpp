/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen_image2_1/runtime/pipeline.h"

#include "families/qwen_image2_1/runtime/sampling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trtmc::qwen_image2_1 {
namespace {

using Clock = std::chrono::steady_clock;
using internal::ConfigError;
using internal::ConfigField;
using internal::ConfigKind;

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void require_tensor(const ITrtModule& module, bool input, const std::string& name,
                    const std::vector<std::int64_t>& shape, DType dtype, const char* engine) {
    const bool present = input ? module.has_input(name) : module.has_output(name);
    if (!present || module.tensor_shape(name) != shape || module.tensor_dtype(name) != dtype)
        throw std::runtime_error(std::string("Qwen-Image 2.1 ") + engine +
                                 " engine does not match the bundle's runtime.json: " + name);
}

std::vector<float> copy_output(const TensorMap& outputs, const char* name, std::size_t count,
                               const char* engine) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || found->second.data == nullptr ||
        found->second.dtype != DType::kFloat32 || found->second.numel() != count)
        throw std::runtime_error(std::string("Qwen-Image 2.1 ") + engine +
                                 " engine returned an unexpected " + name);
    const auto* values = static_cast<const float*>(found->second.data);
    return std::vector<float>(values, values + count);
}

std::int64_t require_i64(internal::ConfigView config, Span<const ConfigField> fields,
                         std::string_view name) {
    return *internal::config_get<std::int64_t>(config, fields, name);
}

} // namespace

Pipeline::Pipeline(Engines engines, std::unique_ptr<ITokenizer> tokenizer, RuntimeConfig config)
    : engines_(std::move(engines)), tokenizer_(std::move(tokenizer)), config_(std::move(config)) {
    if (!engines_.text_encoder || !engines_.denoiser || !engines_.vae || !tokenizer_)
        throw std::invalid_argument("Qwen-Image 2.1 requires three engines and a tokenizer");

    const std::int64_t tokens = config_.image_tokens();
    const std::int64_t text = config_.text_tokens;
    const std::int64_t hidden = config_.text_hidden_size;
    const std::int64_t channels = config_.latent_channels;
    require_tensor(*engines_.text_encoder, true, "input_ids", {config_.prefix_tokens + text},
                   DType::kInt32, "text encoder");
    require_tensor(*engines_.text_encoder, false, "text_hidden", {text, hidden}, DType::kFloat32,
                   "text encoder");
    require_tensor(*engines_.denoiser, true, "latents", {1, tokens, channels}, DType::kFloat32,
                   "denoiser");
    require_tensor(*engines_.denoiser, true, "timestep", {1}, DType::kFloat32, "denoiser");
    require_tensor(*engines_.denoiser, true, "text_hidden", {1, text, hidden}, DType::kFloat32,
                   "denoiser");
    require_tensor(*engines_.denoiser, true, "text_length", {1}, DType::kInt32, "denoiser");
    require_tensor(*engines_.denoiser, false, "velocity", {1, tokens, channels}, DType::kFloat32,
                   "denoiser");
    require_tensor(*engines_.vae, true, "latents",
                   {1, channels, config_.latent_height(), config_.latent_width()}, DType::kFloat32,
                   "VAE");
    require_tensor(*engines_.vae, false, "image",
                   {1, config_.image_channels, config_.image_height, config_.image_width},
                   DType::kFloat32, "VAE");

    fields_ = {
        {"seed", ConfigKind::I64, std::int64_t{0}, "Seed of the Gaussian initial latents"},
        {"num_steps", ConfigKind::I64, std::int64_t{config_.default_num_steps}, "Denoising steps"},
        {"cfg_scale", ConfigKind::F64, 1.0,
         "True classifier-free guidance scale; applied above 1 together with negative_prompt"},
        {"negative_prompt", ConfigKind::String, std::nullopt,
         "Negative prompt for classifier-free guidance; unused unless cfg_scale is above 1"},
        {"height", ConfigKind::I64, std::int64_t{config_.image_height},
         "Image height; fixed by the bundle"},
        {"width", ConfigKind::I64, std::int64_t{config_.image_width},
         "Image width; fixed by the bundle"},
    };
}

std::vector<internal::TaskInstance> Pipeline::task_bindings() {
    return {internal::bind<internal::ITextToImage>(
        *this, Span<const ConfigField>{fields_.data(), fields_.size()})};
}

GenerationSettings Pipeline::resolve(internal::ConfigView config) const {
    const Span<const ConfigField> fields{fields_.data(), fields_.size()};
    internal::validate_config(fields, config);

    GenerationSettings settings;
    const auto seed = require_i64(config, fields, "seed");
    if (seed < 0)
        throw ConfigError("Qwen-Image 2.1 seed must be non-negative");
    settings.seed = static_cast<std::uint64_t>(seed);

    const auto steps = require_i64(config, fields, "num_steps");
    // One step leaves the schedule's terminal stretch undefined, as in the reference.
    if (steps < 2 || steps > std::numeric_limits<std::int32_t>::max())
        throw ConfigError("Qwen-Image 2.1 num_steps must be an integer of at least 2");
    settings.num_steps = static_cast<std::int32_t>(steps);

    if (require_i64(config, fields, "height") != config_.image_height ||
        require_i64(config, fields, "width") != config_.image_width) {
        throw ConfigError("Qwen-Image 2.1 bundle generates " + std::to_string(config_.image_width) +
                          "x" + std::to_string(config_.image_height) +
                          " images; rebuild it for another size");
    }

    settings.cfg_scale = *internal::config_get<double>(config, fields, "cfg_scale");
    // The scale is applied in float32, like the velocities it combines.
    if (!std::isfinite(settings.cfg_scale) || settings.cfg_scale < 0.0 ||
        settings.cfg_scale > static_cast<double>(std::numeric_limits<float>::max()))
        throw ConfigError("Qwen-Image 2.1 cfg_scale must be a finite, non-negative float32");

    // The pipeline guides only when the scale is above 1 and a negative prompt is given; it
    // warns about, and otherwise ignores, either one without the other.
    const auto negative = internal::config_get<std::string_view>(config, fields, "negative_prompt");
    if (settings.cfg_scale > 1.0 && negative) {
        settings.use_negative_prompt = true;
        settings.negative_prompt = std::string(*negative);
    } else if (settings.cfg_scale > 1.0) {
        std::cerr << "[qwen-image2.1] cfg_scale=" << settings.cfg_scale
                  << " has no effect without negative_prompt\n";
    } else if (negative) {
        std::cerr << "[qwen-image2.1] negative_prompt has no effect unless cfg_scale is above 1\n";
    }
    return settings;
}

Pipeline::EncodedPrompt Pipeline::encode_prompt(const std::string& prompt) {
    // An empty prompt would leave the encoder nothing to read; the pipeline substitutes a space.
    const std::string text = prompt.empty() ? " " : prompt;
    auto ids = tokenizer_->encode(config_.prompt_prefix + text + config_.prompt_suffix);

    const auto prefix = static_cast<std::size_t>(config_.prefix_tokens);
    const auto capacity = prefix + static_cast<std::size_t>(config_.text_tokens);
    if (ids.size() <= prefix)
        throw std::runtime_error("Qwen-Image 2.1 prompt frame lost its system turn");
    if (ids.size() > capacity) {
        throw std::invalid_argument(
            "Qwen-Image 2.1 prompt needs " + std::to_string(ids.size() - prefix) +
            " text tokens; this bundle was built for " + std::to_string(config_.text_tokens));
    }
    const auto length = static_cast<std::int32_t>(ids.size() - prefix);
    ids.resize(capacity, config_.pad_token_id);

    TensorMap inputs;
    inputs.emplace("input_ids",
                   Tensor{ids.data(), {static_cast<std::int64_t>(capacity)}, DType::kInt32});
    const auto outputs = engines_.text_encoder->forward(inputs);
    const auto width = static_cast<std::size_t>(config_.text_hidden_size);
    auto hidden =
        copy_output(outputs, "text_hidden", static_cast<std::size_t>(config_.text_tokens) * width,
                    "text encoder");
    // Rows of padding slots carry no prompt content. The denoiser masks them out as keys; zero
    // them so that what it computes for those rows does not depend on the padding token.
    std::fill(hidden.begin() +
                  static_cast<std::ptrdiff_t>(static_cast<std::size_t>(length) * width),
              hidden.end(), 0.0F);
    return {std::move(hidden), length};
}

std::vector<float> Pipeline::predict_velocity(const std::vector<float>& latents, float timestep,
                                              const EncodedPrompt& prompt) {
    std::int32_t length = prompt.length;
    TensorMap inputs;
    inputs.emplace("latents", Tensor{const_cast<float*>(latents.data()),
                                     {1, config_.image_tokens(), config_.latent_channels},
                                     DType::kFloat32});
    inputs.emplace("timestep", Tensor{&timestep, {1}, DType::kFloat32});
    inputs.emplace("text_hidden", Tensor{const_cast<float*>(prompt.hidden.data()),
                                         {1, config_.text_tokens, config_.text_hidden_size},
                                         DType::kFloat32});
    inputs.emplace("text_length", Tensor{&length, {1}, DType::kInt32});
    const auto outputs = engines_.denoiser->forward(inputs);
    return copy_output(outputs, "velocity", config_.latent_count(), "denoiser");
}

ImageResult Pipeline::decode(const std::vector<float>& latents) {
    auto chw = unpack_latents(latents, config_.latent_channels, config_.latent_height(),
                              config_.latent_width());
    TensorMap inputs;
    inputs.emplace("latents", Tensor{chw.data(),
                                     {1, config_.latent_channels, config_.latent_height(),
                                      config_.latent_width()},
                                     DType::kFloat32});
    const auto outputs = engines_.vae->forward(inputs);
    const auto count = static_cast<std::size_t>(config_.image_channels) *
                       static_cast<std::size_t>(config_.image_height) *
                       static_cast<std::size_t>(config_.image_width);
    const auto image = copy_output(outputs, "image", count, "VAE");

    ImageResult result;
    result.height = config_.image_height;
    result.width = config_.image_width;
    result.channels = config_.image_channels;
    result.num_frames = 1;
    result.pixels = image_to_hwc_unit(image.data(), config_.image_channels, config_.image_height,
                                      config_.image_width);
    return result;
}

ImageResult Pipeline::run(const internal::TextToImageRequest& request,
                          internal::ConfigView config) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto settings = resolve(config);

    // Caller-supplied latents use the layout [latent_channels, latent_height, latent_width].
    std::vector<float> latents;
    if (request.initial_latents.empty()) {
        const auto noise = gaussian_latents(config_.latent_count(), settings.seed);
        latents = pack_latents(noise.data(), config_.latent_channels, config_.latent_height(),
                               config_.latent_width());
    } else {
        if (request.initial_latents.size() != config_.latent_count()) {
            throw std::invalid_argument("Qwen-Image 2.1 initial latents must hold " +
                                        std::to_string(config_.latent_count()) +
                                        " float32 values laid out as [" +
                                        std::to_string(config_.latent_channels) + ", " +
                                        std::to_string(config_.latent_height()) + ", " +
                                        std::to_string(config_.latent_width()) + "]");
        }
        if (!std::all_of(request.initial_latents.begin(), request.initial_latents.end(),
                         [](float value) { return std::isfinite(value); }))
            throw std::invalid_argument("Qwen-Image 2.1 initial latents must be finite");
        latents = pack_latents(request.initial_latents.data(), config_.latent_channels,
                               config_.latent_height(), config_.latent_width());
    }

    const auto begin = Clock::now();
    const auto prompt = encode_prompt(std::string(request.prompt));
    EncodedPrompt negative;
    if (settings.use_negative_prompt)
        negative = encode_prompt(settings.negative_prompt);
    const auto encoded = Clock::now();

    const auto sigmas = flow_match_sigmas(config_, settings.num_steps);
    const auto guidance = static_cast<float>(settings.cfg_scale);
    for (std::int32_t step = 0; step < settings.num_steps; ++step) {
        const auto index = static_cast<std::size_t>(step);
        const float timestep = reference_timestep(sigmas[index], config_.num_train_timesteps);
        auto velocity = predict_velocity(latents, timestep, prompt);
        if (settings.use_negative_prompt) {
            const auto unconditional = predict_velocity(latents, timestep, negative);
            for (std::size_t element = 0; element < velocity.size(); ++element) {
                velocity[element] = unconditional[element] +
                                    guidance * (velocity[element] - unconditional[element]);
            }
        }
        euler_step(latents, velocity, sigmas[index], sigmas[index + 1]);
    }
    const auto denoised = Clock::now();

    auto result = decode(latents);
    const auto decoded = Clock::now();
    std::cerr << std::fixed << std::setprecision(3)
              << "[qwen-image2.1.perf] text_encoder_ms=" << milliseconds(begin, encoded)
              << " denoiser_ms=" << milliseconds(encoded, denoised)
              << " vae_decoder_ms=" << milliseconds(denoised, decoded)
              << " total_ms=" << milliseconds(begin, decoded) << " steps=" << settings.num_steps
              << " text_tokens=" << prompt.length << '\n';
    return result;
}

} // namespace trtmc::qwen_image2_1

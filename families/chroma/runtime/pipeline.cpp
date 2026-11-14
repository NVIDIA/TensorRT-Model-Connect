/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/chroma/runtime/pipeline.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

namespace trtmc::chroma {
namespace {

using internal::ConfigError;
using internal::ConfigField;
using internal::ConfigKind;

constexpr std::int32_t kMaxSteps = 1000;

std::vector<float> copy_output(const TensorMap& outputs, const char* name, std::size_t count) {
    const auto found = outputs.find(name);
    if (found == outputs.end() || found->second.dtype != DType::kFloat32 ||
        found->second.numel() != count)
        throw std::runtime_error(std::string("Chroma engine output is invalid: ") + name);
    const auto* data = static_cast<const float*>(found->second.data);
    return std::vector<float>(data, data + count);
}

void require_input(const ITrtModule& module, const char* name, const std::vector<int64_t>& shape,
                   DType dtype) {
    if (!module.has_input(name) || module.tensor_shape(name) != shape ||
        module.tensor_dtype(name) != dtype)
        throw std::invalid_argument(std::string("Chroma engine input contract mismatch: ") + name);
}

void require_output(const ITrtModule& module, const char* name, const std::vector<int64_t>& shape) {
    if (!module.has_output(name) || module.tensor_shape(name) != shape ||
        module.tensor_dtype(name) != DType::kFloat32)
        throw std::invalid_argument(std::string("Chroma engine output contract mismatch: ") + name);
}

std::int64_t require_positive(const nlohmann::json& root, const char* key) {
    const auto value = root.at(key).get<std::int64_t>();
    if (value < 1 || value > std::numeric_limits<std::int32_t>::max())
        throw std::invalid_argument(std::string("Chroma runtime.json has an invalid ") + key);
    return value;
}

} // namespace

RuntimeConfig parse_runtime_config(std::string_view json_text) {
    const auto root = nlohmann::json::parse(json_text);
    RuntimeConfig config;
    const auto precision = root.at("precision").get<std::string>();
    if (precision != "bf16" && precision != "fp32")
        throw std::invalid_argument("Chroma runtime.json has an unsupported precision");
    config.bfloat16 = precision == "bf16";
    config.height = static_cast<std::int32_t>(require_positive(root, "height"));
    config.width = static_cast<std::int32_t>(require_positive(root, "width"));
    config.text_length = static_cast<std::int32_t>(require_positive(root, "text_length"));
    config.latent_channels = static_cast<std::int32_t>(require_positive(root, "latent_channels"));
    config.vae_scale_factor = static_cast<std::int32_t>(require_positive(root, "vae_scale_factor"));
    config.vae_scaling = root.at("vae_scaling_factor").get<float>();
    config.vae_shift = root.at("vae_shift_factor").get<float>();
    config.scheduler_shift = root.at("scheduler_shift").get<float>();
    config.num_train_timesteps =
        static_cast<std::int32_t>(require_positive(root, "num_train_timesteps"));
    config.default_steps = require_positive(root, "default_num_inference_steps");
    config.default_guidance = root.at("default_guidance_scale").get<double>();
    if (config.height % (2 * config.vae_scale_factor) != 0 ||
        config.width % (2 * config.vae_scale_factor) != 0 || config.vae_scaling == 0.0F ||
        config.num_train_timesteps != 1000)
        throw std::invalid_argument("Chroma runtime.json describes an unsupported geometry");
    return config;
}

ChromaPipeline::ChromaPipeline(const ModuleLoader& loader, T5Tokenizer tokenizer,
                               RuntimeConfig config)
    : config_(config), tokenizer_(std::move(tokenizer)) {
    text_encoder_ = loader("text_encoder.plan");
    transformer_ = loader("transformer.plan");
    vae_decoder_ = loader("vae_decoder.plan");
    const std::int64_t length = config_.text_length;
    const auto shape = config_.latent_shape();
    const std::int64_t tokens = shape.tokens();
    const std::int64_t packed = 4 * static_cast<std::int64_t>(shape.channels);
    require_input(*text_encoder_, "input_ids", {1, length}, DType::kInt32);
    require_input(*text_encoder_, "attention_mask", {1, length}, DType::kInt32);
    const auto width = text_encoder_->tensor_shape("text_embeddings");
    if (width.size() != 3)
        throw std::invalid_argument("Chroma text encoder output must be rank 3");
    require_output(*text_encoder_, "text_embeddings", {1, length, width[2]});
    require_input(*transformer_, "hidden_states", {1, tokens, packed}, DType::kFloat32);
    require_input(*transformer_, "encoder_hidden_states", {1, length, width[2]}, DType::kFloat32);
    require_input(*transformer_, "timestep_features", {1, 16}, DType::kFloat32);
    require_input(*transformer_, "text_mask", {1, length}, DType::kFloat32);
    require_output(*transformer_, "noise_pred", {1, tokens, packed});
    require_input(*vae_decoder_, "latents", {1, shape.channels, shape.height, shape.width},
                  DType::kFloat32);
    require_output(*vae_decoder_, "image", {1, 3, config_.height, config_.width});

    static constexpr std::string_view kEmpty{};
    fields_ = {
        {"negative_prompt", ConfigKind::String, internal::ConfigValue{kEmpty},
         "Classifier-free guidance negative prompt (used when guidance_scale > 1)"},
        {"num_inference_steps", ConfigKind::I64, internal::ConfigValue{config_.default_steps},
         "Flow-matching Euler steps, 1 to 1000"},
        {"guidance_scale", ConfigKind::F64, internal::ConfigValue{config_.default_guidance},
         "Classifier-free guidance scale; values <= 1 run the prompt only"},
        {"seed", ConfigKind::I64, internal::ConfigValue{std::int64_t{0}},
         "Noise seed, 0 to 4294967295; not combined with initial latents"},
        {"height", ConfigKind::I64, internal::ConfigValue{std::int64_t{config_.height}},
         "Image height; fixed by the bundle"},
        {"width", ConfigKind::I64, internal::ConfigValue{std::int64_t{config_.width}},
         "Image width; fixed by the bundle"},
    };
}

std::vector<internal::TaskInstance> ChromaPipeline::task_bindings() {
    return {internal::bind<internal::ITextToImage>(*this, {fields_.data(), fields_.size()})};
}

ChromaPipeline::Settings ChromaPipeline::resolve(internal::ConfigView config, bool replay) const {
    const Span<const ConfigField> fields{fields_.data(), fields_.size()};
    internal::validate_config(fields, config);
    Settings settings;
    settings.negative_prompt =
        std::string(*internal::config_get<std::string_view>(config, fields, "negative_prompt"));
    const auto steps = *internal::config_get<std::int64_t>(config, fields, "num_inference_steps");
    if (steps < 1 || steps > kMaxSteps)
        throw ConfigError("Chroma num_inference_steps must be in [1, 1000]");
    settings.steps = static_cast<std::int32_t>(steps);
    const auto guidance = *internal::config_get<double>(config, fields, "guidance_scale");
    if (!std::isfinite(guidance) || guidance < 0.0 || guidance > 100.0)
        throw ConfigError("Chroma guidance_scale must be finite and in [0, 100]");
    settings.guidance = static_cast<float>(guidance);
    const auto seed = *internal::config_get<std::int64_t>(config, fields, "seed");
    if (seed < 0 || seed > std::numeric_limits<std::uint32_t>::max())
        throw ConfigError("Chroma seed must be in [0, 4294967295]");
    if (replay && internal::config_provided(config, "seed"))
        throw ConfigError("Chroma seed cannot be combined with initial latents");
    settings.seed = static_cast<std::uint32_t>(seed);
    if (*internal::config_get<std::int64_t>(config, fields, "height") != config_.height ||
        *internal::config_get<std::int64_t>(config, fields, "width") != config_.width)
        throw ConfigError("Chroma image size is fixed by the bundle (" +
                          std::to_string(config_.height) + "x" + std::to_string(config_.width) +
                          ")");
    return settings;
}

ChromaPipeline::EncodedText ChromaPipeline::encode(std::string_view text) {
    const auto length = static_cast<std::size_t>(config_.text_length);
    auto prompt = tokenizer_.encode_padded(text, length);
    TensorMap inputs;
    inputs.emplace("input_ids", Tensor{prompt.ids.data(), {1, config_.text_length}, DType::kInt32});
    inputs.emplace("attention_mask",
                   Tensor{prompt.mask.data(), {1, config_.text_length}, DType::kInt32});
    const auto width = static_cast<std::size_t>(text_encoder_->tensor_shape("text_embeddings")[2]);
    EncodedText encoded;
    encoded.embeddings =
        copy_output(text_encoder_->forward(inputs), "text_embeddings", length * width);
    encoded.mask = denoiser_text_mask(prompt.mask);
    return encoded;
}

std::vector<float> ChromaPipeline::denoise(const std::vector<float>& latents,
                                           const EncodedText& text,
                                           const std::array<float, 16>& features) {
    const auto shape = config_.latent_shape();
    const std::int64_t tokens = shape.tokens();
    const std::int64_t packed = 4 * static_cast<std::int64_t>(shape.channels);
    const std::int64_t width =
        static_cast<std::int64_t>(text.embeddings.size()) / config_.text_length;
    auto features_copy = features;
    auto latents_copy = latents;
    auto embeddings = text.embeddings;
    auto mask = text.mask;
    TensorMap inputs;
    inputs.emplace("hidden_states",
                   Tensor{latents_copy.data(), {1, tokens, packed}, DType::kFloat32});
    inputs.emplace("encoder_hidden_states",
                   Tensor{embeddings.data(), {1, config_.text_length, width}, DType::kFloat32});
    inputs.emplace("timestep_features", Tensor{features_copy.data(), {1, 16}, DType::kFloat32});
    inputs.emplace("text_mask", Tensor{mask.data(), {1, config_.text_length}, DType::kFloat32});
    return copy_output(transformer_->forward(inputs), "noise_pred", latents.size());
}

std::vector<float> ChromaPipeline::decode(std::vector<float> packed) {
    const auto shape = config_.latent_shape();
    auto latents = unpack_latents(packed, shape);
    to_vae_input(latents, config_.vae_scaling, config_.vae_shift, config_.bfloat16);
    TensorMap inputs;
    inputs.emplace(
        "latents",
        Tensor{latents.data(), {1, shape.channels, shape.height, shape.width}, DType::kFloat32});
    const auto pixels =
        static_cast<std::size_t>(config_.height) * static_cast<std::size_t>(config_.width);
    return copy_output(vae_decoder_->forward(inputs), "image", 3 * pixels);
}

internal::ImageResult ChromaPipeline::run(const internal::TextToImageRequest& request,
                                          internal::ConfigView config) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto begin = std::chrono::steady_clock::now();
    const bool replay = !request.initial_latents.empty();
    const auto settings = resolve(config, replay);
    const auto shape = config_.latent_shape();
    std::vector<float> latents;
    if (replay) {
        if (request.initial_latents.size() != shape.count())
            throw std::invalid_argument("Chroma initial latents must be packed [" +
                                        std::to_string(shape.tokens()) + ", " +
                                        std::to_string(4 * shape.channels) + "] float32");
        latents.assign(request.initial_latents.begin(), request.initial_latents.end());
        for (const float value : latents)
            if (!std::isfinite(value))
                throw std::invalid_argument("Chroma initial latents must be finite");
    } else {
        latents = initial_noise(settings.seed, shape);
    }
    to_model_type(latents, config_.bfloat16);

    const bool guided = settings.guidance > 1.0F;
    const auto prompt = encode(request.prompt);
    EncodedText negative;
    if (guided)
        negative = encode(settings.negative_prompt);

    const auto sigmas = flow_match_sigmas(settings.steps, config_.scheduler_shift);
    for (std::int32_t step = 0; step < settings.steps; ++step) {
        const float sigma = sigmas[static_cast<std::size_t>(step)];
        const float timestep = sigma * static_cast<float>(config_.num_train_timesteps);
        const auto features = timestep_features(conditioned_timestep(timestep, config_.bfloat16));
        auto velocity = denoise(latents, prompt, features);
        if (guided)
            apply_guidance(velocity, denoise(latents, negative, features), settings.guidance,
                           config_.bfloat16);
        euler_step(latents, velocity, sigma, sigmas[static_cast<std::size_t>(step) + 1],
                   config_.bfloat16);
    }

    internal::ImageResult result;
    result.pixels =
        to_rgb(decode(std::move(latents)), config_.height, config_.width, config_.bfloat16);
    result.height = config_.height;
    result.width = config_.width;
    result.channels = 3;
    result.num_frames = 1;
    const auto elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin);
    std::cerr << "[chroma] " << settings.steps << " steps, guidance " << settings.guidance << ", "
              << config_.width << "x" << config_.height << " in " << elapsed.count() << " ms\n";
    return result;
}

} // namespace trtmc::chroma

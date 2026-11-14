/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen_image2_1/runtime/runtime_config.h"

#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::qwen_image2_1 {
namespace {

// Sizes far beyond anything a bundle is built for; they keep every derived count inside int32.
constexpr std::int32_t kMaxImageSide = 32768;
constexpr std::int32_t kMaxTokens = 1 << 20;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(std::string("Qwen-Image 2.1 runtime.json: ") + message);
}

RuntimeConfig read_fields(const std::string& document) {
    const auto json = nlohmann::json::parse(document);
    require(json.at("tensor_parallel_size").get<std::int32_t>() == 1,
            "tensor_parallel_size must be 1");

    RuntimeConfig config;
    config.image_height = json.at("image_height").get<std::int32_t>();
    config.image_width = json.at("image_width").get<std::int32_t>();
    config.image_channels = json.at("image_channels").get<std::int32_t>();
    config.latent_channels = json.at("latent_channels").get<std::int32_t>();
    config.vae_scale_factor = json.at("vae_scale_factor").get<std::int32_t>();
    config.text_tokens = json.at("text_tokens").get<std::int32_t>();
    config.text_hidden_size = json.at("text_hidden_size").get<std::int32_t>();
    config.prefix_tokens = json.at("prefix_tokens").get<std::int32_t>();
    config.pad_token_id = json.at("pad_token_id").get<std::int32_t>();
    config.prompt_prefix = json.at("prompt_prefix").get<std::string>();
    config.prompt_suffix = json.at("prompt_suffix").get<std::string>();
    config.num_train_timesteps = json.at("num_train_timesteps").get<std::int32_t>();
    config.base_shift = json.at("base_shift").get<double>();
    config.max_shift = json.at("max_shift").get<double>();
    config.base_image_seq_len = json.at("base_image_seq_len").get<std::int32_t>();
    config.max_image_seq_len = json.at("max_image_seq_len").get<std::int32_t>();
    config.shift_terminal = json.at("shift_terminal").get<double>();
    config.default_num_steps = json.at("default_num_steps").get<std::int32_t>();
    return config;
}

} // namespace

RuntimeConfig parse_runtime_config(const std::string& document) {
    RuntimeConfig config;
    try {
        config = read_fields(document);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("Qwen-Image 2.1 runtime.json: ") + error.what());
    }

    require(config.vae_scale_factor > 0 && config.latent_channels > 0 &&
                config.latent_channels <= kMaxTokens &&
                (config.image_channels == 3 || config.image_channels == 4),
            "invalid latent or image channel layout");
    require(config.image_height > 0 && config.image_width > 0 &&
                config.image_height <= kMaxImageSide && config.image_width <= kMaxImageSide &&
                config.image_height % config.vae_scale_factor == 0 &&
                config.image_width % config.vae_scale_factor == 0,
            "image size must be a positive multiple of the VAE scale");
    require(config.text_tokens > 0 && config.text_tokens <= kMaxTokens &&
                config.text_hidden_size > 0 && config.text_hidden_size <= kMaxTokens &&
                config.prefix_tokens >= 0 && config.prefix_tokens <= kMaxTokens &&
                config.pad_token_id >= 0,
            "invalid text layout");
    // The pipeline divides the scheduler's timestep by the literal 1000; see reference_timestep.
    require(config.num_train_timesteps == 1000 &&
                config.max_image_seq_len > config.base_image_seq_len &&
                config.base_image_seq_len > 0 && std::isfinite(config.base_shift) &&
                std::isfinite(config.max_shift) && config.shift_terminal >= 0.0 &&
                config.shift_terminal < 1.0,
            "invalid scheduler settings");
    require(config.default_num_steps > 0, "default_num_steps must be positive");
    return config;
}

} // namespace trtmc::qwen_image2_1

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/stable_video_diffusion/runtime/runtime_config.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::stable_video_diffusion {

namespace {

std::array<float, 3> triple(const nlohmann::json& value) {
    const auto values = value.get<std::vector<float>>();
    if (values.size() != 3)
        throw std::runtime_error("stable_video_diffusion runtime.json needs three channel values");
    return {values[0], values[1], values[2]};
}

} // namespace

RuntimeConfig parse_runtime_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    RuntimeConfig config;
    config.precision = json.at("precision").get<std::string>();
    config.height = json.at("height").get<std::int32_t>();
    config.width = json.at("width").get<std::int32_t>();
    config.num_frames = json.at("num_frames").get<std::int32_t>();
    config.latent_channels = json.at("latent_channels").get<std::int32_t>();
    config.latent_height = json.at("latent_height").get<std::int32_t>();
    config.latent_width = json.at("latent_width").get<std::int32_t>();
    config.cross_attention_dim = json.at("cross_attention_dim").get<std::int32_t>();
    config.added_time_ids = json.at("added_time_ids").get<std::int32_t>();
    config.clip_image_size = json.at("clip_image_size").get<std::int32_t>();
    config.clip_image_mean = triple(json.at("clip_image_mean"));
    config.clip_image_std = triple(json.at("clip_image_std"));
    config.scaling_factor = json.at("scaling_factor").get<double>();
    config.sigma_min = json.at("sigma_min").get<double>();
    config.sigma_max = json.at("sigma_max").get<double>();
    config.init_noise_sigma_includes_unit = json.at("init_noise_sigma_includes_unit").get<bool>();
    const auto& defaults = json.at("defaults");
    config.default_num_steps = defaults.at("num_steps").get<std::int64_t>();
    config.default_min_guidance_scale = defaults.at("min_guidance_scale").get<double>();
    config.default_max_guidance_scale = defaults.at("max_guidance_scale").get<double>();
    config.default_fps = defaults.at("fps").get<std::int64_t>();
    config.default_motion_bucket_id = defaults.at("motion_bucket_id").get<std::int64_t>();
    config.default_noise_aug_strength = defaults.at("noise_aug_strength").get<double>();
    config.default_seed = defaults.at("seed").get<std::int64_t>();
    if (config.precision != "fp16" && config.precision != "fp32")
        throw std::runtime_error("stable_video_diffusion runtime.json names an unknown precision");
    if (config.height <= 0 || config.width <= 0 || config.num_frames <= 0 ||
        config.latent_channels <= 0 || config.latent_height * 8 != config.height ||
        config.latent_width * 8 != config.width || config.cross_attention_dim <= 0 ||
        config.added_time_ids != 3 || config.clip_image_size <= 1 ||
        !(config.scaling_factor > 0.0) || !(config.sigma_min > 0.0) ||
        !(config.sigma_max > config.sigma_min)) {
        throw std::runtime_error("stable_video_diffusion runtime.json does not match its contract");
    }
    return config;
}

} // namespace trtmc::stable_video_diffusion

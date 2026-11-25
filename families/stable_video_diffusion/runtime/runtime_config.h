/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace trtmc::stable_video_diffusion {

// runtime.json written by the family builder.
struct RuntimeConfig {
    std::string precision;
    std::int32_t height{0};
    std::int32_t width{0};
    std::int32_t num_frames{0};
    std::int32_t latent_channels{0};
    std::int32_t latent_height{0};
    std::int32_t latent_width{0};
    std::int32_t cross_attention_dim{0};
    std::int32_t added_time_ids{0};
    std::int32_t clip_image_size{0};
    std::array<float, 3> clip_image_mean{};
    std::array<float, 3> clip_image_std{};
    double scaling_factor{0.0};
    double sigma_min{0.0};
    double sigma_max{0.0};
    bool init_noise_sigma_includes_unit{true};
    // StableVideoDiffusionPipeline.__call__ defaults, owned by the builder.
    std::int64_t default_num_steps{0};
    double default_min_guidance_scale{0.0};
    double default_max_guidance_scale{0.0};
    std::int64_t default_fps{0};
    std::int64_t default_motion_bucket_id{0};
    double default_noise_aug_strength{0.0};
    std::int64_t default_seed{0};
};

RuntimeConfig parse_runtime_config(const std::vector<char>& json);

} // namespace trtmc::stable_video_diffusion

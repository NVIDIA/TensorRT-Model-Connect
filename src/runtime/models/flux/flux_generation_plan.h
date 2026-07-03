/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/models/flux/flux_diffusion_types.h"
#include "runtime/models/flux/flux_scheduler_helpers.h"

#include <cstddef>
#include <cstdint>

namespace trtmc {
namespace diffusion {

struct FluxPackLayout {
    int32_t ph{2};
    int32_t pw{2};
    int32_t packed_channels{0};
    int32_t h_packed{0};
    int32_t w_packed{0};
};

inline FluxPackLayout make_flux_pack_layout(const FluxDiffusionConfig& config, int32_t z_dim,
                                            int32_t h_lat, int32_t w_lat) {
    FluxPackLayout layout;
    if (config.patch_size.size() >= 3) {
        layout.ph = config.patch_size[1];
        layout.pw = config.patch_size[2];
    }
    layout.packed_channels = z_dim * layout.ph * layout.pw;
    layout.h_packed = h_lat / layout.ph;
    layout.w_packed = w_lat / layout.pw;
    return layout;
}

struct FluxGenerationPlan {
    int32_t num_inference_steps{0};
    float guidance_scale{0.0F};
    int32_t dit_dim{0};
    int32_t text_seq{0};
    int32_t z_dim{0};
    FluxPackLayout layout;
    bool is_flux2{false};
    std::size_t latent_size{0};
    flux_scheduler::FlowMatchEulerConfig scheduler_config;
};

inline bool apply_flux_initial_latents(std::size_t expected_size,
                                       const std::vector<float>& supplied,
                                       std::vector<float>& latents, std::string& error) {
    if (supplied.empty()) {
        return true;
    }
    if (supplied.size() != expected_size) {
        error = "Flux initial latents contain " + std::to_string(supplied.size()) +
                " floats; expected " + std::to_string(expected_size);
        return false;
    }
    latents = supplied;
    return true;
}

inline FluxGenerationPlan make_flux_generation_plan(const FluxDiffusionConfig& config,
                                                    const FluxPreprocessorWeights& weights,
                                                    int32_t requested_steps,
                                                    float requested_guidance, int32_t h_lat,
                                                    int32_t w_lat, int32_t num_img_tokens) {
    FluxGenerationPlan plan;
    plan.num_inference_steps =
        flux_scheduler::resolve_requested_steps(requested_steps, config.num_inference_steps, true);
    plan.guidance_scale =
        flux_scheduler::resolve_requested_guidance(requested_guidance, config.guidance_scale);
    plan.dit_dim = config.dit_dim;
    plan.text_seq = config.text_seq_len;
    plan.z_dim = config.z_dim;
    plan.layout = make_flux_pack_layout(config, plan.z_dim, h_lat, w_lat);
    plan.is_flux2 = !weights.vae_bn_mean.empty();
    plan.latent_size = plan.is_flux2
                           ? (static_cast<std::size_t>(plan.layout.packed_channels) *
                              static_cast<std::size_t>(plan.layout.h_packed) *
                              static_cast<std::size_t>(plan.layout.w_packed))
                           : (static_cast<std::size_t>(plan.z_dim) *
                              static_cast<std::size_t>(h_lat) * static_cast<std::size_t>(w_lat));

    plan.scheduler_config.num_train_timesteps = 1000;
    plan.scheduler_config.shift = config.flow_shift;
    plan.scheduler_config.use_dynamic_shifting = config.use_dynamic_shifting;
    plan.scheduler_config.base_shift = config.base_shift;
    plan.scheduler_config.max_shift = config.max_shift;
    plan.scheduler_config.base_image_seq_len = config.base_image_seq_len;
    plan.scheduler_config.max_image_seq_len = config.max_image_seq_len;
    plan.scheduler_config.shift_terminal = config.shift_terminal;
    plan.scheduler_config.image_seq_len = num_img_tokens;
    plan.scheduler_config.use_empirical_mu = plan.is_flux2;
    return plan;
}

inline flux_scheduler::FlowMatchEulerState
make_flux_scheduler_state(const FluxGenerationPlan& plan) {
    flux_scheduler::FlowMatchEulerState scheduler;
    scheduler.num_train_timesteps = plan.scheduler_config.num_train_timesteps;
    scheduler.shift = plan.scheduler_config.shift;
    scheduler.use_dynamic_shifting = plan.scheduler_config.use_dynamic_shifting;
    scheduler.base_shift = plan.scheduler_config.base_shift;
    scheduler.max_shift = plan.scheduler_config.max_shift;
    scheduler.base_image_seq_len = plan.scheduler_config.base_image_seq_len;
    scheduler.max_image_seq_len = plan.scheduler_config.max_image_seq_len;
    scheduler.shift_terminal = plan.scheduler_config.shift_terminal;
    scheduler.image_seq_len = plan.scheduler_config.image_seq_len;
    scheduler.use_empirical_mu = plan.scheduler_config.use_empirical_mu;
    scheduler.use_zero_sigma_min = plan.scheduler_config.use_zero_sigma_min;
    scheduler.set_timesteps(plan.num_inference_steps);
    return scheduler;
}

} // namespace diffusion
} // namespace trtmc

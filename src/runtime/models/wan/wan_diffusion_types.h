/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trtmc {

struct WanDiffusionConfig {
    std::string scheduler{"flow_match_euler"};
    int32_t num_inference_steps{50};
    float guidance_scale{5.0F};
    float flow_shift{1.0F};
    bool use_dynamic_shifting{false};
    float base_shift{0.5F};
    float max_shift{1.15F};
    int32_t base_image_seq_len{256};
    int32_t max_image_seq_len{4096};
    float shift_terminal{0.0F};

    int32_t video_height{0};
    int32_t video_width{0};
    int32_t video_num_frames{0};

    int32_t z_dim{0};
    int32_t scale_factor_temporal{0};
    int32_t scale_factor_spatial{0};
    int32_t dit_dim{0};
    int32_t dit_num_heads{0};
    int32_t freq_dim{0};
    int32_t text_seq_len{0};
    int32_t text_encoder_dim{0};

    int32_t num_vae_caches{0};
    std::vector<float> latents_mean;
    std::vector<float> latents_std;
    std::vector<int32_t> patch_size;
    std::vector<int32_t> axes_dims_rope;
    float rope_theta{10000.0F};
    std::string vae_model_id;

    bool guidance_embeds{false};
    bool use_rope{true};
    float vae_scaling_factor{0.0F};

    std::string diffusion_backend_type;

    int32_t batch_size{1};

    struct {
        int32_t dit{1};
        int32_t text_encoder{1};
        int32_t vae{1};
    } max_batch_size;
};

struct WanPreprocessorWeights {
    std::vector<float> patch_embed_weight;
    std::vector<float> patch_embed_bias;
    int32_t patch_dim{0};

    std::vector<float> time_emb_0_weight;
    std::vector<float> time_emb_0_bias;
    std::vector<float> time_emb_2_weight;
    std::vector<float> time_emb_2_bias;

    std::vector<float> time_proj_weight;
    std::vector<float> time_proj_bias;

    std::vector<float> text_proj_weight;
    std::vector<float> text_proj_bias;
    std::vector<float> text_proj_2_weight;
    std::vector<float> text_proj_2_bias;

    std::vector<float> context_embed_weight;
    std::vector<float> context_embed_bias;

    std::vector<float> guidance_emb_0_weight;
    std::vector<float> guidance_emb_0_bias;
    std::vector<float> guidance_emb_2_weight;
    std::vector<float> guidance_emb_2_bias;

    std::vector<float> vae_bn_mean;
    std::vector<float> vae_bn_var;

    bool valid{false};
};

struct WanVideoResult {
    std::vector<float> frames;
    int32_t num_frames{0};
    int32_t height{0};
    int32_t width{0};
};

} // namespace trtmc

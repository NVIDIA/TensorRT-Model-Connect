/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
// =============================================================================
// qwen_image_types.h — Qwen-Image runtime configuration types.
// =============================================================================
//
// Mirrors the bundle config.json schema produced by
// tensorrt_model_connect.qwen_image_bundle_config.build_bundle_config().
// Each top-level JSON section (diffusion, text_encoder, denoiser, vae,
// image, tokenizer) maps onto a dedicated struct, and
// QwenImageConfig::parse() walks the JSON blob to populate the aggregate.
//
// Missing fields fall back to defaults so partial / minimal configs (e.g.
// Edit-mode stubs that only set has_encoder=true) still parse cleanly.
//
// Trace: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01.
// =============================================================================

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trtmc {

enum class QwenImageTaskMode {
    T2I,
    Edit,
};

struct QwenImageDiffusionConfig {
    std::string scheduler{"flow_match_euler"};
    int32_t num_train_timesteps{1000};
    float shift{1.0F};
    bool use_dynamic_shifting{false};
    float base_shift{0.5F};
    float max_shift{0.9F};
    int32_t base_image_seq_len{256};
    int32_t max_image_seq_len{8192};
    float shift_terminal{0.0F};
    std::string time_shift_type; // "exponential" or empty (static)
    int32_t default_num_inference_steps{50};
    float default_cfg_scale{4.0F};
    std::string default_negative_prompt; // " "
};

struct QwenImageDenoiserConfig {
    std::string type{"qwen_image_mmdit"};
    int32_t in_channels{64};
    int32_t out_channels{16};
    int32_t patch_size{2};
    int32_t hidden_size{3072};
    int32_t num_joint_blocks{60};
    int32_t num_single_blocks{0};
    int32_t num_attention_heads{24};
    int32_t attention_head_dim{128};
    std::vector<int32_t> rope_axes_dim; // {16, 56, 56}
    float rope_theta{10000.0F};
    int32_t text_embed_dim{3584};
    bool guidance_embeds{false};
    int32_t max_image_tokens{8192};
    int32_t max_text_tokens{1024};
};

struct QwenImageVAEConfig {
    std::string type{"autoencoder_kl_qwen_image"};
    int32_t latent_channels{16};
    int32_t spatial_scale_factor{8};
    int32_t base_dim{96};
    std::vector<int32_t> dim_mult;         // {1, 2, 4, 4}
    std::vector<bool> temporal_downsample; // {false, true, true}
    std::vector<float> latents_mean;       // length-latent_channels (16)
    std::vector<float> latents_std;        // length-latent_channels (16)
    bool has_encoder{false};
    bool has_decoder{true};
};

struct QwenImageImageConfig {
    int32_t default_height{1024};
    int32_t default_width{1024};
    int32_t min_height{256};
    int32_t min_width{256};
    int32_t max_height{2048};
    int32_t max_width{2048};
    int32_t height_alignment{16};
    int32_t width_alignment{16};
};

struct QwenImageTokenizerConfig {
    std::string class_name{"Qwen2Tokenizer"}; // "class" in JSON (C++ reserved word)
    std::string prompt_template_kind;
    int32_t prompt_template_drop_idx{0};
    int32_t tokenizer_max_length{1024};
    bool add_special_tokens{false};
};

struct QwenImageTextEncoderConfig {
    std::string type{"qwen2_5_vl_lm"};
    int32_t hidden_size{3584};
    int32_t num_layers{28};
    int32_t num_heads{28};
    int32_t num_kv_heads{4};
    int32_t head_dim{128};
    int32_t intermediate_size{18944};
    int32_t vocab_size{152064};
    float rope_theta{1000000.0F};
    float rms_norm_eps{1e-6F};
    int32_t max_seq_len{1024};
    int32_t extract_hidden_state_layer{-1};
    bool apply_final_norm{true};
    std::string tokenizer_template_kind;
};

struct QwenImageVisionEncoderConfig {
    std::string type{"qwen2_5_vl_vision"};
    int32_t image_size{384};
    int32_t image_height{0};
    int32_t image_width{0};
    int32_t patch_size{14};
    int32_t merge_size{2};
    int32_t hidden_size{1280};
    int32_t num_layers{32};
    int32_t out_hidden_size{3584};
};

struct QwenImageConditioningConfig {
    int32_t vl_image_size{384};
    int32_t vae_image_size{1024};
    int32_t vae_image_height{0};
    int32_t vae_image_width{0};
    std::string vae_concat_axis{"sequence"};
    int32_t max_input_images{1};
};

struct QwenImageConfig {
    std::string engine_backend;
    std::string runtime_strategy;
    std::string model_family;
    std::string model_variant;
    QwenImageTaskMode task_mode{QwenImageTaskMode::T2I};

    QwenImageDiffusionConfig diffusion;
    QwenImageTextEncoderConfig text_encoder;
    QwenImageDenoiserConfig denoiser;
    QwenImageVAEConfig vae;
    QwenImageImageConfig image;
    QwenImageTokenizerConfig tokenizer;
    QwenImageVisionEncoderConfig vision_encoder;
    QwenImageConditioningConfig image_conditioning;

    // Engine batch envelope sourced from the bundle's `max_batch_size`
    // top-level block. Defaults to {1,1,1} so legacy bundles preserve B=1
    // semantics. Populated by QwenImagePlugin from `bundle.info`, not from
    // config_json (the block lives at the bundle header level).
    struct {
        int32_t dit{1};
        int32_t text_encoder{1};
        int32_t vae{1};
    } max_batch_size;

    // Parse a bundle config.json blob into a QwenImageConfig. Missing keys
    // / sections retain their struct defaults.
    static QwenImageConfig parse(std::string_view config_json);
};

// Preprocessor weights blob for Qwen-Image.
// Forward-declared here so downstream pipeline code can refer to the type;
// the parser implementation lives next to the preprocessor weights helpers.
struct QwenImagePreprocessorWeights {
    std::vector<float> latents_mean; // length-latent_channels
    std::vector<float> latents_std;  // length-latent_channels
    bool valid{false};
};

// Parse a Qwen-Image preprocessor_weights bundle section into the struct.
//
// The blob format is the canonical diffusion preprocessor wire format:
//   <u32 LE index_len><UTF-8 JSON index><raw float32 payload>
//
// Recognized index entries:
//   - "latents_mean": float[16] — per-channel VAE latent mean.
//   - "latents_std":  float[16] — per-channel VAE latent std.
//
// Both vectors must be present for valid=true. Missing keys leave the
// corresponding vector empty (graceful degradation — the caller can
// decide whether that is fatal for its pipeline mode).
//
// Trace: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01.
QwenImagePreprocessorWeights parse_qwen_image_preprocessor_weights(const std::vector<char>& data);

} // namespace trtmc

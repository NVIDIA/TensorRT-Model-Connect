/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// qwen_image_types.cpp — QwenImageConfig::parse() implementation.
// =============================================================================
//
// Walks a bundle config.json blob and populates the nested QwenImageConfig
// struct hierarchy. Per-section parsing is scoped via
// extract_json_object_text() so sibling keys with the same name (e.g.
// "hidden_size" in both text_encoder and denoiser, or "rope_theta" in both
// text_encoder and denoiser, or "type" in text_encoder/denoiser/vae) do not
// collide.
//
// Trace: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01.
// =============================================================================

#include "qwen_image_types.h"

#include "preprocessor_weights_helpers.h"
#include "utils/json_helpers.h"

#include <cstddef>
#include <iostream>
#include <string>

namespace trtmc {
namespace {

QwenImageTaskMode parse_task_mode(const std::string& raw) {
    if (raw == "edit") {
        return QwenImageTaskMode::Edit;
    }
    // "t2i" or anything else defaults to T2I (matches Python default).
    return QwenImageTaskMode::T2I;
}

void parse_diffusion(const std::string& obj, QwenImageDiffusionConfig& dc) {
    if (obj.empty())
        return;
    dc.scheduler = extract_json_string(obj, "scheduler", dc.scheduler);
    dc.num_train_timesteps = extract_json_int(obj, "num_train_timesteps", dc.num_train_timesteps);
    dc.shift = extract_json_float(obj, "shift", dc.shift);
    dc.use_dynamic_shifting =
        extract_json_bool(obj, "use_dynamic_shifting", dc.use_dynamic_shifting);
    dc.base_shift = extract_json_float(obj, "base_shift", dc.base_shift);
    dc.max_shift = extract_json_float(obj, "max_shift", dc.max_shift);
    dc.base_image_seq_len = extract_json_int(obj, "base_image_seq_len", dc.base_image_seq_len);
    dc.max_image_seq_len = extract_json_int(obj, "max_image_seq_len", dc.max_image_seq_len);
    dc.shift_terminal = extract_json_float(obj, "shift_terminal", dc.shift_terminal);
    dc.time_shift_type = extract_json_string(obj, "time_shift_type", dc.time_shift_type);
    dc.default_num_inference_steps =
        extract_json_int(obj, "default_num_inference_steps", dc.default_num_inference_steps);
    dc.default_cfg_scale = extract_json_float(obj, "default_cfg_scale", dc.default_cfg_scale);
    dc.default_negative_prompt =
        extract_json_string(obj, "default_negative_prompt", dc.default_negative_prompt);
}

void parse_text_encoder(const std::string& obj, QwenImageTextEncoderConfig& tc) {
    if (obj.empty())
        return;
    tc.type = extract_json_string(obj, "type", tc.type);
    tc.hidden_size = extract_json_int(obj, "hidden_size", tc.hidden_size);
    tc.num_layers = extract_json_int(obj, "num_layers", tc.num_layers);
    tc.num_heads = extract_json_int(obj, "num_heads", tc.num_heads);
    tc.num_kv_heads = extract_json_int(obj, "num_kv_heads", tc.num_kv_heads);
    tc.head_dim = extract_json_int(obj, "head_dim", tc.head_dim);
    tc.intermediate_size = extract_json_int(obj, "intermediate_size", tc.intermediate_size);
    tc.vocab_size = extract_json_int(obj, "vocab_size", tc.vocab_size);
    tc.rope_theta = extract_json_float(obj, "rope_theta", tc.rope_theta);
    tc.rms_norm_eps = extract_json_float(obj, "rms_norm_eps", tc.rms_norm_eps);
    tc.max_seq_len = extract_json_int(obj, "max_seq_len", tc.max_seq_len);
    tc.extract_hidden_state_layer =
        extract_json_int(obj, "extract_hidden_state_layer", tc.extract_hidden_state_layer);
    tc.apply_final_norm = extract_json_bool(obj, "apply_final_norm", tc.apply_final_norm);
    tc.tokenizer_template_kind =
        extract_json_string(obj, "tokenizer_template_kind", tc.tokenizer_template_kind);
}

void parse_denoiser(const std::string& obj, QwenImageDenoiserConfig& dn) {
    if (obj.empty())
        return;
    dn.type = extract_json_string(obj, "type", dn.type);
    dn.in_channels = extract_json_int(obj, "in_channels", dn.in_channels);
    dn.out_channels = extract_json_int(obj, "out_channels", dn.out_channels);
    dn.patch_size = extract_json_int(obj, "patch_size", dn.patch_size);
    dn.hidden_size = extract_json_int(obj, "hidden_size", dn.hidden_size);
    dn.num_joint_blocks = extract_json_int(obj, "num_joint_blocks", dn.num_joint_blocks);
    dn.num_single_blocks = extract_json_int(obj, "num_single_blocks", dn.num_single_blocks);
    dn.num_attention_heads = extract_json_int(obj, "num_attention_heads", dn.num_attention_heads);
    dn.attention_head_dim = extract_json_int(obj, "attention_head_dim", dn.attention_head_dim);
    auto axes = extract_json_int_array(obj, "rope_axes_dim", 8);
    if (!axes.empty()) {
        dn.rope_axes_dim = std::move(axes);
    }
    dn.rope_theta = extract_json_float(obj, "rope_theta", dn.rope_theta);
    dn.text_embed_dim = extract_json_int(obj, "text_embed_dim", dn.text_embed_dim);
    dn.guidance_embeds = extract_json_bool(obj, "guidance_embeds", dn.guidance_embeds);
    dn.max_image_tokens = extract_json_int(obj, "max_image_tokens", dn.max_image_tokens);
    dn.max_text_tokens = extract_json_int(obj, "max_text_tokens", dn.max_text_tokens);
}

void parse_vae(const std::string& obj, QwenImageVAEConfig& vc) {
    if (obj.empty())
        return;
    vc.type = extract_json_string(obj, "type", vc.type);
    vc.latent_channels = extract_json_int(obj, "latent_channels", vc.latent_channels);
    vc.spatial_scale_factor =
        extract_json_int(obj, "spatial_scale_factor", vc.spatial_scale_factor);
    vc.base_dim = extract_json_int(obj, "base_dim", vc.base_dim);
    auto dim_mult = extract_json_int_array(obj, "dim_mult", 16);
    if (!dim_mult.empty()) {
        vc.dim_mult = std::move(dim_mult);
    }
    auto td = extract_json_bool_array(obj, "temporal_downsample", 16);
    if (!td.empty()) {
        vc.temporal_downsample = std::move(td);
    }
    // latents_mean / latents_std may have up to latent_channels entries (16
    // by default). Cap at 64 to be safe for any future variants.
    auto lm = extract_json_float_array(obj, "latents_mean", 64);
    if (!lm.empty()) {
        vc.latents_mean = std::move(lm);
    }
    auto ls = extract_json_float_array(obj, "latents_std", 64);
    if (!ls.empty()) {
        vc.latents_std = std::move(ls);
    }
    vc.has_encoder = extract_json_bool(obj, "has_encoder", vc.has_encoder);
    vc.has_decoder = extract_json_bool(obj, "has_decoder", vc.has_decoder);
}

void parse_image(const std::string& obj, QwenImageImageConfig& ic) {
    if (obj.empty())
        return;
    ic.default_height = extract_json_int(obj, "default_height", ic.default_height);
    ic.default_width = extract_json_int(obj, "default_width", ic.default_width);
    ic.min_height = extract_json_int(obj, "min_height", ic.min_height);
    ic.min_width = extract_json_int(obj, "min_width", ic.min_width);
    ic.max_height = extract_json_int(obj, "max_height", ic.max_height);
    ic.max_width = extract_json_int(obj, "max_width", ic.max_width);
    ic.height_alignment = extract_json_int(obj, "height_alignment", ic.height_alignment);
    ic.width_alignment = extract_json_int(obj, "width_alignment", ic.width_alignment);
}

void parse_tokenizer(const std::string& obj, QwenImageTokenizerConfig& tk) {
    if (obj.empty())
        return;
    tk.kind = extract_json_string(obj, "kind", tk.kind);
    tk.class_name = extract_json_string(obj, "class", tk.class_name);
    tk.prompt_template_kind =
        extract_json_string(obj, "prompt_template_kind", tk.prompt_template_kind);
    tk.prompt_template_drop_idx =
        extract_json_int(obj, "prompt_template_drop_idx", tk.prompt_template_drop_idx);
    tk.tokenizer_max_length =
        extract_json_int(obj, "tokenizer_max_length", tk.tokenizer_max_length);
    tk.add_special_tokens = extract_json_bool(obj, "add_special_tokens", tk.add_special_tokens);
}

void parse_vision_encoder(const std::string& obj, QwenImageVisionEncoderConfig& vc) {
    if (obj.empty())
        return;
    vc.type = extract_json_string(obj, "type", vc.type);
    vc.image_size = extract_json_int(obj, "image_size", vc.image_size);
    vc.image_height = extract_json_int(obj, "image_height", vc.image_height);
    vc.image_width = extract_json_int(obj, "image_width", vc.image_width);
    vc.patch_size = extract_json_int(obj, "patch_size", vc.patch_size);
    vc.merge_size = extract_json_int(obj, "merge_size", vc.merge_size);
    vc.hidden_size = extract_json_int(obj, "hidden_size", vc.hidden_size);
    vc.num_layers = extract_json_int(obj, "num_layers", vc.num_layers);
    vc.out_hidden_size = extract_json_int(obj, "out_hidden_size", vc.out_hidden_size);
}

void parse_image_conditioning(const std::string& obj, QwenImageConditioningConfig& ic) {
    if (obj.empty())
        return;
    ic.vl_image_size = extract_json_int(obj, "vl_image_size", ic.vl_image_size);
    ic.vae_image_size = extract_json_int(obj, "vae_image_size", ic.vae_image_size);
    ic.vae_image_height = extract_json_int(obj, "vae_image_height", ic.vae_image_height);
    ic.vae_image_width = extract_json_int(obj, "vae_image_width", ic.vae_image_width);
    ic.vae_concat_axis = extract_json_string(obj, "vae_concat_axis", ic.vae_concat_axis);
    ic.max_input_images = extract_json_int(obj, "max_input_images", ic.max_input_images);
}

} // namespace

QwenImageConfig QwenImageConfig::parse(std::string_view config_json) {
    QwenImageConfig cfg;
    const std::string text(config_json);

    // Top-level fields (sibling to the nested sections — flat lookup is safe
    // because these keys do not appear inside any nested section).
    cfg.engine_backend = extract_json_string(text, "engine_backend", cfg.engine_backend);
    cfg.runtime_strategy = extract_json_string(text, "runtime_strategy", cfg.runtime_strategy);
    cfg.model_family = extract_json_string(text, "model_family", cfg.model_family);
    cfg.model_variant = extract_json_string(text, "model_variant", cfg.model_variant);
    const std::string task_mode_raw = extract_json_string(text, "task_mode", "t2i");
    cfg.task_mode = parse_task_mode(task_mode_raw);

    // Nested sections: scope each one via its object text to avoid key
    // collisions between siblings (e.g. "type", "hidden_size", "rope_theta").
    parse_diffusion(extract_json_object_text(text, "diffusion"), cfg.diffusion);
    parse_text_encoder(extract_json_object_text(text, "text_encoder"), cfg.text_encoder);
    parse_denoiser(extract_json_object_text(text, "denoiser"), cfg.denoiser);
    parse_vae(extract_json_object_text(text, "vae"), cfg.vae);
    parse_image(extract_json_object_text(text, "image"), cfg.image);
    parse_tokenizer(extract_json_object_text(text, "tokenizer"), cfg.tokenizer);
    parse_vision_encoder(extract_json_object_text(text, "vision_encoder"), cfg.vision_encoder);
    parse_image_conditioning(extract_json_object_text(text, "image_conditioning"),
                             cfg.image_conditioning);

    return cfg;
}

// -----------------------------------------------------------------------------
// Preprocessor weights parser.
// -----------------------------------------------------------------------------
//
// The blob carries latents_mean (float[16]) and latents_std (float[16]) for
// the Qwen-Image VAE. Format is the Qwen Image preprocessor wire layout. We
// reuse the local extract_preprocessor_index + load_preprocessor_floats helpers
// so the C++ parser mirrors Python's
// pack_qwen_image_preprocessor_weights() exactly.
//
// On a malformed header (blob < 4 bytes or index length overflow) the helper
// reports the failure to stderr and we return a default-constructed struct
// (valid=false). On a partial blob (one missing key), the present vector is
// populated and the missing one remains empty.
QwenImagePreprocessorWeights parse_qwen_image_preprocessor_weights(const std::vector<char>& data) {
    QwenImagePreprocessorWeights w;

    std::string index_json;
    const char* blob = nullptr;
    std::size_t blob_size = 0;
    if (!qwen_image_preprocessor_weights::extract_preprocessor_index(data, index_json, blob,
                                                                     blob_size)) {
        return w;
    }

    const bool got_mean = qwen_image_preprocessor_weights::load_preprocessor_floats(
        index_json, blob, blob_size, "latents_mean", w.latents_mean);
    const bool got_std = qwen_image_preprocessor_weights::load_preprocessor_floats(
        index_json, blob, blob_size, "latents_std", w.latents_std);

    w.valid = got_mean && got_std;

    std::cerr << "[qwen-image] Preprocessor weights: " << (w.valid ? "OK" : "INCOMPLETE")
              << " (latents_mean=" << w.latents_mean.size()
              << ", latents_std=" << w.latents_std.size() << ")\n";
    return w;
}

} // namespace trtmc

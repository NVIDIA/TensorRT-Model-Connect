/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/qwen_image/runtime/qwen_image_types.h"

#include "preprocessor_weights_helpers.h"

#include <cstddef>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {
namespace {

const nlohmann::json& require_object(const nlohmann::json& parent, const char* key) {
    const auto& value = parent.at(key);
    if (!value.is_object())
        throw std::runtime_error(std::string("Qwen-Image '") + key + "' must be an object");
    return value;
}

int32_t require_int(const nlohmann::json& object, const char* key) {
    const auto& value = object.at(key);
    if (!value.is_number_integer() && !value.is_number_unsigned())
        throw std::runtime_error(std::string("Qwen-Image '") + key + "' must be an integer");
    return value.get<int32_t>();
}

float require_number(const nlohmann::json& object, const char* key) {
    const auto& value = object.at(key);
    if (!value.is_number())
        throw std::runtime_error(std::string("Qwen-Image '") + key + "' must be numeric");
    return value.get<float>();
}

std::string require_string(const nlohmann::json& object, const char* key) {
    const auto& value = object.at(key);
    if (!value.is_string())
        throw std::runtime_error(std::string("Qwen-Image '") + key + "' must be a string");
    return value.get<std::string>();
}

bool require_bool(const nlohmann::json& object, const char* key) {
    const auto& value = object.at(key);
    if (!value.is_boolean())
        throw std::runtime_error(std::string("Qwen-Image '") + key + "' must be a boolean");
    return value.get<bool>();
}

std::vector<int32_t> require_int_array(const nlohmann::json& object, const char* key) {
    const auto& value = object.at(key);
    if (!value.is_array())
        throw std::runtime_error(std::string("Qwen-Image '") + key + "' must be an array");
    std::vector<int32_t> result;
    result.reserve(value.size());
    for (const auto& element : value) {
        if (!element.is_number_integer() && !element.is_number_unsigned())
            throw std::runtime_error(std::string("Qwen-Image '") + key + "' must contain integers");
        result.push_back(element.get<int32_t>());
    }
    return result;
}

std::vector<float> require_number_array(const nlohmann::json& object, const char* key) {
    const auto& value = object.at(key);
    if (!value.is_array())
        throw std::runtime_error(std::string("Qwen-Image '") + key + "' must be an array");
    std::vector<float> result;
    result.reserve(value.size());
    for (const auto& element : value) {
        if (!element.is_number())
            throw std::runtime_error(std::string("Qwen-Image '") + key + "' must contain numbers");
        result.push_back(element.get<float>());
    }
    return result;
}

std::vector<bool> require_bool_array(const nlohmann::json& object, const char* key) {
    const auto& value = object.at(key);
    if (!value.is_array())
        throw std::runtime_error(std::string("Qwen-Image '") + key + "' must be an array");
    std::vector<bool> result;
    result.reserve(value.size());
    for (const auto& element : value) {
        if (!element.is_boolean())
            throw std::runtime_error(std::string("Qwen-Image '") + key + "' must contain booleans");
        result.push_back(element.get<bool>());
    }
    return result;
}

QwenImageTaskMode parse_task_mode(const std::string& raw) {
    if (raw == "t2i")
        return QwenImageTaskMode::T2I;
    if (raw == "edit")
        return QwenImageTaskMode::Edit;
    throw std::runtime_error("Qwen-Image task_mode must be 't2i' or 'edit'");
}

void parse_diffusion(const nlohmann::json& object, QwenImageDiffusionConfig& config) {
    config.scheduler = require_string(object, "scheduler");
    config.num_train_timesteps = require_int(object, "num_train_timesteps");
    config.shift = require_number(object, "shift");
    config.use_dynamic_shifting = require_bool(object, "use_dynamic_shifting");
    config.base_shift = require_number(object, "base_shift");
    config.max_shift = require_number(object, "max_shift");
    config.base_image_seq_len = require_int(object, "base_image_seq_len");
    config.max_image_seq_len = require_int(object, "max_image_seq_len");
    config.shift_terminal = require_number(object, "shift_terminal");
    config.time_shift_type = require_string(object, "time_shift_type");
    config.default_num_inference_steps = require_int(object, "default_num_inference_steps");
    config.default_cfg_scale = require_number(object, "default_cfg_scale");
    config.default_negative_prompt = require_string(object, "default_negative_prompt");
}

void parse_text_encoder(const nlohmann::json& object, QwenImageTextEncoderConfig& config) {
    config.type = require_string(object, "type");
    config.hidden_size = require_int(object, "hidden_size");
    config.num_layers = require_int(object, "num_layers");
    config.num_heads = require_int(object, "num_heads");
    config.num_kv_heads = require_int(object, "num_kv_heads");
    config.head_dim = require_int(object, "head_dim");
    config.intermediate_size = require_int(object, "intermediate_size");
    config.vocab_size = require_int(object, "vocab_size");
    config.rope_theta = require_number(object, "rope_theta");
    config.rms_norm_eps = require_number(object, "rms_norm_eps");
    config.max_seq_len = require_int(object, "max_seq_len");
    config.extract_hidden_state_layer = require_int(object, "extract_hidden_state_layer");
    config.apply_final_norm = require_bool(object, "apply_final_norm");
    config.tokenizer_template_kind = require_string(object, "tokenizer_template_kind");
}

void parse_denoiser(const nlohmann::json& object, QwenImageDenoiserConfig& config) {
    config.type = require_string(object, "type");
    config.in_channels = require_int(object, "in_channels");
    config.out_channels = require_int(object, "out_channels");
    config.patch_size = require_int(object, "patch_size");
    config.hidden_size = require_int(object, "hidden_size");
    config.num_joint_blocks = require_int(object, "num_joint_blocks");
    config.num_single_blocks = require_int(object, "num_single_blocks");
    config.num_attention_heads = require_int(object, "num_attention_heads");
    config.attention_head_dim = require_int(object, "attention_head_dim");
    config.rope_axes_dim = require_int_array(object, "rope_axes_dim");
    config.rope_theta = require_number(object, "rope_theta");
    config.text_embed_dim = require_int(object, "text_embed_dim");
    config.guidance_embeds = require_bool(object, "guidance_embeds");
    config.max_image_tokens = require_int(object, "max_image_tokens");
    config.max_text_tokens = require_int(object, "max_text_tokens");
}

void parse_vae(const nlohmann::json& object, QwenImageVAEConfig& config) {
    config.type = require_string(object, "type");
    config.latent_channels = require_int(object, "latent_channels");
    config.spatial_scale_factor = require_int(object, "spatial_scale_factor");
    config.base_dim = require_int(object, "base_dim");
    config.dim_mult = require_int_array(object, "dim_mult");
    config.temporal_downsample = require_bool_array(object, "temporal_downsample");
    config.latents_mean = require_number_array(object, "latents_mean");
    config.latents_std = require_number_array(object, "latents_std");
    config.has_encoder = require_bool(object, "has_encoder");
    config.has_decoder = require_bool(object, "has_decoder");
}

void parse_image(const nlohmann::json& object, QwenImageImageConfig& config) {
    config.default_height = require_int(object, "default_height");
    config.default_width = require_int(object, "default_width");
    config.min_height = require_int(object, "min_height");
    config.min_width = require_int(object, "min_width");
    config.max_height = require_int(object, "max_height");
    config.max_width = require_int(object, "max_width");
    config.height_alignment = require_int(object, "height_alignment");
    config.width_alignment = require_int(object, "width_alignment");
}

void parse_tokenizer(const nlohmann::json& object, QwenImageTokenizerConfig& config) {
    config.prompt_template_drop_idx = require_int(object, "prompt_template_drop_idx");
}

void parse_vision_encoder(const nlohmann::json& object, QwenImageVisionEncoderConfig& config) {
    config.type = require_string(object, "type");
    config.image_size = require_int(object, "image_size");
    config.image_height = require_int(object, "image_height");
    config.image_width = require_int(object, "image_width");
    config.patch_size = require_int(object, "patch_size");
    config.merge_size = require_int(object, "merge_size");
    config.hidden_size = require_int(object, "hidden_size");
    config.num_layers = require_int(object, "num_layers");
    config.out_hidden_size = require_int(object, "out_hidden_size");
}

void parse_image_conditioning(const nlohmann::json& object, QwenImageConditioningConfig& config) {
    config.vl_image_size = require_int(object, "vl_image_size");
    config.vae_image_size = require_int(object, "vae_image_size");
    config.vae_image_height = require_int(object, "vae_image_height");
    config.vae_image_width = require_int(object, "vae_image_width");
    config.vae_concat_axis = require_string(object, "vae_concat_axis");
    config.max_input_images = require_int(object, "max_input_images");
}

} // namespace

QwenImageConfig QwenImageConfig::parse(const nlohmann::json& document) {
    if (!document.is_object())
        throw std::runtime_error("Qwen-Image runtime.json must be an object");

    QwenImageConfig config;
    config.engine_backend = require_string(document, "engine_backend");
    config.model_family = require_string(document, "model_family");
    config.model_variant = require_string(document, "model_variant");
    config.task_mode = parse_task_mode(require_string(document, "task_mode"));
    parse_diffusion(require_object(document, "diffusion"), config.diffusion);
    parse_text_encoder(require_object(document, "text_encoder"), config.text_encoder);
    parse_denoiser(require_object(document, "denoiser"), config.denoiser);
    parse_vae(require_object(document, "vae"), config.vae);
    parse_image(require_object(document, "image"), config.image);
    parse_tokenizer(require_object(document, "tokenizer"), config.tokenizer);
    if (config.task_mode == QwenImageTaskMode::Edit) {
        parse_vision_encoder(require_object(document, "vision_encoder"), config.vision_encoder);
        parse_image_conditioning(require_object(document, "image_conditioning"),
                                 config.image_conditioning);
    }
    return config;
}

QwenImagePreprocessorWeights parse_qwen_image_preprocessor_weights(const std::vector<char>& data) {
    const char* blob = nullptr;
    std::size_t blob_size = 0;
    const auto index =
        qwen_image_preprocessor_weights::extract_preprocessor_index(data, blob, blob_size);

    QwenImagePreprocessorWeights weights;
    qwen_image_preprocessor_weights::load_preprocessor_floats(index, blob, blob_size,
                                                              "latents_mean", weights.latents_mean);
    qwen_image_preprocessor_weights::load_preprocessor_floats(index, blob, blob_size, "latents_std",
                                                              weights.latents_std);
    if (weights.latents_mean.empty() || weights.latents_mean.size() != weights.latents_std.size())
        throw std::runtime_error("Qwen-Image preprocessor.weights is incomplete");
    weights.valid = true;
    return weights;
}

} // namespace trtmc

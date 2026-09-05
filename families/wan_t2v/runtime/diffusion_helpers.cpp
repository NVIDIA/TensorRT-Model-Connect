/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "diffusion_helpers.h"

#include <iostream>
#include <nlohmann/json.hpp>

namespace trtmc {
namespace {

nlohmann::json parse_runtime_object(const std::string& text) {
    auto document = nlohmann::json::parse(text);
    if (!document.is_object())
        throw std::runtime_error("runtime.json must be a JSON object");
    return document;
}

const nlohmann::json& require_member(const nlohmann::json& document, const char* key) {
    const auto found = document.find(key);
    if (found == document.end())
        throw std::runtime_error(std::string("runtime.json is missing '") + key + "'");
    return *found;
}

int32_t require_int(const nlohmann::json& document, const char* key) {
    const auto& value = require_member(document, key);
    if (!value.is_number_integer() && !value.is_number_unsigned())
        throw std::runtime_error(std::string("runtime.json '") + key + "' must be an integer");
    return value.get<int32_t>();
}

int32_t optional_int(const nlohmann::json& document, const char* key, int32_t default_value) {
    return document.contains(key) ? require_int(document, key) : default_value;
}

float require_number(const nlohmann::json& document, const char* key) {
    const auto& value = require_member(document, key);
    if (!value.is_number())
        throw std::runtime_error(std::string("runtime.json '") + key + "' must be numeric");
    return value.get<float>();
}

float optional_number(const nlohmann::json& document, const char* key, float default_value) {
    return document.contains(key) ? require_number(document, key) : default_value;
}

std::string require_string(const nlohmann::json& document, const char* key) {
    const auto& value = require_member(document, key);
    if (!value.is_string())
        throw std::runtime_error(std::string("runtime.json '") + key + "' must be a string");
    return value.get<std::string>();
}

std::vector<int32_t> require_int_array(const nlohmann::json& document, const char* key) {
    const auto& value = require_member(document, key);
    if (!value.is_array())
        throw std::runtime_error(std::string("runtime.json '") + key + "' must be an array");
    std::vector<int32_t> result;
    result.reserve(value.size());
    for (const auto& element : value) {
        if (!element.is_number_integer() && !element.is_number_unsigned())
            throw std::runtime_error(std::string("runtime.json '") + key +
                                     "' must contain only integers");
        result.push_back(element.get<int32_t>());
    }
    return result;
}

std::vector<int32_t> optional_int_array(const nlohmann::json& document, const char* key) {
    return document.contains(key) ? require_int_array(document, key) : std::vector<int32_t>{};
}

std::vector<float> require_number_array(const nlohmann::json& document, const char* key) {
    const auto& value = require_member(document, key);
    if (!value.is_array())
        throw std::runtime_error(std::string("runtime.json '") + key + "' must be an array");
    std::vector<float> result;
    result.reserve(value.size());
    for (const auto& element : value) {
        if (!element.is_number())
            throw std::runtime_error(std::string("runtime.json '") + key +
                                     "' must contain only numbers");
        result.push_back(element.get<float>());
    }
    return result;
}

bool optional_int_flag(const nlohmann::json& document, const char* key, bool default_value) {
    if (!document.contains(key))
        return default_value;
    const int32_t value = require_int(document, key);
    if (value != 0 && value != 1)
        throw std::runtime_error(std::string("runtime.json '") + key + "' must be 0 or 1");
    return value != 0;
}

void load_preprocessor_weights(const nlohmann::json& index_json, const char* blob,
                               std::size_t blob_size, WanPreprocessorWeights& w) {
    wan_preprocessor_weights::load_preprocessor_floats(
        index_json, blob, blob_size, "patch_embedding.weight", w.patch_embed_weight);
    wan_preprocessor_weights::load_preprocessor_floats(index_json, blob, blob_size,
                                                       "patch_embedding.bias", w.patch_embed_bias);
    wan_preprocessor_weights::load_preprocessor_floats(index_json, blob, blob_size,
                                                       "condition_embedder.time_embedding.0.weight",
                                                       w.time_emb_0_weight);
    wan_preprocessor_weights::load_preprocessor_floats(
        index_json, blob, blob_size, "condition_embedder.time_embedding.0.bias", w.time_emb_0_bias);
    wan_preprocessor_weights::load_preprocessor_floats(index_json, blob, blob_size,
                                                       "condition_embedder.time_embedding.2.weight",
                                                       w.time_emb_2_weight);
    wan_preprocessor_weights::load_preprocessor_floats(
        index_json, blob, blob_size, "condition_embedder.time_embedding.2.bias", w.time_emb_2_bias);

    wan_preprocessor_weights::load_preprocessor_floats(
        index_json, blob, blob_size, "condition_embedder.time_proj.weight", w.time_proj_weight);
    wan_preprocessor_weights::load_preprocessor_floats(
        index_json, blob, blob_size, "condition_embedder.time_proj.bias", w.time_proj_bias);

    wan_preprocessor_weights::load_preprocessor_floats(index_json, blob, blob_size,
                                                       "condition_embedder.text_embedding.weight",
                                                       w.text_proj_weight);
    wan_preprocessor_weights::load_preprocessor_floats(
        index_json, blob, blob_size, "condition_embedder.text_embedding.bias", w.text_proj_bias);
    wan_preprocessor_weights::load_preprocessor_floats(index_json, blob, blob_size,
                                                       "condition_embedder.text_embedding_2.weight",
                                                       w.text_proj_2_weight);
    wan_preprocessor_weights::load_preprocessor_floats(index_json, blob, blob_size,
                                                       "condition_embedder.text_embedding_2.bias",
                                                       w.text_proj_2_bias);

    wan_preprocessor_weights::load_preprocessor_floats(
        index_json, blob, blob_size, "context_embedder.weight", w.context_embed_weight);
    wan_preprocessor_weights::load_preprocessor_floats(
        index_json, blob, blob_size, "context_embedder.bias", w.context_embed_bias);

    wan_preprocessor_weights::load_preprocessor_floats(
        index_json, blob, blob_size, "condition_embedder.guidance_embedding.0.weight",
        w.guidance_emb_0_weight);
    wan_preprocessor_weights::load_preprocessor_floats(
        index_json, blob, blob_size, "condition_embedder.guidance_embedding.0.bias",
        w.guidance_emb_0_bias);
    wan_preprocessor_weights::load_preprocessor_floats(
        index_json, blob, blob_size, "condition_embedder.guidance_embedding.2.weight",
        w.guidance_emb_2_weight);
    wan_preprocessor_weights::load_preprocessor_floats(
        index_json, blob, blob_size, "condition_embedder.guidance_embedding.2.bias",
        w.guidance_emb_2_bias);

    wan_preprocessor_weights::load_preprocessor_floats(index_json, blob, blob_size,
                                                       "vae_bn.running_mean", w.vae_bn_mean);
    wan_preprocessor_weights::load_preprocessor_floats(index_json, blob, blob_size,
                                                       "vae_bn.running_var", w.vae_bn_var);
}

void finalize_preprocessor_weights(WanPreprocessorWeights& w) {
    if (!w.patch_embed_weight.empty() && !w.patch_embed_bias.empty()) {
        const auto dit_dim = static_cast<int32_t>(w.patch_embed_bias.size());
        w.patch_dim = static_cast<int32_t>(w.patch_embed_weight.size()) / dit_dim;
    }
    w.valid = !w.patch_embed_weight.empty() && !w.time_emb_0_weight.empty();
}

WanPreprocessorWeights parse_preprocessor_weights(const std::vector<char>& data) {
    WanPreprocessorWeights w;
    const char* blob = nullptr;
    std::size_t blob_size = 0;
    const auto index_json =
        wan_preprocessor_weights::extract_preprocessor_index(data, blob, blob_size);
    load_preprocessor_weights(index_json, blob, blob_size, w);
    finalize_preprocessor_weights(w);

    std::cerr << "[wan] Preprocessor weights loaded: " << (w.valid ? "OK" : "INCOMPLETE")
              << " (patch_dim=" << w.patch_dim << ")\n";
    return w;
}

} // namespace

WanDiffusionConfig make_diffusion_config(const std::string& json) {
    const auto document = parse_runtime_object(json);
    WanDiffusionConfig dc;
    dc.scheduler = require_string(document, "scheduler");
    dc.num_inference_steps = require_int(document, "num_inference_steps");
    dc.guidance_scale = require_number(document, "guidance_scale");
    dc.flow_shift = optional_number(document, "flow_shift", 1.0F);
    dc.unipc_lower_order_final = optional_int_flag(document, "unipc_lower_order_final", true);
    const auto& add_special_tokens = require_member(document, "tokenizer_add_special_tokens");
    if (!add_special_tokens.is_boolean())
        throw std::runtime_error("runtime.json 'tokenizer_add_special_tokens' must be a boolean");
    dc.tokenizer_add_special_tokens = add_special_tokens.get<bool>();
    dc.use_dynamic_shifting = optional_int_flag(document, "use_dynamic_shifting", false);
    dc.base_shift = optional_number(document, "base_shift", 0.5F);
    dc.max_shift = optional_number(document, "max_shift", 1.15F);
    dc.base_image_seq_len = optional_int(document, "base_image_seq_len", 256);
    dc.max_image_seq_len = optional_int(document, "max_image_seq_len", 4096);
    dc.shift_terminal = optional_number(document, "shift_terminal", 0.0F);
    dc.video_height = require_int(document, "video_height");
    dc.video_width = require_int(document, "video_width");
    dc.video_num_frames = require_int(document, "video_num_frames");
    dc.z_dim = require_int(document, "z_dim");
    dc.scale_factor_temporal = require_int(document, "scale_factor_temporal");
    dc.scale_factor_spatial = require_int(document, "scale_factor_spatial");
    dc.dit_dim = require_int(document, "dit_dim");
    dc.dit_num_heads = require_int(document, "dit_num_heads");
    dc.freq_dim = require_int(document, "freq_dim");
    dc.text_seq_len = require_int(document, "text_seq_len");
    dc.text_encoder_dim = require_int(document, "text_encoder_dim");
    dc.num_vae_caches = require_int(document, "num_vae_caches");
    dc.latents_mean = require_number_array(document, "latents_mean");
    dc.latents_std = require_number_array(document, "latents_std");
    dc.patch_size = require_int_array(document, "patch_size");
    dc.axes_dims_rope = optional_int_array(document, "axes_dims_rope");
    dc.rope_theta = optional_number(document, "rope_theta", 10000.0F);
    dc.vae_model_id = require_string(document, "vae_model_id");
    dc.guidance_embeds = optional_int_flag(document, "guidance_embeds", false);
    dc.use_rope = optional_int_flag(document, "use_rope", true);
    dc.vae_scaling_factor = optional_number(document, "vae_scaling_factor", 0.0F);
    dc.diffusion_backend_type = require_string(document, "diffusion_backend_type");
    return dc;
}

DiffusionParts load_diffusion_parts(IBackend* backend, const BundleReader& bundle,
                                    const std::string& json, const ModuleCreateOptions& options,
                                    const std::string& denoiser_section_name,
                                    const ModuleCreateOptions* denoiser_options) {
    DiffusionParts parts;

    const ModuleCreateOptions& effective_denoiser_options =
        denoiser_options != nullptr ? *denoiser_options : options;
    const auto denoiser_plan = bundle.read_section(denoiser_section_name);
    parts.denoiser = load_trt_module_from_plan(
        backend, &denoiser_plan, denoiser_section_name.c_str(), effective_denoiser_options);
    const auto vae_plan = bundle.read_section("vae.plan");
    parts.vae = load_trt_module_from_plan(backend, &vae_plan, "vae.plan", options);
    const auto vae_first_frame_plan = bundle.read_section("vae.first_frame.plan");
    parts.vae_first_frame =
        load_trt_module_from_plan(backend, &vae_first_frame_plan, "vae.first_frame.plan", options);

    const auto text_encoder_count =
        nlohmann::json::parse(json).at("num_text_encoders").get<std::int32_t>();
    if (text_encoder_count <= 0)
        throw std::runtime_error("num_text_encoders must be positive");
    for (std::int32_t index = 0; index < text_encoder_count; ++index) {
        const std::string label = "text_encoder." + std::to_string(index) + ".plan";
        const auto plan = bundle.read_section(label);
        parts.text_encoders.push_back(
            load_trt_module_from_plan(backend, &plan, label.c_str(), options));
    }

    parts.config = make_diffusion_config(json);
    const auto* weights = bundle.find_section("preprocessor.weights");
    if (weights != nullptr && weights->length > 0)
        parts.weights = parse_preprocessor_weights(bundle.read_section("preprocessor.weights"));

    parts.tokenizer = create_tokenizer_from_bundle(bundle);
    return parts;
}

} // namespace trtmc

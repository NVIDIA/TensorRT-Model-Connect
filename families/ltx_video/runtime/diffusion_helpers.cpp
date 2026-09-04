/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "diffusion_helpers.h"

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

} // namespace

LTXVideoDiffusionConfig make_diffusion_config(const std::string& json) {
    const auto document = parse_runtime_object(json);
    LTXVideoDiffusionConfig dc;
    dc.scheduler = require_string(document, "scheduler");
    dc.num_inference_steps = require_int(document, "num_inference_steps");
    dc.guidance_scale = require_number(document, "guidance_scale");
    dc.flow_shift = optional_number(document, "flow_shift", 1.0F);
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

} // namespace trtmc

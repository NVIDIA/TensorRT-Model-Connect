/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/deepseek_ocr/runtime/runtime_config.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace trtmc {
namespace {

std::int32_t require_int(const nlohmann::json& json, const char* key) {
    if (!json.contains(key) || !json.at(key).is_number_integer())
        throw std::runtime_error(std::string("DeepSeek-OCR runtime.json requires integer ") + key);
    return json.at(key).get<std::int32_t>();
}

std::int32_t require_positive_int(const nlohmann::json& json, const char* key) {
    const auto value = require_int(json, key);
    if (value <= 0)
        throw std::runtime_error(std::string("DeepSeek-OCR runtime.json has invalid ") + key);
    return value;
}

std::string require_string(const nlohmann::json& json, const char* key) {
    if (!json.contains(key) || !json.at(key).is_string() ||
        json.at(key).get_ref<const std::string&>().empty())
        throw std::runtime_error(std::string("DeepSeek-OCR runtime.json requires string ") + key);
    return json.at(key).get<std::string>();
}

} // namespace

DeepseekOcrRuntimeConfig deepseek_ocr_parse_runtime_config(const std::string& text) {
    const auto json = nlohmann::json::parse(text);
    if (!json.is_object())
        throw std::runtime_error("DeepSeek-OCR runtime.json must be an object");

    DeepseekOcrRuntimeConfig config;
    config.tensor_parallel_size = require_positive_int(json, "tensor_parallel_size");
    config.max_cache_length = require_positive_int(json, "max_cache_length");
    config.model.num_layers = require_positive_int(json, "num_layers");
    config.model.vocab_size = require_positive_int(json, "vocab_size");
    config.model.id_bos = require_int(json, "id_bos");
    config.model.id_eos = require_int(json, "id_eos");
    config.model.image_token_id = require_int(json, "image_token_id");
    config.model.vision_output_dim = require_positive_int(json, "vision_output_dim");
    config.model.prefill_max_length = require_positive_int(json, "prefill_max_length");

    if (!json.contains("io_map") || !json.at("io_map").is_object())
        throw std::runtime_error("DeepSeek-OCR runtime.json requires object io_map");
    const auto& io = json.at("io_map");
    config.cache_k_pattern = require_string(io, "cache_k_pattern");
    config.cache_v_pattern = require_string(io, "cache_v_pattern");
    config.model.present_k_pattern = require_string(io, "present_k_pattern");
    config.model.present_v_pattern = require_string(io, "present_v_pattern");
    return config;
}

} // namespace trtmc

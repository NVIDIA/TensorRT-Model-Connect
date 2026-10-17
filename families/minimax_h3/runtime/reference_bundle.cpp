/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/reference_bundle.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace trtmc::minimax_h3 {
namespace {
int32_t integer(const nlohmann::json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_number_integer() || value.is_boolean())
        throw std::invalid_argument("MiniMax-H3 reference metadata requires an integer: " +
                                    std::string(name));
    if (value.is_number_unsigned() &&
        value.get<uint64_t>() > uint64_t{std::numeric_limits<int32_t>::max()})
        throw std::invalid_argument("MiniMax-H3 reference metadata integer overflow");
    const auto number = value.get<int64_t>();
    if (number < std::numeric_limits<int32_t>::min() ||
        number > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("MiniMax-H3 reference metadata integer overflow");
    return static_cast<int32_t>(number);
}

bool same_value(const nlohmann::json& value, const nlohmann::json& expected) {
    if (expected.is_number_integer())
        return value.is_number_integer() && value == expected;
    if (value.type() != expected.type())
        return false;
    if (expected.is_array()) {
        if (value.size() != expected.size())
            return false;
        for (std::size_t index = 0; index < value.size(); ++index)
            if (!same_value(value[index], expected[index]))
                return false;
        return true;
    }
    return value == expected;
}

void exact(const nlohmann::json& config, const char* name, const nlohmann::json& expected) {
    if (!same_value(config.at(name), expected))
        throw std::invalid_argument("MiniMax-H3 reference metadata mismatch: " + std::string(name));
}

std::array<float, 24> statistics(const nlohmann::json& config, const char* name, bool positive) {
    const auto& values = config.at(name);
    if (!values.is_array() || values.size() != 24)
        throw std::invalid_argument("MiniMax-H3 reference statistics require 24 channels");
    std::array<float, 24> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        if (!values[index].is_number())
            throw std::invalid_argument("MiniMax-H3 reference statistics must be numeric");
        result[index] = values[index].get<float>();
        if (!std::isfinite(result[index]) || (positive && result[index] <= 0))
            throw std::invalid_argument(
                "MiniMax-H3 reference statistics must be finite with positive std");
    }
    return result;
}
} // namespace

std::vector<std::string> image_reference_plan_names() {
    return {"vision.plan", "text_encoder.plan", "vae_encoder.plan", "vae_reference_posterior.plan",
            "adaln.plan",  "denoiser.plan",     "vae.plan",         "audio_vae.plan"};
}

ReferencePipelineConfig parse_image_reference_config(const nlohmann::json& runtime) {
    exact(runtime, "schema_version", "minimax_h3.image_ref2va/v1");
    exact(runtime, "workflow", "image_ref2va");
    exact(runtime, "weights_variant", "transformer_ref");
    const auto capacity = runtime.value("reference_capacity", std::string("standard"));
    if (capacity != "standard" && capacity != "six_portrait")
        throw std::invalid_argument("MiniMax-H3 unsupported reference capacity profile");
    const int32_t text_rows = capacity == "six_portrait" ? 65536 : 32768;
    const int32_t condition_rows = capacity == "six_portrait" ? 49152 : 16384;
    for (const auto& pair : {std::pair{"num_frames", 124},
                             {"fps", 24},
                             {"num_inference_steps", 50},
                             {"denoiser_forwards", 49},
                             {"latent_frames", 37},
                             {"audio_latent_frames", 207},
                             {"audio_sampling_rate", 32000},
                             {"audio_channels", 2},
                             {"layers", 50},
                             {"hidden_size", 5376},
                             {"text_width", 5120},
                             {"context_parallel_size", 1}}) {
        if (integer(runtime, pair.first) != pair.second)
            throw std::invalid_argument(
                "MiniMax-H3 reference metadata declares a different original topology");
    }
    if (integer(runtime, "max_text_rows") != text_rows)
        throw std::invalid_argument("MiniMax-H3 reference capacity and text budget disagree");
    exact(runtime, "first_block_cache", false);
    exact(runtime, "text_rows", nlohmann::json::array({1, 128, text_rows}));
    exact(runtime, "condition_video_rows",
          capacity == "six_portrait" ? nlohmann::json::array({4096, 43776, 49152})
                                     : nlohmann::json::array({4096, 5504, 16384}));
    exact(runtime, "condition_audio_rows", nlohmann::json::array({0, 0, 0}));
    exact(runtime, "reference_pixels", nlohmann::json::array({16384, 22016, 65536}));
    ReferencePipelineConfig config;
    const auto mode = runtime.at("memory_preset").get<std::string>();
    const auto batch = integer(runtime, "vae_tile_batch");
    if (mode == "minimal" && batch == 1)
        config.memory = ReferenceMemoryPreset::Minimal;
    else if (mode == "original" && batch == 28)
        config.memory = ReferenceMemoryPreset::Original;
    else
        throw std::invalid_argument(
            "MiniMax-H3 reference residency and static video batch disagree");
    const auto height = integer(runtime, "height"), width = integer(runtime, "width");
    if (height < 256 || width < 256 || height % 32 || width % 32)
        throw std::invalid_argument(
            "MiniMax-H3 reference output geometry must be 32-aligned and at least 256 pixels");
    const int64_t rows =
        int64_t{37} * (height / 32) * (width / 32) + 414 + text_rows + condition_rows;
    if (rows > std::numeric_limits<int32_t>::max() ||
        integer(runtime, "max_sequence_rows") != rows ||
        integer(runtime, "latent_height") != height / 16 ||
        integer(runtime, "latent_width") != width / 16)
        throw std::invalid_argument(
            "MiniMax-H3 reference latent geometry and packed sequence cap disagree");
    config.denoiser.target = {37, height / 16, width / 16, 207};
    config.denoiser.max_sequence_rows = static_cast<int32_t>(rows);
    config.max_text_rows = text_rows;
    config.max_condition_video_rows = condition_rows;
    config.normalization.mean = statistics(runtime, "latents_mean", false);
    config.normalization.std = statistics(runtime, "latents_std", true);
    const auto& hashes = runtime.at("config_sha256");
    for (const auto* name : {"transformer_ref", "text_encoder", "vae", "audio_vae"}) {
        const auto digest = hashes.at(name).get<std::string>();
        if (digest.size() != 64 || !std::all_of(digest.begin(), digest.end(), [](char c) {
                return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
            }))
            throw std::invalid_argument(
                "MiniMax-H3 reference config digest must be lowercase SHA-256");
    }
    return config;
}

} // namespace trtmc::minimax_h3

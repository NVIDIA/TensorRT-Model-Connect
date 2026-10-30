/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/reference_bundle.h"

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
    if (expected.is_object()) {
        if (value.size() != expected.size())
            return false;
        for (const auto& field : expected.items())
            if (!value.contains(field.key()) || !same_value(value.at(field.key()), field.value()))
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

namespace {
void require_pdd_contract(const nlohmann::json& runtime) {
    exact(runtime, "checkpoint",
          {
              {"hf_id", "FastVideo/FastVideo-FastH3-Omni-8-Step-V1"},
              {"revision", "53e0d26a1691aa473b6262d8196a8feb2c297df0"},
          });
    exact(runtime, "inference_contract",
          {
              {"schema", "fasth3-inference-contract-v1"},
              {"schema_version", "fasth3-inference-contract-v1"},
              {"model_type", "ref2va"},
              {"transformer_component", "transformer_ref"},
              {"base_model_revision",
               "hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da"},
              {"conditioning", "fixed_ordered_references_target_only_flow"},
              {"pdd_steps", 32},
              {"pdd_step_indices", {0, 4, 8, 12, 16, 20, 24, 28, 32}},
              {"num_inference_steps", 8},
              {"transformer_forwards", 8},
              {"grid_max_t", 0.999},
              {"video_scheduler_shift", 12.0},
              {"audio_scheduler_shift", 3.0},
              {"guidance_scale", 1.0},
              {"attention_backend", "VIDEO_SPARSE_ATTN_H3"},
              {"vsa_sparsity", 0.9},
              {"vsa_tile_size", 128},
              {"vsa_ref_keep_rate", 0.1},
              {"vsa_ref_policy", "p2_multi_region"},
          });
}

ReferencePipelineConfig parse_reference_config(const nlohmann::json& runtime, bool pdd) {
    const std::string single = pdd ? "minimax_h3.omniref_pdd/v1" : "minimax_h3.image_ref2va/v1";
    const std::string multi =
        pdd ? "minimax_h3.omniref_pdd_continuation/v1" : "minimax_h3.image_ref2va_continuation/v1";
    const bool continuation = runtime.at("schema_version") == multi;
    exact(runtime, "schema_version", continuation ? multi : single);
    exact(runtime, "workflow",
          pdd ? (continuation ? "omniref_pdd_continuation" : "omniref_pdd")
              : (continuation ? "image_ref2va_continuation" : "image_ref2va"));
    if (pdd)
        require_pdd_contract(runtime);
    exact(runtime, "weights_variant", "transformer_ref");
    const auto capacity = runtime.value("reference_capacity", std::string("standard"));
    if (capacity != "standard" && capacity != "six_portrait")
        throw std::invalid_argument("MiniMax-H3 unsupported reference capacity profile");
    const int32_t text_rows = capacity == "six_portrait" ? 65536 : 32768;
    const int32_t static_condition_rows = capacity == "six_portrait" ? 49152 : 16384;
    for (const auto& pair : {std::pair{"num_frames", continuation ? 226 : 124},
                             {"fps", 24},
                             {"num_inference_steps", pdd ? 8 : 50},
                             {"denoiser_forwards", pdd ? 8 : 49},
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
                "MiniMax-H3 reference metadata declares a different topology or recipe");
    }
    if (integer(runtime, "max_text_rows") != text_rows)
        throw std::invalid_argument("MiniMax-H3 reference capacity and text budget disagree");
    exact(runtime, "first_block_cache", false);
    exact(runtime, "text_rows", nlohmann::json::array({1, 128, text_rows}));
    exact(runtime, "condition_audio_rows",
          continuation ? nlohmann::json::array({0, 74, 74}) : nlohmann::json::array({0, 0, 0}));
    for (const auto& pair : {std::pair{"continuation_window_frames", 124},
                             {"continuation_overlap_frames", 22},
                             {"continuation_windows", 2},
                             {"decoded_latent_frames", 67},
                             {"decoded_audio_latent_frames", 377},
                             {"total_denoiser_forwards", pdd ? 16 : 98}}) {
        if (continuation)
            exact(runtime, pair.first, pair.second);
        else if (runtime.contains(pair.first))
            throw std::invalid_argument(
                "MiniMax-H3 single-window metadata cannot declare continuation");
    }
    exact(runtime, "reference_pixels", nlohmann::json::array({16384, 22016, 65536}));
    ReferencePipelineConfig config;
    if (pdd) {
        const int32_t tiles = integer(runtime, "max_vsa_tiles");
        if (tiles < 2 || tiles > 8192 || tiles % 2)
            throw std::invalid_argument("OmniRef VSA requires an even supported tile capacity");
        config.denoiser.recipe = ReferenceSamplingRecipe::OmniRefPdd8;
        config.denoiser.max_vsa_tiles = tiles;
    }
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
    const int64_t guide_rows = continuation ? int64_t{7} * (height / 32) * (width / 32) : 0;
    const int64_t condition_rows = static_condition_rows + guide_rows;
    const int64_t rows = int64_t{37} * (height / 32) * (width / 32) + 414 + text_rows +
                         condition_rows + (continuation ? 74 : 0);
    if (rows > std::numeric_limits<int32_t>::max() ||
        integer(runtime, "max_sequence_rows") != rows ||
        integer(runtime, "latent_height") != height / 16 ||
        integer(runtime, "latent_width") != width / 16)
        throw std::invalid_argument(
            "MiniMax-H3 reference latent geometry and packed sequence cap disagree");
    exact(runtime, "condition_video_rows",
          nlohmann::json::array(
              {4096, (capacity == "six_portrait" ? 43776 : 5504) + guide_rows, condition_rows}));
    config.denoiser.target = {37, height / 16, width / 16, 207};
    config.denoiser.max_sequence_rows = static_cast<int32_t>(rows);
    config.max_text_rows = text_rows;
    config.max_condition_video_rows = static_cast<int32_t>(condition_rows);
    config.output_frames = continuation ? 226 : 124;
    config.max_condition_audio_rows = continuation ? 74 : 0;
    config.normalization.mean = statistics(runtime, "latents_mean", false);
    config.normalization.std = statistics(runtime, "latents_std", true);
    return config;
}
} // namespace

ReferencePipelineConfig parse_image_reference_config(const nlohmann::json& runtime) {
    return parse_reference_config(runtime, false);
}

ReferencePipelineConfig parse_omniref_pdd_config(const nlohmann::json& runtime) {
    return parse_reference_config(runtime, true);
}

} // namespace trtmc::minimax_h3

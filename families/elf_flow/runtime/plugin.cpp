/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/elf_flow/runtime/pipeline.h"
#include "families/elf_flow/runtime/plugin_helpers.h"
#include "families/elf_flow/runtime/runtime_config.h"
#include "trtmc/runtime/family_factory.h"

#include <array>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

namespace trtmc::elf_flow {

namespace {

struct RuntimeConfig {
    std::int32_t max_length;
    std::int32_t max_input_length;
    std::int32_t input_dim;
    std::int32_t text_encoder_dim;
    std::int32_t vocab_size;
    float denoiser_noise_scale;
    float denoiser_p_mean;
    float denoiser_p_std;
    float timestep_epsilon;
    float latent_mean;
    float latent_std;
    std::int32_t encoder_pad_token_id;
};

constexpr std::array<std::string_view, 12> kRuntimeFields = {
    "max_length",       "max_input_length",     "input_dim",       "text_encoder_dim",
    "vocab_size",       "denoiser_noise_scale", "denoiser_p_mean", "denoiser_p_std",
    "timestep_epsilon", "latent_mean",          "latent_std",      "encoder_pad_token_id",
};

void require_exact_fields(const nlohmann::json& json) {
    if (!json.is_object())
        throw std::runtime_error("elf_flow runtime.json must be an object");
    if (json.size() != kRuntimeFields.size())
        throw std::runtime_error("elf_flow runtime.json has an unexpected field set");
    for (const auto field : kRuntimeFields) {
        if (!json.contains(std::string(field)))
            throw std::runtime_error("elf_flow runtime.json missing '" + std::string(field) + "'");
    }
}

RuntimeConfig parse_runtime_config(const std::string& text) {
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error("elf_flow invalid runtime.json: " + std::string(error.what()));
    }
    require_exact_fields(json);

    RuntimeConfig config{
        json.at("max_length").get<std::int32_t>(),
        json.at("max_input_length").get<std::int32_t>(),
        json.at("input_dim").get<std::int32_t>(),
        json.at("text_encoder_dim").get<std::int32_t>(),
        json.at("vocab_size").get<std::int32_t>(),
        json.at("denoiser_noise_scale").get<float>(),
        json.at("denoiser_p_mean").get<float>(),
        json.at("denoiser_p_std").get<float>(),
        json.at("timestep_epsilon").get<float>(),
        json.at("latent_mean").get<float>(),
        json.at("latent_std").get<float>(),
        json.at("encoder_pad_token_id").get<std::int32_t>(),
    };
    if (config.max_length <= 0 || config.max_input_length < 0 || config.input_dim <= 0 ||
        config.text_encoder_dim <= 0 || config.vocab_size <= 0 ||
        config.denoiser_noise_scale <= 0.0F || config.denoiser_p_std <= 0.0F ||
        config.timestep_epsilon <= 0.0F || config.latent_std <= 0.0F) {
        throw std::runtime_error("elf_flow runtime.json contains invalid dimensions or scales");
    }
    return config;
}

} // namespace

void validate_runtime_config_json(std::string_view json) {
    (void)parse_runtime_config(std::string(json));
}

ITask* create(const FamilyContext& context) {
    const RuntimeConfig config =
        parse_runtime_config(require_text_section(context.reader, "runtime.json"));
    auto model = load_engine(context.backend, require_section(context.reader, "engine.plan"),
                             "elf_flow engine");
    auto text_encoder =
        load_engine(context.backend, require_section(context.reader, "text_encoder.plan"),
                    "elf_flow text encoder");
    auto tokenizer = create_tokenizer(context.reader);
    return new ElfFlowPipeline(std::move(model), config.max_length, config.max_input_length,
                               config.input_dim, config.text_encoder_dim, config.vocab_size,
                               config.denoiser_noise_scale, config.denoiser_p_mean,
                               config.denoiser_p_std, config.timestep_epsilon, std::move(tokenizer),
                               std::string{}, std::move(text_encoder), config.latent_mean,
                               config.latent_std, config.encoder_pad_token_id);
}

} // namespace trtmc::elf_flow

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    return trtmc::elf_flow::create(context);
}

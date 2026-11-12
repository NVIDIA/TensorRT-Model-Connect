/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/stable_diffusion_3/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::stable_diffusion_3 {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("stable_diffusion_3 bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

ClipTokens parse_clip_tokens(const nlohmann::json& tokens, std::int32_t config_eos) {
    ClipTokens result;
    result.bos = tokens.at("bos").get<std::int32_t>();
    result.eos = tokens.at("eos").get<std::int32_t>();
    result.pad = tokens.at("pad").get<std::int32_t>();
    result.config_eos = config_eos;
    return result;
}

StableDiffusion3Config parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    StableDiffusion3Config config;
    config.height = json.at("height").get<std::int32_t>();
    config.width = json.at("width").get<std::int32_t>();
    config.latent_height = json.at("latent_height").get<std::int32_t>();
    config.latent_width = json.at("latent_width").get<std::int32_t>();
    config.latent_channels = json.at("latent_channels").get<std::int32_t>();
    config.clip_length = json.at("clip_length").get<std::int32_t>();
    config.clip_l_width = json.at("clip_l_width").get<std::int32_t>();
    config.clip_g_width = json.at("clip_g_width").get<std::int32_t>();
    config.clip_l_projection = json.at("clip_l_projection").get<std::int32_t>();
    config.clip_g_projection = json.at("clip_g_projection").get<std::int32_t>();
    config.clip_l_tokens = parse_clip_tokens(json.at("clip_l_tokens"),
                                             json.at("clip_l_config_eos").get<std::int32_t>());
    config.clip_g_tokens = parse_clip_tokens(json.at("clip_g_tokens"),
                                             json.at("clip_g_config_eos").get<std::int32_t>());
    config.t5_length = json.at("t5_length").get<std::int32_t>();
    config.t5_eos_id = json.at("t5_eos_id").get<std::int32_t>();
    config.t5_pad_id = json.at("t5_pad_id").get<std::int32_t>();
    config.joint_attention_dim = json.at("joint_attention_dim").get<std::int32_t>();
    config.pooled_projection_dim = json.at("pooled_projection_dim").get<std::int32_t>();
    config.timestep_channels = json.at("timestep_channels").get<std::int32_t>();
    config.num_train_timesteps = json.at("num_train_timesteps").get<std::int32_t>();
    config.shift = json.at("shift").get<double>();
    config.scaling_factor = json.at("scaling_factor").get<float>();
    config.shift_factor = json.at("shift_factor").get<float>();
    config.max_batch = json.at("max_batch").get<std::int32_t>();
    config.default_num_steps = json.at("default_num_steps").get<std::int32_t>();
    config.default_guidance_scale = json.at("default_guidance_scale").get<double>();
    if (config.height <= 0 || config.width <= 0 || config.latent_height <= 0 ||
        config.latent_width <= 0 || config.latent_channels <= 0 || config.clip_length < 2 ||
        config.t5_length < 1 || config.timestep_channels <= 0 || config.max_batch < 1 ||
        config.scaling_factor == 0.0F || config.default_num_steps < 1 ||
        config.num_train_timesteps < 2)
        throw std::runtime_error("stable_diffusion_3 runtime.json does not match its contract");
    return config;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan,
                                        const char* what) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("stable_diffusion_3 " + std::string(what) +
                                 " engine failed to load");
    return engine;
}

std::unique_ptr<ITokenizer> load_tokenizer(const BundleReader& bundle, const char* section,
                                           bool unigram) {
    const auto data = require_section(bundle, section);
    // Special tokens are framed by the pipeline, not by the tokenizers.
    auto tokenizer = unigram ? CreateUnigramTokenizer(data.data(), data.size(), false)
                             : CreateBpeTokenizer(data.data(), data.size(), false);
    if (!tokenizer)
        throw std::runtime_error("stable_diffusion_3 could not build " + std::string(section));
    return tokenizer;
}

} // namespace
} // namespace trtmc::stable_diffusion_3

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("stable_diffusion_3 does not support --kv-cache-size");
    namespace sd3 = trtmc::stable_diffusion_3;
    if (context.reader.info().task != "text_to_image")
        throw std::runtime_error("stable_diffusion_3 bundles carry the text_to_image Task");
    auto config = sd3::parse_config(sd3::require_section(context.reader, "runtime.json"));

    trtmc::StableDiffusion3Tokenizers tokenizers;
    tokenizers.clip_l = sd3::load_tokenizer(context.reader, "clip_l_tokenizer.json", false);
    tokenizers.clip_g = sd3::load_tokenizer(context.reader, "clip_g_tokenizer.json", false);
    tokenizers.t5 = sd3::load_tokenizer(context.reader, "t5_tokenizer.json", true);

    trtmc::StableDiffusion3Engines engines;
    engines.clip_l = sd3::load_engine(
        context.backend, sd3::require_section(context.reader, "clip_l.plan"), "CLIP-L");
    engines.clip_g = sd3::load_engine(
        context.backend, sd3::require_section(context.reader, "clip_g.plan"), "CLIP-G");
    engines.t5 =
        sd3::load_engine(context.backend, sd3::require_section(context.reader, "t5.plan"), "T5");
    engines.transformer = sd3::load_engine(
        context.backend, sd3::require_section(context.reader, "transformer.plan"), "transformer");
    engines.vae =
        sd3::load_engine(context.backend, sd3::require_section(context.reader, "vae.plan"), "VAE");
    return new trtmc::StableDiffusion3Pipeline(std::move(engines), std::move(tokenizers),
                                               std::move(config));
}

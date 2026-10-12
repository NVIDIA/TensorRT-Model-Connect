/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/stable_diffusion/runtime/pipeline.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::stable_diffusion {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

StableDiffusionConfig parse_config(const std::vector<char>& data) {
    const auto json = nlohmann::json::parse(data.begin(), data.end());
    StableDiffusionConfig config;
    config.latent_size = json.at("latent_size").get<std::int32_t>();
    config.latent_channels = json.at("latent_channels").get<std::int32_t>();
    config.image_size = json.at("image_size").get<std::int32_t>();
    config.context_length = json.at("context_length").get<std::int32_t>();
    config.context_width = json.at("context_width").get<std::int32_t>();
    config.scaling_factor = json.at("scaling_factor").get<float>();
    config.num_train_timesteps = json.at("num_train_timesteps").get<std::int32_t>();
    config.steps_offset = json.at("steps_offset").get<std::int32_t>();
    config.default_num_steps = json.at("default_num_steps").get<std::int32_t>();
    config.default_guidance_scale = json.at("default_guidance_scale").get<float>();
    config.alphas_cumprod = json.at("alphas_cumprod").get<std::vector<float>>();
    if (config.latent_size <= 0 || config.latent_channels <= 0 || config.image_size <= 0 ||
        config.context_length <= 0 || config.context_width <= 0 || config.scaling_factor == 0.0F ||
        config.alphas_cumprod.empty()) {
        throw std::runtime_error("stable_diffusion runtime.json does not match its contract");
    }
    return config;
}

std::unique_ptr<ITrtModule> load_engine(IBackend& backend, const std::vector<char>& plan,
                                        const char* what) {
    ModuleCreateOptions options{};
    auto engine = backend.create_module(plan.data(), plan.size(), options);
    if (!engine || !engine->ok())
        throw std::runtime_error("stable_diffusion " + std::string(what) +
                                 " engine failed to load");
    return engine;
}

} // namespace
} // namespace trtmc::stable_diffusion

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("stable_diffusion does not support --kv-cache-size");
    namespace sd = trtmc::stable_diffusion;
    const auto config_data = sd::require_section(context.reader, "runtime.json");
    const auto text_plan = sd::require_section(context.reader, "text_encoder.plan");
    const auto unet_plan = sd::require_section(context.reader, "unet.plan");
    const auto vae_plan = sd::require_section(context.reader, "vae.plan");
    const auto tokenizer_json = sd::require_section(context.reader, "tokenizer.json");

    auto config = sd::parse_config(config_data);
    // The pipeline brackets the prompt itself, because this post-processor
    // emits nothing for an empty string and the unconditional prompt is empty.
    auto tokenizer = trtmc::CreateBpeTokenizer(tokenizer_json.data(), tokenizer_json.size(), false);
    if (!tokenizer)
        throw std::runtime_error("stable_diffusion could not build its tokenizer");

    return new trtmc::StableDiffusionPipeline(
        sd::load_engine(context.backend, text_plan, "text encoder"),
        sd::load_engine(context.backend, unet_plan, "unet"),
        sd::load_engine(context.backend, vae_plan, "vae"), std::move(tokenizer), std::move(config));
}

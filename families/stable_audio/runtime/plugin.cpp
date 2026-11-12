/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/stable_audio/runtime/pipeline.h"
#include "families/stable_audio/runtime/tokenizer.h"
#include "trtmc/internal/audio.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::stable_audio {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("stable_audio bundle section is missing or empty: " +
                                 std::string(name));
    return bundle.read_section(name);
}

std::unique_ptr<ITrtModule> load_engine(const FamilyContext& context, const char* section) {
    const auto plan = require_section(context.reader, section);
    auto module = context.backend.create_module(plan.data(), plan.size(), ModuleCreateOptions{});
    if (!module || !module->ok())
        throw std::runtime_error("stable_audio could not load " + std::string(section));
    module->set_timing_label(section);
    return module;
}

SchedulerConfig parse_scheduler(const nlohmann::json& json) {
    SchedulerConfig config;
    config.sigma_min = json.at("sigma_min").get<double>();
    config.sigma_max = json.at("sigma_max").get<double>();
    config.sigma_data = json.at("sigma_data").get<double>();
    config.rho = json.at("rho").get<double>();
    config.sigma_schedule = json.at("sigma_schedule").get<std::string>();
    config.prediction_type = json.at("prediction_type").get<std::string>();
    config.solver_type = json.at("solver_type").get<std::string>();
    config.final_sigmas_type = json.at("final_sigmas_type").get<std::string>();
    config.solver_order = json.at("solver_order").get<int>();
    return config;
}

StableAudioConfig parse_config(const BundleReader& bundle) {
    const auto bytes = require_section(bundle, "runtime.json");
    const auto json = nlohmann::json::parse(bytes.begin(), bytes.end());
    StableAudioConfig config;
#define STABLE_AUDIO_FIELD(field) config.field = json.at(#field).get<decltype(config.field)>()
    STABLE_AUDIO_FIELD(sample_rate);
    STABLE_AUDIO_FIELD(audio_channels);
    STABLE_AUDIO_FIELD(audio_samples);
    STABLE_AUDIO_FIELD(hop_length);
    STABLE_AUDIO_FIELD(latent_channels);
    STABLE_AUDIO_FIELD(latent_length);
    STABLE_AUDIO_FIELD(text_length);
    STABLE_AUDIO_FIELD(pad_token_id);
    STABLE_AUDIO_FIELD(eos_token_id);
    STABLE_AUDIO_FIELD(encoder_tokens);
    STABLE_AUDIO_FIELD(conditioning_dim);
    STABLE_AUDIO_FIELD(global_dim);
    STABLE_AUDIO_FIELD(duration_min);
    STABLE_AUDIO_FIELD(duration_max);
    STABLE_AUDIO_FIELD(default_num_inference_steps);
    STABLE_AUDIO_FIELD(default_guidance_scale);
#undef STABLE_AUDIO_FIELD
    config.scheduler = parse_scheduler(json.at("scheduler"));
    if (config.sample_rate <= 0 || config.audio_channels <= 0 || config.hop_length <= 0 ||
        config.latent_channels <= 0 || config.latent_length <= 0 || config.text_length <= 1 ||
        config.encoder_tokens != config.text_length + 2 || config.conditioning_dim <= 0 ||
        config.global_dim != 2 * config.conditioning_dim ||
        config.audio_samples !=
            static_cast<std::int64_t>(config.latent_length) * config.hop_length ||
        !(config.duration_max > config.duration_min) || config.default_num_inference_steps <= 0)
        throw std::runtime_error("stable_audio runtime.json does not match its runtime contract");
    return config;
}

void require_task(const BundleInfo& info) {
    if (info.task != std::string(trtmc::internal::ITextToAudio::kTask))
        throw std::runtime_error("stable_audio does not implement task: " + info.task);
}

} // namespace
} // namespace trtmc::stable_audio

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    namespace sa = trtmc::stable_audio;
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("stable_audio does not support --kv-cache-size");
    sa::require_task(context.reader.info());
    auto config = sa::parse_config(context.reader);
    const auto tokenizer_json = sa::require_section(context.reader, "tokenizer.json");
    // The pipeline truncates and appends </s> itself, so the tokenizer adds nothing.
    auto tokenizer =
        sa::CreateUnigramTokenizer(tokenizer_json.data(), tokenizer_json.size(), false);
    if (!tokenizer)
        throw std::runtime_error("stable_audio could not build its T5 tokenizer");
    sa::StableAudioEngines engines;
    engines.conditioner = sa::load_engine(context, "conditioner.plan");
    engines.transformer = sa::load_engine(context, "transformer.plan");
    engines.decoder = sa::load_engine(context, "vae_decoder.plan");
    return new sa::StableAudioPipeline(std::move(engines), std::move(tokenizer), std::move(config));
}

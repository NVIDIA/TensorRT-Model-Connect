/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_music3/runtime/pipeline.h"
#include "families/minimax_music3/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::minimax_music3_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("MiniMax-Music3 bundle is missing " + std::string(name));
    return bundle.read_section(name);
}

std::unique_ptr<ITrtModule> load_engine(const FamilyContext& context, const char* section) {
    const auto plan = require_section(context.reader, section);
    auto module = context.backend.create_module(plan.data(), plan.size(), {});
    if (!module || !module->ok())
        throw std::runtime_error("MiniMax-Music3 could not create " + std::string(section));
    module->set_timing_label(section);
    return module;
}

MinimaxMusic3Config read_config(const BundleReader& bundle) {
    const auto bytes = require_section(bundle, "runtime.json");
    const auto document = nlohmann::json::parse(bytes.begin(), bytes.end());
    MinimaxMusic3Config config;
#define TRTMC_MM3_READ(field) config.field = document.value(#field, config.field)
    TRTMC_MM3_READ(sampling_rate);
    TRTMC_MM3_READ(output_channels);
    TRTMC_MM3_READ(frame_rate_hz);
    TRTMC_MM3_READ(latent_hop_length);
    TRTMC_MM3_READ(latent_resample_ratio);
    TRTMC_MM3_READ(chunk_latent_length);
    TRTMC_MM3_READ(chunk_frames);
    TRTMC_MM3_READ(chunk_hop);
    TRTMC_MM3_READ(crop_left_latent);
    TRTMC_MM3_READ(crop_right_latent);
    TRTMC_MM3_READ(default_inference_steps);
    TRTMC_MM3_READ(max_audio_frames);
    TRTMC_MM3_READ(guidance_branches);
    TRTMC_MM3_READ(num_codebooks);
    TRTMC_MM3_READ(num_residual_codebooks);
    TRTMC_MM3_READ(audio_vocab_size);
    TRTMC_MM3_READ(latent_channels);
    TRTMC_MM3_READ(condition_dim);
    TRTMC_MM3_READ(frame_hidden_width);
    TRTMC_MM3_READ(condition_streams);
    TRTMC_MM3_READ(language_model_hidden_size);
    TRTMC_MM3_READ(language_model_kv_width);
    TRTMC_MM3_READ(guidance_scale);
    TRTMC_MM3_READ(language_model_vocab_size);
    TRTMC_MM3_READ(language_model_layers);
    TRTMC_MM3_READ(top_k);
    TRTMC_MM3_READ(temperature);
#undef TRTMC_MM3_READ
    config.max_frames = config.max_audio_frames;

    const auto require_positive = [](const char* key, auto value) {
        if (value <= 0)
            throw std::runtime_error(std::string("MiniMax-Music3 runtime.json field ") + key +
                                     " must be positive");
    };
    require_positive("sampling_rate", config.sampling_rate);
    require_positive("output_channels", config.output_channels);
    require_positive("latent_hop_length", config.latent_hop_length);
    require_positive("chunk_latent_length", config.chunk_latent_length);
    require_positive("chunk_frames", config.chunk_frames);
    require_positive("chunk_hop", config.chunk_hop);
    require_positive("max_audio_frames", config.max_audio_frames);
    require_positive("default_inference_steps", config.default_inference_steps);
    require_positive("language_model_layers", config.language_model_layers);
    if (config.crop_left_latent < 0 || config.crop_right_latent < 0 ||
        config.crop_left_latent + config.crop_right_latent >= config.chunk_latent_length)
        throw std::runtime_error("MiniMax-Music3 runtime.json declares invalid window crops");
    return config;
}

std::shared_ptr<ITokenizer> load_tokenizer(const BundleReader& bundle) {
    const auto data = require_section(bundle, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(data.data(), data.size(), false);
    if (!tokenizer)
        throw std::runtime_error("MiniMax-Music3 tokenizer.json is not a BPE tokenizer");
    return tokenizer;
}

} // namespace
} // namespace trtmc::minimax_music3_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("minimax_music3 does not support --kv-cache-size");
    using namespace trtmc;
    MinimaxMusic3Engines engines;
    engines.language_model = minimax_music3_factory::load_engine(context, "language_model.plan");
    engines.depth_decoder = minimax_music3_factory::load_engine(context, "depth_decoder.plan");
    engines.condition_encoder =
        minimax_music3_factory::load_engine(context, "condition_encoder.plan");
    engines.dit = minimax_music3_factory::load_engine(context, "dit.plan");
    engines.vocoder = minimax_music3_factory::load_engine(context, "vocoder.plan");
    return new MinimaxMusic3TextToMusicPipeline(
        std::move(engines), minimax_music3_factory::read_config(context.reader),
        minimax_music3_factory::load_tokenizer(context.reader));
}

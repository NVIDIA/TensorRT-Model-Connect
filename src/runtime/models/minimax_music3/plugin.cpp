/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_view.h"
#include "runtime/models/minimax_music3/pipeline.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/tokenizer.h"
#include "utils/json_helpers.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc {
namespace {

// Bundle section names. build_engine() writes the diffusion transformer as the
// primary plan and build_extra_engines() writes the other four as
// "<engine>_plan" -- see the family's plugin.py.
constexpr const char* kDitSection = "engine_plan";
constexpr const char* kLanguageModelSection = "language_model_plan";
constexpr const char* kDepthDecoderSection = "depth_decoder_plan";
constexpr const char* kConditionEncoderSection = "condition_encoder_plan";
constexpr const char* kVocoderSection = "vocoder_plan";

std::unique_ptr<ITrtModule> load_engine(const PipelineContext& context, const char* section) {
    const auto* plan = find_section(context.bundle, section);
    if (plan == nullptr || plan->empty())
        throw std::runtime_error(std::string("MiniMax-Music3 bundle is missing ") + section);
    if (context.backend == nullptr)
        throw std::runtime_error("MiniMax-Music3 has no backend loaded");

    ModuleCreateOptions options;
    options.runtime_cache_path = context.runtime_cache_path.c_str();
    options.cuda_graphs = context.cuda_graphs;
    auto module = context.backend->create_module(plan->data(), plan->size(), options);
    if (!module || !module->ok())
        throw std::runtime_error(std::string("MiniMax-Music3 could not create ") + section);
    module->set_timing_label(section);
    return module;
}

// The eighteen facts engines.bundle_config_overrides() writes. Reading them
// rather than recomputing them is the point: the window plan, the crop widths
// and the latent ratio were measured against a recorded generation, and a
// runtime that rebuilds them drifts from the run they were measured on.
MinimaxMusic3Config read_config(const PipelineContext& context) {
    const std::string& json = context.config_json;
    MinimaxMusic3Config config;
    config.sampling_rate = extract_json_int(json, "sampling_rate", config.sampling_rate);
    config.output_channels = extract_json_int(json, "output_channels", config.output_channels);
    config.frame_rate_hz = extract_json_float(json, "frame_rate_hz", config.frame_rate_hz);
    config.latent_hop_length =
        extract_json_int(json, "latent_hop_length", config.latent_hop_length);
    config.latent_resample_ratio =
        extract_json_float(json, "latent_resample_ratio", config.latent_resample_ratio);
    config.chunk_latent_length =
        extract_json_int(json, "chunk_latent_length", config.chunk_latent_length);
    config.chunk_frames = extract_json_int(json, "chunk_frames", config.chunk_frames);
    config.chunk_hop = extract_json_int(json, "chunk_hop", config.chunk_hop);
    config.crop_left_latent = extract_json_int(json, "crop_left_latent", config.crop_left_latent);
    config.crop_right_latent =
        extract_json_int(json, "crop_right_latent", config.crop_right_latent);
    config.default_inference_steps =
        extract_json_int(json, "default_inference_steps", config.default_inference_steps);
    config.max_audio_frames = extract_json_int(json, "max_audio_frames", config.max_audio_frames);
    config.guidance_branches =
        extract_json_int(json, "guidance_branches", config.guidance_branches);
    config.num_codebooks = extract_json_int(json, "num_codebooks", config.num_codebooks);
    config.num_residual_codebooks =
        extract_json_int(json, "num_residual_codebooks", config.num_residual_codebooks);
    config.audio_vocab_size = extract_json_int(json, "audio_vocab_size", config.audio_vocab_size);
    config.latent_channels = extract_json_int(json, "latent_channels", config.latent_channels);
    config.condition_dim = extract_json_int(json, "condition_dim", config.condition_dim);
    config.frame_hidden_width =
        extract_json_int(json, "frame_hidden_width", config.frame_hidden_width);
    config.condition_streams =
        extract_json_int(json, "condition_streams", config.condition_streams);
    config.language_model_hidden_size =
        extract_json_int(json, "language_model_hidden_size", config.language_model_hidden_size);
    config.language_model_kv_width =
        extract_json_int(json, "language_model_kv_width", config.language_model_kv_width);
    config.guidance_scale = extract_json_float(json, "guidance_scale", config.guidance_scale);
    config.language_model_vocab_size =
        extract_json_int(json, "language_model_vocab_size", config.language_model_vocab_size);
    config.language_model_layers =
        extract_json_int(json, "language_model_layers", config.language_model_layers);

    // The request's own namespace. The caption is the music description; the
    // lyrics travel in the prompt, because the task contract scores a
    // transcript against that field.
    if (context.runtime_config != nullptr) {
        config.caption =
            context.runtime_config->get<std::string>("music_minimax_music3", "caption");
        config.max_frames =
            context.runtime_config->get<std::int32_t>("music_minimax_music3", "max_frames");
        config.seed = context.runtime_config->get<std::int64_t>("music_minimax_music3", "seed");
    } else {
        config.max_frames = config.max_audio_frames;
    }
    if (config.max_frames <= 0 || config.max_frames > config.max_audio_frames)
        config.max_frames = config.max_audio_frames;
    return config;
}

std::shared_ptr<ITokenizer> load_tokenizer(const PipelineContext& context) {
    const auto* section = find_section(context.bundle, "tokenizer.json");
    if (section == nullptr || section->empty())
        throw std::runtime_error("MiniMax-Music3 bundle is missing tokenizer.json");
    // The prompt carries lyrics rather than a chat turn, so no special frame
    // is added around it.
    auto tokenizer = CreateBpeTokenizer(section->data(), section->size(),
                                        /*add_special_tokens=*/false);
    if (!tokenizer)
        throw std::runtime_error("MiniMax-Music3 could not build its tokenizer");
    return tokenizer;
}

} // namespace

class MinimaxMusic3Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& context) override {
        MinimaxMusic3Engines engines;
        // Loaded in the order a generation uses them, so a truncated bundle
        // fails on the first stage that is missing rather than the last.
        engines.language_model = load_engine(context, kLanguageModelSection);
        engines.depth_decoder = load_engine(context, kDepthDecoderSection);
        engines.condition_encoder = load_engine(context, kConditionEncoderSection);
        engines.dit = load_engine(context, kDitSection);
        engines.vocoder = load_engine(context, kVocoderSection);

        return std::make_unique<MinimaxMusic3TextToMusicPipeline>(
            std::move(engines), read_config(context), load_tokenizer(context),
            context.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_minimax_music3_plugin, MinimaxMusic3Plugin,
                                       "minimax_music3_text_to_music");

} // namespace trtmc

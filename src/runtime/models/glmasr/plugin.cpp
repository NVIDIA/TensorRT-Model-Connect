/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// GlmAsrPlugin: handles the "glmasr_speech_to_text" strategy.
// Audio encoder plus projector in one engine, feeding a Llama decoder that
// consumes the projected frames through its embed-input contract.

#include "plugin_helpers.h"
#include "runtime/models/glmasr/glmasr_config.h"
#include "runtime/models/glmasr/pipeline.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

namespace trtmc {

namespace {

GlmAsrConfig read_config(const std::string& json, const BaseConfig& base) {
    GlmAsrConfig config;

    config.mel_num_bins = extract_json_int(json, "mel_num_bins", config.mel_num_bins);
    config.mel_n_fft = extract_json_int(json, "mel_n_fft", config.mel_n_fft);
    config.mel_hop_length = extract_json_int(json, "mel_hop_length", config.mel_hop_length);
    config.mel_chunk_length = extract_json_int(json, "mel_chunk_length", config.mel_chunk_length);
    config.mel_sampling_rate =
        extract_json_int(json, "mel_sampling_rate", config.mel_sampling_rate);

    config.mel_length = extract_json_int(json, "mel_length", config.mel_length);
    config.encoder_frames = extract_json_int(json, "encoder_frames", config.encoder_frames);
    config.audio_merge_factor =
        extract_json_int(json, "audio_merge_factor", config.audio_merge_factor);
    config.max_audio_embeddings =
        extract_json_int(json, "num_audio_embeddings", config.max_audio_embeddings);
    config.audio_embedding_size = extract_json_int(json, "audio_embedding_size", base.hidden_size);

    config.audio_token_id = extract_json_int(json, "audio_token_id", config.audio_token_id);
    config.audio_start_token_id =
        extract_json_int(json, "audio_start_token_id", config.audio_start_token_id);
    config.audio_end_token_id =
        extract_json_int(json, "audio_end_token_id", config.audio_end_token_id);
    config.user_token_id = extract_json_int(json, "user_token_id", config.user_token_id);
    config.assistant_token_id =
        extract_json_int(json, "assistant_token_id", config.assistant_token_id);

    config.vocab_size = base.vocab_size > 0 ? base.vocab_size : config.vocab_size;
    config.max_cache_length = base.max_cache_length;
    config.eos_token_id = extract_json_int(json, "eos_token_id", config.eos_token_id);
    config.transcription_prompt =
        extract_json_string(json, "transcription_prompt", config.transcription_prompt);
    return config;
}

} // namespace

class GlmAsrPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        auto decoder_modules = load_dual_profile_modules(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "glmasr decoder", opts);
        cudaStream_t stream = decoder_modules.decode->stream();

        auto encoder_loaded =
            load_trt_module_from_plan(ctx.backend, find_section(ctx.bundle, "vision_engine_plan"),
                                      "glmasr audio encoder", opts);
        if (!encoder_loaded.module || !encoder_loaded.module->ok())
            throw std::runtime_error("GlmAsrPipeline: bundle is missing the audio encoder engine");

        const int32_t kv_dim = compute_kv_dim(ctx.config);
        const DType cache_dtype = decoder_modules.decode->tensor_dtype("cache_k_0");
        auto state = std::make_unique<GlmAsrKvCache>(
            ctx.config.num_layers, ctx.config.max_cache_length, kv_dim, stream, cache_dtype);
        if (!state->ok())
            throw std::runtime_error("GlmAsrPipeline: failed to create GlmAsrKvCache");

        GlmAsrConfig config = read_config(ctx.config_json, ctx.config);
        MelFilterbank mel_filterbank = load_mel_filterbank(ctx.bundle);
        if (mel_filterbank.data.empty())
            throw std::runtime_error("GlmAsrPipeline: bundle is missing the mel filterbank");

        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        return std::make_unique<GlmAsrPipeline>(
            std::move(encoder_loaded.module), std::move(decoder_modules.decode), std::move(state),
            std::move(config), std::move(mel_filterbank), stream, std::move(tokenizer),
            ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_glmasr_plugin, GlmAsrPlugin,
                                       "glmasr_speech_to_text");

} // namespace trtmc

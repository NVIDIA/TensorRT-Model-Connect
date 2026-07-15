/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Qwen3OmniPlugin: handles "qwen3_omni_multimodal" strategy.
// Omni pipeline with TensorRT Thinker/Code2Wav and the official model-owned Talker bridge.

#include "plugin_helpers.h"
#include "runtime/models/qwen3_omni/pipeline.h"
#include "runtime/models/qwen3_omni/recurrent_state.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstdlib>

namespace trtmc {

class Qwen3OmniPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto& json = ctx.config_json;

        // Thinker (MoE decoder) -- main engine plan
        auto thinker_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "omni thinker", opts);
        cudaStream_t stream = thinker_loaded.module->stream();
        int32_t kv_dim = compute_kv_dim(ctx.config);
        // The cache allocation must follow the engine binding, not the bundle's
        // requested precision. Legacy Qwen3-Omni builders emitted FP32 cache
        // bindings even for a bundle labelled BF16; allocating BF16 here
        // truncated every present K/V row and corrupted subsequent tokens.
        DType cache_dtype = thinker_loaded.module->tensor_dtype("cache_k_0");
        std::unique_ptr<Qwen3OmniInferenceState> thinker_state = std::make_unique<Qwen3OmniKvCache>(
            ctx.config.num_layers, ctx.config.max_cache_length, kv_dim, stream, cache_dtype);
        if (!thinker_state->ok())
            throw std::runtime_error("OmniPipeline: failed to create thinker Qwen3OmniKvCache");

        // Code2Wav is required for the audio-capable Qwen3-Omni bundle.
        std::unique_ptr<TrtModule> code2wav_module;
        auto code2wav_loaded = try_load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "code2wav_engine_plan"), "code2wav", opts);
        if (code2wav_loaded.module && code2wav_loaded.module->ok())
            code2wav_module = std::move(code2wav_loaded.module);
        if (!code2wav_module)
            throw std::runtime_error(
                "OmniPipeline: required official Code2Wav engine is missing from bundle");

        // Build OmniConfig
        OmniConfig omni_cfg;
        omni_cfg.sample_rate = extract_json_int(json, "audio_sample_rate", 24000);
        omni_cfg.thinker_hidden_size = ctx.config.hidden_size;
        omni_cfg.thinker_num_layers = ctx.config.num_layers;
        omni_cfg.thinker_num_heads = ctx.config.num_heads;
        omni_cfg.num_experts = extract_json_int(json, "num_local_experts", 8);
        omni_cfg.num_experts_per_tok = extract_json_int(json, "num_experts_per_tok", 2);
        omni_cfg.talker_hidden_size = extract_json_int(json, "omni_talker_hidden_size", 0);
        omni_cfg.talker_num_layers = extract_json_int(json, "omni_talker_num_layers", 0);
        omni_cfg.talker_n_codebooks = extract_json_int(json, "omni_n_codebooks", 16);
        omni_cfg.talker_codebook_size = extract_json_int(json, "omni_codebook_size", 2048);
        omni_cfg.code2wav_max_frames = extract_json_int(json, "omni_code2wav_max_frames", 32);
        omni_cfg.code2wav_upsample_factor =
            extract_json_int(json, "omni_code2wav_upsample_factor", 1920);
        omni_cfg.code2wav_output_delay = extract_json_int(json, "omni_code2wav_output_delay", 555);
        omni_cfg.hf_python = ctx.hf_python;
        omni_cfg.talker_model_id = extract_json_string(
            json, "omni_talker_model_id",
            extract_json_string(json, "omni_talker_model_path", ctx.bundle.info.model_id));
        omni_cfg.talker_model_revision =
            extract_json_string(json, "omni_talker_model_revision", "");
        if (const char* override_path = std::getenv("TRTMC_QWEN3_OMNI_MODEL_PATH");
            override_path != nullptr && override_path[0] != '\0') {
            omni_cfg.talker_model_id = override_path;
            omni_cfg.talker_model_revision.clear();
        }

        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        return std::make_unique<OmniPipeline>(
            std::move(thinker_loaded.module), std::move(thinker_state), std::move(code2wav_module),
            std::move(omni_cfg), stream, std::move(tokenizer), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_qwen3_omni_plugin, Qwen3OmniPlugin,
                                       "qwen3_omni_multimodal");

} // namespace trtmc

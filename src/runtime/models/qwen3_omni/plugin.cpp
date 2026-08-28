/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Qwen3OmniPlugin: handles "qwen3_omni_multimodal" strategy.
// Text-only Omni pipeline with the TensorRT Thinker. Audio generation fails
// closed until a complete native Talker path is available.

#include "plugin_helpers.h"
#include "runtime/models/qwen3_omni/pipeline.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

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
        auto thinker_modules = load_dual_profile_modules(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "omni thinker", opts);
        cudaStream_t stream = thinker_modules.decode->stream();
        int32_t kv_dim = compute_kv_dim(ctx.config);
        // The cache allocation must follow the engine binding, not the bundle's
        // requested precision. Legacy Qwen3-Omni builders emitted FP32 cache
        // bindings even for a bundle labelled BF16; allocating BF16 here
        // truncated every present K/V row and corrupted subsequent tokens.
        DType cache_dtype = thinker_modules.decode->tensor_dtype("cache_k_0");
        std::unique_ptr<Qwen3OmniInferenceState> thinker_state = std::make_unique<Qwen3OmniKvCache>(
            ctx.config.num_layers, ctx.config.max_cache_length, kv_dim, stream, cache_dtype);
        if (!thinker_state->ok())
            throw std::runtime_error("OmniPipeline: failed to create thinker Qwen3OmniKvCache");

        // Build OmniConfig
        OmniConfig omni_cfg;
        omni_cfg.sample_rate = extract_json_int(json, "audio_sample_rate", 24000);
        omni_cfg.thinker_hidden_size = ctx.config.hidden_size;
        omni_cfg.thinker_vocab_size = ctx.config.vocab_size;
        omni_cfg.thinker_num_layers = ctx.config.num_layers;
        omni_cfg.thinker_num_heads = ctx.config.num_heads;
        omni_cfg.thinker_eos_token_id = extract_json_int(json, "im_end_token_id", 151645);
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
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);

        return std::make_unique<OmniPipeline>(
            std::move(thinker_modules.decode), std::move(thinker_state), nullptr,
            std::move(omni_cfg), stream, std::move(tokenizer), ctx.bundle.info.model_id,
            std::move(thinker_modules.prefill));
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_qwen3_omni_plugin, Qwen3OmniPlugin,
                                       "qwen3_omni_multimodal");

} // namespace trtmc

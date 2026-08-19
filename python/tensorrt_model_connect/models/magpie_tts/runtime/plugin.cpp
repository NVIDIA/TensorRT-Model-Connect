/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// MagpiePlugin: handles "text_to_audio_magpie" strategy.
// Magpie TTS encoder-decoder pipeline with IPA tokenizer and optional CFG.

#include "audio_helpers.h"
#include "plugin_helpers.h"
#include "pipeline.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <vector>

namespace trtmc {

namespace {

struct TensorParallelRuntimeConfig {
    bool enabled{false};
    int32_t tp_size{1};
};

struct MagpieDecoderRuntime {
    DistributedRuntimeGroup tp_group;
    std::string section{"engine_plan"};
};

TensorParallelRuntimeConfig parse_tensor_parallel_runtime_config(const std::string& config_json) {
    TensorParallelRuntimeConfig cfg;
    cfg.tp_size = extract_json_int(config_json, "tensor_parallel_size", 1);
    const auto mode = extract_json_string(config_json, "tensor_parallel_mode", "single");
    cfg.enabled = (mode == "tensor_parallel" && cfg.tp_size > 1);
    return cfg;
}

std::string tp_engine_section_name(int32_t rank) {
    return "engine_plan_tp_rank" + std::to_string(rank);
}

int32_t dim_at(const std::vector<int64_t>& shape, int32_t dim) {
    if (dim < 0 || static_cast<std::size_t>(dim) >= shape.size())
        return -1;
    const auto value = shape[static_cast<std::size_t>(dim)];
    if (value <= 0 || value > std::numeric_limits<int32_t>::max())
        return -1;
    return static_cast<int32_t>(value);
}

int32_t compute_magpie_kv_dim(const BaseConfig& base_cfg, const MagpieTTSConfig& magpie_cfg) {
    if (base_cfg.num_kv_heads > 0 && base_cfg.head_dim > 0)
        return base_cfg.num_kv_heads * base_cfg.head_dim;
    if (base_cfg.attention_size > 0)
        return base_cfg.attention_size;
    return magpie_cfg.hidden_size;
}

int32_t decoder_cache_row_width(const TrtModule& module, int32_t fallback) {
    const int32_t from_engine = dim_at(module.tensor_shape("cache_k_0"), 1);
    return from_engine > 0 ? from_engine : fallback;
}

MagpieDecoderRuntime make_magpie_decoder_runtime(const PipelineContext& ctx) {
    MagpieDecoderRuntime runtime;

    const auto tp_config = parse_tensor_parallel_runtime_config(ctx.config_json);
    if (!tp_config.enabled)
        return runtime;

    runtime.tp_group = initialize_tensor_parallel_group(tp_config.tp_size);
    runtime.section = tp_engine_section_name(runtime.tp_group.rank);
    return runtime;
}

} // namespace

// audio_magpie.* registry overlay — replaces the TRTMC_MAGPIE_{GREEDY,
// CFG_SCALE,TEMPERATURE,FINISHED_LIMIT,SEED} env vars. Only apply
// non-default registry values so pre-migration bundles keep their
// config-derived defaults for fields the caller never touched.
static void apply_magpie_registry_overlay(MagpieTTSConfig& magpie_cfg,
                                          const config::ConfigBundle* cfg) {
    if (cfg == nullptr)
        return;
    try {
        if (cfg->source_of("audio_magpie", "greedy") != config::Layer::SchemaDefault)
            magpie_cfg.greedy = cfg->get<bool>("audio_magpie", "greedy");
        const float cfg_scale = cfg->get<float>("audio_magpie", "cfg_scale");
        if (cfg_scale > 0.0F)
            magpie_cfg.cfg_scale = cfg_scale;
        const float temp = cfg->get<float>("audio_magpie", "temperature");
        if (temp > 0.0F)
            magpie_cfg.temperature = temp;
        const std::int32_t finished_limit =
            cfg->get<std::int32_t>("audio_magpie", "finished_limit");
        if (finished_limit >= 0) {
            magpie_cfg.finished_limit_with_eot = finished_limit;
            magpie_cfg.enable_finished_limit_stop = (finished_limit > 0);
        }
        magpie_cfg.seed = cfg->get<std::int64_t>("audio_magpie", "seed");
    } catch (const std::exception&) {
        // Schema not registered or type mismatch — leave defaults.
    }
}

class MagpiePlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        auto decoder_runtime = make_magpie_decoder_runtime(ctx);
        auto shared_stream = std::make_shared<MagpieCudaStream>();
        if (!shared_stream->ok())
            throw std::runtime_error("MagpiePlugin: failed to create CUDA stream");

        ModuleCreateOptions opts;
        opts.stream = shared_stream->get();
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;
        ModuleCreateOptions decoder_opts = opts;
        if (decoder_runtime.tp_group.communicator != nullptr) {
            decoder_opts.distributed_communicator = decoder_runtime.tp_group.communicator;
            decoder_opts.distributed_owner = decoder_runtime.tp_group.owner;
        }

        auto enc_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "vision_engine_plan"), "magpie encoder", opts);
        auto dec_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, decoder_runtime.section), "magpie decoder",
            decoder_opts);
        enc_loaded.module->keep_alive(shared_stream);
        dec_loaded.module->keep_alive(shared_stream);

        cudaStream_t stream = enc_loaded.module->stream();

        auto magpie_cfg = build_magpie_config(ctx.config_json, ctx.config);
        apply_magpie_registry_overlay(magpie_cfg, ctx.runtime_config);
        const int32_t fallback_kv_dim = compute_magpie_kv_dim(ctx.config, magpie_cfg);
        int32_t kv_dim = decoder_cache_row_width(*dec_loaded.module, fallback_kv_dim);

        DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        std::unique_ptr<MagpieInferenceState> decoder_state = std::make_unique<MagpieKvCache>(
            magpie_cfg.decoder_layers, ctx.config.max_cache_length, kv_dim, stream, cache_dtype);
        if (!decoder_state->ok())
            throw std::runtime_error("MagpiePipeline: failed to create decoder MagpieKvCache");

        std::unique_ptr<MagpieInferenceState> decoder_state_uncond;
        if (magpie_cfg.cfg_scale > 1.0F) {
            decoder_state_uncond = std::make_unique<MagpieKvCache>(magpie_cfg.decoder_layers,
                                                                   ctx.config.max_cache_length,
                                                                   kv_dim, stream, cache_dtype);
        }

        const std::size_t enc_buf_size = static_cast<std::size_t>(magpie_cfg.max_source_positions) *
                                         static_cast<std::size_t>(magpie_cfg.hidden_size) *
                                         sizeof(float);

        std::vector<MagpieCudaBuffer> cross_k, cross_v;
        allocate_cross_kv_buffers(magpie_cfg.decoder_layers, enc_buf_size, cross_k, cross_v);

        std::vector<MagpieCudaBuffer> cross_k_uncond, cross_v_uncond;
        if (magpie_cfg.cfg_scale > 1.0F)
            allocate_cross_kv_buffers(magpie_cfg.decoder_layers, enc_buf_size, cross_k_uncond,
                                      cross_v_uncond);

        MagpieCudaBuffer encoder_output(enc_buf_size);
        MagpieCudaBuffer encoder_output_uncond(magpie_cfg.cfg_scale > 1.0F ? enc_buf_size : 0);

        auto codec_module = extract_optional_module(
            ctx.backend, find_section(ctx.bundle, "codec_engine_plan"), "magpie codec", opts);
        if (codec_module)
            codec_module->keep_alive(shared_stream);

        auto lt_module =
            extract_optional_module(ctx.backend, find_section(ctx.bundle, "lt_engine_plan"),
                                    "magpie local transformer", opts);
        if (lt_module)
            lt_module->keep_alive(shared_stream);

        // Backend module creation does not yet expose optimization profile selection,
        // so the profile-1 prefill module cannot be created through ctx.backend.
        std::unique_ptr<TrtModule> prefill_module;

        auto tok = make_ipa_tok(ctx.bundle);

        return std::make_unique<MagpiePipeline>(
            std::move(enc_loaded.module), std::move(dec_loaded.module), std::move(decoder_state),
            std::move(codec_module), std::move(lt_module), std::move(prefill_module),
            std::move(decoder_state_uncond), std::move(cross_k), std::move(cross_v),
            std::move(cross_k_uncond), std::move(cross_v_uncond), std::move(encoder_output),
            std::move(encoder_output_uncond),
            section_to_floats(find_section(ctx.bundle, "magpie_audio_embed")),
            section_to_floats(find_section(ctx.bundle, "magpie_text_embed")),
            section_to_floats(find_section(ctx.bundle, "magpie_context_embed")),
            section_to_int32s(find_section(ctx.bundle, "magpie_context_lengths")),
            std::move(magpie_cfg), stream, std::move(tok), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_magpie_plugin, MagpiePlugin,
                                       "text_to_audio_magpie");

} // namespace trtmc

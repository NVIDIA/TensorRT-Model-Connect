/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// BarkPlugin: handles "text_to_audio_bark" strategy.
// Bark semantic + coarse pipeline with optional codec and fine engines.

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

std::string coarse_tp_engine_section_name(int32_t rank) {
    return "coarse_engine_plan_tp_rank" + std::to_string(rank);
}

int32_t dim_at(const std::vector<int64_t>& shape, int32_t dim) {
    if (dim < 0 || static_cast<std::size_t>(dim) >= shape.size())
        return -1;
    const auto value = shape[static_cast<std::size_t>(dim)];
    if (value <= 0 || value > std::numeric_limits<int32_t>::max())
        return -1;
    return static_cast<int32_t>(value);
}

int32_t decoder_cache_row_width(const TrtModule& module, int32_t fallback) {
    const int32_t from_engine = dim_at(module.tensor_shape("cache_k_0"), 1);
    return from_engine > 0 ? from_engine : fallback;
}

std::unique_ptr<BarkKvCache> make_bark_coarse_kv_cache(const std::string& json,
                                                       const BaseConfig& base,
                                                       const TrtModule& module, cudaStream_t stream,
                                                       DType cache_dtype) {
    int32_t hidden = extract_json_int(json, "coarse_hidden_size", base.hidden_size);
    int32_t layers = extract_json_int(json, "coarse_num_layers", base.num_layers);
    int32_t heads = extract_json_int(json, "coarse_num_heads", base.num_heads);
    int32_t hd = (heads > 0) ? hidden / heads : 128;
    int32_t max_cache = extract_json_int(json, "coarse_max_cache_length", base.max_cache_length);
    const int32_t row_width = decoder_cache_row_width(module, heads * hd);
    return std::make_unique<BarkKvCache>(layers, max_cache, row_width, stream, cache_dtype);
}

BarkConfig make_bark_config(const PipelineContext& ctx) {
    const auto& json = ctx.config_json;
    BarkConfig bark_cfg;
    bark_cfg.sample_rate = extract_json_int(json, "sample_rate", 24000);
    bark_cfg.hidden_size = ctx.config.hidden_size;
    bark_cfg.semantic_input_vocab = extract_json_int(json, "semantic_input_vocab", 129600);
    bark_cfg.semantic_output_vocab = ctx.config.vocab_size;
    bark_cfg.text_encoding_offset = extract_json_int(json, "text_encoding_offset", 10048);
    bark_cfg.text_pad_token = extract_json_int(json, "text_pad_token", 129595);
    bark_cfg.semantic_pad_token = extract_json_int(json, "semantic_pad_token", 10000);
    bark_cfg.semantic_infer_token = extract_json_int(json, "semantic_infer_token", 129599);
    bark_cfg.semantic_vocab_size = extract_json_int(json, "semantic_vocab_size", 10000);
    bark_cfg.coarse_input_vocab = extract_json_int(json, "coarse_input_vocab", 12096);
    bark_cfg.coarse_semantic_pad_token = extract_json_int(json, "coarse_semantic_pad_token", 12048);
    bark_cfg.coarse_infer_token = extract_json_int(json, "coarse_infer_token", 12050);
    bark_cfg.n_coarse_codebooks = extract_json_int(json, "n_coarse_codebooks", 2);
    bark_cfg.codebook_size = extract_json_int(json, "codebook_size", 1024);
    bark_cfg.codec_seq_length = extract_json_int(json, "codec_seq_length", 0);
    bark_cfg.codec_upsample_factor = extract_json_int(json, "codec_upsample_factor", 320);
    bark_cfg.codec_n_codebooks = extract_json_int(json, "codec_n_codebooks", 8);
    bark_cfg.fine_hidden_size = extract_json_int(json, "fine_hidden_size", ctx.config.hidden_size);
    bark_cfg.fine_n_lm_heads = extract_json_int(json, "fine_n_lm_heads", 7);
    bark_cfg.fine_codebook_size = extract_json_int(json, "fine_codebook_size", 1056);
    bark_cfg.fine_seq_length = extract_json_int(json, "fine_seq_length", 0);

    // audio_bark.* namespace (replaces TRTMC_BARK_{DUMP,GREEDY,SEED}).
    if (ctx.runtime_config != nullptr) {
        try {
            bark_cfg.dump_path = ctx.runtime_config->get<std::string>("audio_bark", "dump_path");
            bark_cfg.greedy = ctx.runtime_config->get<bool>("audio_bark", "greedy");
            bark_cfg.seed = ctx.runtime_config->get<std::int64_t>("audio_bark", "seed");
            bark_cfg.fine_temperature =
                ctx.runtime_config->get<float>("audio_bark", "fine_temperature");
        } catch (const std::exception&) {
            // Schema not registered or type mismatch: stay at defaults.
        }
    }

    return bark_cfg;
}

void attach_optional_bark_modules(BarkPipeline& pipeline, const PipelineContext& ctx,
                                  const ModuleCreateOptions& opts) {
    auto codec_loaded = try_load_trt_module_from_plan(
        ctx.backend, find_section(ctx.bundle, "codec_engine_plan"), "bark codec", opts);
    if (codec_loaded.module && codec_loaded.module->ok())
        pipeline.set_codec_module(std::move(codec_loaded.module));

    auto fine_loaded = try_load_trt_module_from_plan(
        ctx.backend, find_section(ctx.bundle, "fine_engine_plan"), "bark fine", opts);
    if (fine_loaded.module && fine_loaded.module->ok()) {
        pipeline.set_fine_module(std::move(fine_loaded.module));
        auto fe = section_to_floats(find_section(ctx.bundle, "fine_embed"));
        auto fp = section_to_floats(find_section(ctx.bundle, "fine_position_embed"));
        if (!fe.empty())
            pipeline.set_fine_embeddings(std::move(fe), std::move(fp));
    }
}

} // namespace

class BarkPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto& json = ctx.config_json;
        const auto tp_config = parse_tensor_parallel_runtime_config(json);
        DistributedRuntimeGroup tp_group;
        ModuleCreateOptions decoder_opts = opts;
        if (tp_config.enabled) {
            tp_group = initialize_tensor_parallel_group(tp_config.tp_size);
            decoder_opts.distributed_communicator = tp_group.communicator;
            decoder_opts.distributed_owner = tp_group.owner;
        }

        // Load semantic engine (main plan or rank-local TP section)
        const std::string semantic_section =
            tp_config.enabled ? tp_engine_section_name(tp_group.rank) : std::string("engine_plan");
        auto sem_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, semantic_section), "bark semantic", decoder_opts);

        // Load coarse engine (single plan or rank-local TP section)
        const std::string coarse_section = tp_config.enabled
                                               ? coarse_tp_engine_section_name(tp_group.rank)
                                               : std::string("coarse_engine_plan");
        auto coarse_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, coarse_section), "bark coarse", decoder_opts);

        cudaStream_t stream = sem_loaded.module->stream();
        BarkConfig bark_cfg = make_bark_config(ctx);

        // Create KvCaches for semantic and coarse stages
        int32_t sem_kv_dim =
            decoder_cache_row_width(*sem_loaded.module, compute_kv_dim(ctx.config));
        DType cache_dtype = cache_dtype_from_precision(ctx.config.precision);
        std::unique_ptr<BarkInferenceState> sem_state = std::make_unique<BarkKvCache>(
            ctx.config.num_layers, ctx.config.max_cache_length, sem_kv_dim, stream, cache_dtype);

        // Coarse engine may have different dimensions -- resolve with semantic fallbacks
        std::unique_ptr<BarkInferenceState> coarse_state(make_bark_coarse_kv_cache(
            json, ctx.config, *coarse_loaded.module, stream, cache_dtype));

        // Load embeddings
        auto sem_embed = section_to_floats(find_section(ctx.bundle, "semantic_embed"));
        auto coarse_embed = section_to_floats(find_section(ctx.bundle, "coarse_embed"));

        // Bark uses BertTokenizer WITHOUT special tokens ([CLS]/[SEP]).
        // HF's BarkProcessor calls encode(text, add_special_tokens=False).
        // Always use add_special_tokens=false to match the HF Bark pipeline.
        auto bark_tokenizer = try_create_native_tokenizer(ctx.bundle, /*add_special_tokens=*/false);

        auto pipeline = std::make_unique<BarkPipeline>(
            std::move(sem_loaded.module), std::move(coarse_loaded.module), std::move(sem_state),
            std::move(coarse_state), std::move(sem_embed), std::move(coarse_embed),
            std::move(bark_cfg), stream, std::move(bark_tokenizer), ctx.bundle.info.model_id);

        attach_optional_bark_modules(*pipeline, ctx, opts);

        return pipeline;
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_bark_plugin, BarkPlugin, "text_to_audio_bark");

} // namespace trtmc

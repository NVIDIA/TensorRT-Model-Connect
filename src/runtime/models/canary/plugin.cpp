/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CanaryPlugin: handles "canary_speech_to_text" strategy.
// Canary encoder-decoder pipeline with mel spectrogram input.

#include "plugin_helpers.h"
#include "runtime/models/canary/pipeline.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <algorithm>
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

int32_t dim_at(const std::vector<int64_t>& shape, int32_t dim) {
    if (dim < 0 || static_cast<std::size_t>(dim) >= shape.size())
        return -1;
    const auto value = shape[static_cast<std::size_t>(dim)];
    if (value <= 0 || value > std::numeric_limits<int32_t>::max())
        return -1;
    return static_cast<int32_t>(value);
}

int32_t decoder_cache_row_width(const TrtModule& module, const BaseConfig& config) {
    const auto shape = module.tensor_shape("cache_k_0");
    const int32_t from_engine =
        shape.empty() ? -1 : dim_at(shape, static_cast<int32_t>(shape.size()) - 1);
    return from_engine > 0 ? from_engine : compute_kv_dim(config);
}

int32_t decoder_batch_capacity(const TrtModule& module) {
    if (!module.input_is_dynamic("token_id"))
        return 1;
    const auto shape =
        module.input_profile_shape("token_id", module.profile_idx(), ProfileShapeSelector::kMax);
    return std::max(dim_at(shape, 0), 1);
}

bool canary_cuda_graph_disabled(const PipelineContext& ctx) {
    if (ctx.runtime_config == nullptr)
        return false;
    try {
        return ctx.runtime_config->get<bool>("runtime", "disable_cuda_graph");
    } catch (const std::exception&) {
        // Schema not registered — retain the default graph policy.
        return false;
    }
}

} // namespace

class CanaryPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        opts.cuda_graphs = ctx.cuda_graphs;

        const auto& json = ctx.config_json;
        const auto tp_config = parse_tensor_parallel_runtime_config(json);
        DistributedRuntimeGroup tp_group;
        if (tp_config.enabled)
            tp_group = initialize_tensor_parallel_group(tp_config.tp_size);

        // Load encoder (stored as vision_engine_plan in Canary bundles)
        const auto* enc_plan = find_section(ctx.bundle, "vision_engine_plan");
        if (!enc_plan || enc_plan->empty())
            enc_plan = find_section(ctx.bundle, "coarse_engine_plan");
        auto enc_loaded = load_trt_module_from_plan(ctx.backend, enc_plan, "canary encoder", opts);

        if (tp_config.enabled) {
            opts.distributed_communicator = tp_group.communicator;
            opts.distributed_owner = tp_group.owner;
        }

        // Load decoder (main engine_plan or rank-local TP section)
        const std::string decoder_section =
            tp_config.enabled ? tp_engine_section_name(tp_group.rank) : std::string("engine_plan");
        auto dec_loaded = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, decoder_section), "canary decoder", opts);

        // Build CanaryConfig
        int32_t encoder_layers = extract_json_int(json, "encoder_layers", ctx.config.num_layers);
        int32_t decoder_layers = extract_json_int(json, "decoder_layers", ctx.config.num_layers);
        int32_t dl = (decoder_layers > 0) ? decoder_layers : ctx.config.num_layers;
        CanaryConfig wc;
        wc.num_mel_bins = extract_json_int(json, "num_mel_bins", 80);
        wc.max_source_positions = extract_json_int(json, "max_source_positions", 1500);
        wc.max_target_positions = extract_json_int(json, "max_target_positions", 448);
        wc.encoder_layers = encoder_layers;
        wc.decoder_layers = dl;
        int32_t eot_token_id = extract_json_int(json, "eot_token_id", -1);
        wc.eot_token_id = (eot_token_id >= 0) ? eot_token_id : ctx.config.id_eos;
        wc.mel_length = extract_json_int(json, "mel_length", 0);
        wc.decoder_start_token_ids = extract_json_int_array(json, "decoder_start_token_ids", 256);
        wc.supported_languages = extract_json_string_array(json, "canary_supported_languages");
        wc.language_token_ids = extract_json_int_array(json, "canary_language_token_ids", 256);
        wc.source_language_position = extract_json_int(json, "canary_source_language_position", 4);
        wc.target_language_position = extract_json_int(json, "canary_target_language_position", 5);
        wc.punctuation_position = extract_json_int(json, "canary_punctuation_position", 6);
        wc.timestamp_position = extract_json_int(json, "canary_timestamp_position", 8);
        wc.punctuation_token_id = extract_json_int(json, "canary_punctuation_token_id", -1);
        wc.no_punctuation_token_id = extract_json_int(json, "canary_no_punctuation_token_id", -1);
        wc.timestamp_token_id = extract_json_int(json, "canary_timestamp_token_id", -1);
        wc.no_timestamp_token_id = extract_json_int(json, "canary_no_timestamp_token_id", -1);
        wc.translation_requires_english =
            extract_json_bool(json, "canary_translation_requires_english", true);
        wc.disable_cuda_graph = canary_cuda_graph_disabled(ctx);

        // Create CanaryKvCache for decoder self-attention
        cudaStream_t stream = dec_loaded.module->stream();
        int32_t kv_dim = decoder_cache_row_width(*dec_loaded.module, ctx.config);
        int32_t max_cache = ctx.config.max_cache_length;
        DType cache_dtype = dec_loaded.module->tensor_dtype("cache_k_0");
        const int32_t batch_capacity = decoder_batch_capacity(*dec_loaded.module);
        std::unique_ptr<CanaryInferenceState> state = std::make_unique<CanaryKvCache>(
            dl, max_cache, kv_dim, stream, cache_dtype, batch_capacity);
        if (!state->ok())
            throw std::runtime_error("Failed to create CanaryKvCache for Canary decoder");

        // Load mel filterbank + tokenizer
        auto mel_fb = load_mel_filterbank(ctx.bundle);
        auto tok = create_tokenizer_from_bundle(ctx.bundle);

        int32_t mel_n_fft = extract_json_int(json, "mel_n_fft", 400);
        int32_t mel_hop_length = extract_json_int(json, "mel_hop_length", 160);
        int32_t mel_chunk_length = extract_json_int(json, "mel_chunk_length", 30);
        int32_t mel_sampling_rate = extract_json_int(json, "mel_sampling_rate", 16000);

        return std::make_unique<CanaryPipeline>(
            std::move(enc_loaded.module), std::move(dec_loaded.module), std::move(state),
            std::move(wc), ctx.config.hidden_size, dl, std::move(mel_fb), mel_n_fft, mel_hop_length,
            mel_chunk_length, mel_sampling_rate, stream, std::move(tok), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_canary_plugin, CanaryPlugin,
                                       "canary_speech_to_text");

} // namespace trtmc

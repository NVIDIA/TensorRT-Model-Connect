/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_view.h"
#include "runtime/models/lfm2/hybrid_state.h"
#include "runtime/models/lfm2/plugin_helpers.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace trtmc {
namespace {

struct Lfm2Geometry {
    int32_t num_attention_layers{0};
    int32_t num_conv_layers{0};
    int32_t conv_cache_length{0};
    int32_t conv_state_dim{0};
};

void validate_lfm2_native_contract(const PipelineContext& ctx) {
    if (!extract_json_bool(ctx.config_json, "native_kv_cache", false) ||
        extract_json_int(ctx.config_json, "native_kv_contract_version", 0) != 1) {
        throw std::runtime_error("LFM2 bundle must declare native_kv_cache contract version 1");
    }
    if (ctx.kv_cache_size_bytes != 0) {
        throw std::runtime_error(
            "LFM2 native KV cache has fixed engine capacity and does not support "
            "--kv-cache-size");
    }
}

Lfm2Geometry parse_lfm2_geometry(const PipelineContext& ctx) {
    Lfm2Geometry geometry;
    geometry.num_attention_layers = extract_json_int(ctx.config_json, "num_attention_layers", 0);
    geometry.num_conv_layers = extract_json_int(ctx.config_json, "num_conv_layers", 0);
    geometry.conv_cache_length = extract_json_int(ctx.config_json, "conv_L_cache", 0);
    geometry.conv_state_dim = extract_json_int(ctx.config_json, "conv_dim", ctx.config.hidden_size);
    if (geometry.num_attention_layers <= 0 || geometry.num_conv_layers <= 0 ||
        geometry.num_attention_layers + geometry.num_conv_layers != ctx.config.num_layers) {
        throw std::runtime_error("LFM2 bundle has inconsistent hybrid layer counts");
    }
    if (geometry.conv_cache_length <= 0 || geometry.conv_state_dim != ctx.config.hidden_size)
        throw std::runtime_error("LFM2 bundle has unsupported convolution state geometry");
    return geometry;
}

void validate_lfm2_attention_geometry(const BaseConfig& config) {
    if (config.num_kv_heads <= 0 || config.head_dim <= 0 || config.max_cache_length <= 0)
        throw std::runtime_error("LFM2 bundle has invalid attention cache geometry");
}

std::unique_ptr<Lfm2HybridState> create_lfm2_state(const PipelineContext& ctx,
                                                   const Lfm2Geometry& geometry,
                                                   cudaStream_t stream, DType state_dtype) {
    auto kv = std::make_unique<Lfm2KvCache>(
        geometry.num_attention_layers, ctx.config.max_cache_length, ctx.config.num_kv_heads,
        ctx.config.head_dim, stream, state_dtype,
        lfm2_kv_names(ctx.config, geometry.num_attention_layers));
    auto conv = std::make_unique<Lfm2ConvState>(geometry.num_conv_layers, geometry.conv_state_dim,
                                                geometry.conv_cache_length, stream, state_dtype);
    auto state = std::make_unique<Lfm2HybridState>(std::move(kv), std::move(conv));
    if (!state->ok())
        throw std::runtime_error("Failed to allocate LFM2 hybrid inference state");
    return state;
}

Lfm2TextGenConfig create_lfm2_generation_config(const PipelineContext& ctx) {
    Lfm2TextGenConfig generation;
    generation.vocab_size = ctx.config.vocab_size;
    generation.id_bos = ctx.config.id_bos;
    generation.id_eos = ctx.config.id_eos;
    generation.id_eos_ids = ctx.config.id_eos_ids;
    generation.token_id_name = ctx.config.io_map.token_id;
    generation.logits_output_name = ctx.config.io_map.logits;
    apply_lfm2_chat_template_format(ctx.bundle, generation);
    return generation;
}

} // namespace

class Lfm2Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        validate_lfm2_native_contract(ctx);
        const auto geometry = parse_lfm2_geometry(ctx);
        validate_lfm2_attention_geometry(ctx.config);

        ModuleCreateOptions options;
        options.runtime_cache_path = ctx.runtime_cache_path.c_str();
        options.cuda_graphs = ctx.cuda_graphs;
        auto decoder =
            load_lfm2_module(ctx.backend, find_section(ctx.bundle, "engine_plan"), options);
        const cudaStream_t stream = decoder->stream();
        const DType state_dtype = lfm2_state_dtype(ctx.config.precision);

        auto state = create_lfm2_state(ctx, geometry, stream, state_dtype);
        auto generation = create_lfm2_generation_config(ctx);
        auto tokenizer = create_lfm2_tokenizer(ctx.bundle);
        return std::make_unique<Lfm2TextGenerationPipeline>(
            std::move(decoder), std::move(state), std::move(generation), stream,
            std::move(tokenizer), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_lfm2_plugin, Lfm2Plugin,
                                       "lfm2_hybrid_conv_attention");

} // namespace trtmc

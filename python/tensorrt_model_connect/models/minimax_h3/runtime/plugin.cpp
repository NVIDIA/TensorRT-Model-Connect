/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "pipeline.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/tokenizer.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace trtmc {
namespace {

using SectionMap = std::unordered_map<std::string, BundleSectionInfo>;

struct CacheConfig {
    bool enabled{false};
    float threshold{0.025F};
};

SectionMap index_sections(const BundleInfo& info, bool first_block_cache) {
    constexpr std::array<const char*, 4> monolithic_names = {
        "text_encoder_plan", "adaln_precompute_plan", "denoiser_plan", "vae_tile_decoder_plan"};
    constexpr std::array<const char*, 6> first_block_cache_names = {
        "text_encoder_plan",  "adaln_precompute_plan", "denoiser_head_plan",
        "denoiser_tail_plan", "denoiser_finish_plan",  "vae_tile_decoder_plan"};
    SectionMap sections;
    const auto add_section = [&](const char* name) {
        const auto it =
            std::find_if(info.sections.begin(), info.sections.end(),
                         [name](const BundleSectionInfo& item) { return item.name == name; });
        if (it == info.sections.end() || it->size == 0)
            throw std::runtime_error(std::string("MiniMax-H3 bundle is missing ") + name);
        sections.emplace(name, *it);
    };
    if (first_block_cache) {
        for (const char* name : first_block_cache_names)
            add_section(name);
    } else {
        for (const char* name : monolithic_names)
            add_section(name);
    }
    return sections;
}

std::unique_ptr<ITokenizer> load_tokenizer(const BundleFile& bundle) {
    const auto* data = find_section(bundle, "tokenizer.json");
    if (data == nullptr || data->empty())
        throw std::runtime_error("MiniMax-H3 bundle is missing tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(data->data(), data->size(), false);
    if (!tokenizer)
        throw std::runtime_error("MiniMax-H3 could not create the native Qwen BPE tokenizer");
    return tokenizer;
}

void validate_profile(const PipelineContext& ctx) {
    if (ctx.backend == nullptr)
        throw std::runtime_error("MiniMax-H3 requires the TensorRT backend");
    if (extract_json_int(ctx.config_json, "context_parallel_size", 1) != 1)
        throw std::runtime_error("MiniMax-H3 requires context_parallel_size=1");
    if (extract_json_int(ctx.config_json, "padded_sequence_length", 38247) != 38247)
        throw std::runtime_error("MiniMax-H3 requires 38247 unpadded sequence rows");
    if (extract_json_int(ctx.config_json, "vae_tile_batch", 28) != 28)
        throw std::runtime_error("MiniMax-H3 requires vae_tile_batch=28");
}

CacheConfig load_cache_config(const PipelineContext& ctx) {
    CacheConfig result;
    const std::string mode =
        extract_json_string(ctx.config_json, "denoiser_cache_mode", "monolithic");
    if (mode != "monolithic" && mode != "first_block")
        throw std::runtime_error("MiniMax-H3 bundle has an invalid denoiser_cache_mode");
    result.enabled = extract_json_bool(ctx.config_json, "first_block_cache", false);
    if (result.enabled != (mode == "first_block"))
        throw std::runtime_error("MiniMax-H3 bundle cache mode and profile flag disagree");
    result.threshold =
        extract_json_float(ctx.config_json, "first_block_cache_threshold", result.threshold);
    if (ctx.runtime_config != nullptr &&
        ctx.runtime_config->source_of("minimax_h3", "first_block_cache_threshold") !=
            config::Layer::SchemaDefault) {
        result.threshold = static_cast<float>(
            ctx.runtime_config->get<double>("minimax_h3", "first_block_cache_threshold"));
    }
    if (!std::isfinite(result.threshold) || result.threshold <= 0.0F)
        throw std::runtime_error(
            "MiniMax-H3 first_block_cache_threshold must be finite and positive");
    return result;
}

MiniMaxH3ModuleLoader make_module_loader(const PipelineContext& ctx, SectionMap sections) {
    const std::string bundle_path = ctx.bundle_path;
    const std::string runtime_cache = ctx.runtime_cache_path;
    IBackend* const backend = ctx.backend;
    const bool cuda_graphs = ctx.cuda_graphs;
    return [sections = std::move(sections), bundle_path, runtime_cache, backend,
            cuda_graphs](const std::string& name, cudaStream_t stream) {
        const auto it = sections.find(name);
        if (it == sections.end())
            throw std::runtime_error("Unknown MiniMax-H3 plan section: " + name);
        auto plan = ReadBundleSection(bundle_path, it->second);
        ModuleCreateOptions options;
        options.stream = stream;
        options.runtime_cache_path = runtime_cache.c_str();
        options.cuda_graphs = cuda_graphs;
        return backend->create_module(plan.data(), plan.size(), options);
    };
}

} // namespace

class MiniMaxH3Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        validate_profile(ctx);
        const CacheConfig cache = load_cache_config(ctx);
        auto loader = make_module_loader(ctx, index_sections(ctx.bundle.info, cache.enabled));
        return std::make_unique<MiniMaxH3Pipeline>(std::move(loader), load_tokenizer(ctx.bundle),
                                                   ctx.bundle.info.model_id, cache.enabled,
                                                   cache.threshold);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_minimax_h3_plugin, MiniMaxH3Plugin,
                                       "diffusion_minimax_h3");

} // namespace trtmc

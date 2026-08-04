/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "runtime/models/minimax_h3/pipeline.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/tokenizer.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace trtmc {
namespace {

using SectionMap = std::unordered_map<std::string, BundleSectionInfo>;

SectionMap index_sections(const BundleInfo& info) {
    constexpr std::array<const char*, 4> names = {"text_encoder_plan", "adaln_precompute_plan",
                                                  "denoiser_plan", "vae_tile_decoder_plan"};
    SectionMap sections;
    for (const char* name : names) {
        const auto it =
            std::find_if(info.sections.begin(), info.sections.end(),
                         [name](const BundleSectionInfo& item) { return item.name == name; });
        if (it == info.sections.end() || it->size == 0)
            throw std::runtime_error(std::string("MiniMax-H3 bundle is missing ") + name);
        sections.emplace(name, *it);
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

} // namespace

class MiniMaxH3Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        if (ctx.backend == nullptr)
            throw std::runtime_error("MiniMax-H3 requires the TensorRT backend");
        const int32_t cp_size = extract_json_int(ctx.config_json, "context_parallel_size", 1);
        if (cp_size != 1)
            throw std::runtime_error("MiniMax-H3 requires context_parallel_size=1");
        const int32_t sequence_rows =
            extract_json_int(ctx.config_json, "padded_sequence_length", 38247);
        if (sequence_rows != 38247)
            throw std::runtime_error("MiniMax-H3 requires 38247 unpadded sequence rows");
        const int32_t vae_tile_batch = extract_json_int(ctx.config_json, "vae_tile_batch", 28);
        if (vae_tile_batch != 28)
            throw std::runtime_error("MiniMax-H3 requires vae_tile_batch=28");
        auto sections = index_sections(ctx.bundle.info);
        const std::string bundle_path = ctx.bundle_path;
        const std::string runtime_cache = ctx.runtime_cache_path;
        IBackend* const backend = ctx.backend;
        const bool cuda_graphs = ctx.cuda_graphs;
        MiniMaxH3ModuleLoader loader = [sections = std::move(sections), bundle_path, runtime_cache,
                                        backend,
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
        return std::make_unique<MiniMaxH3Pipeline>(std::move(loader), load_tokenizer(ctx.bundle),
                                                   ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_minimax_h3_plugin, MiniMaxH3Plugin,
                                       "diffusion_minimax_h3");

} // namespace trtmc

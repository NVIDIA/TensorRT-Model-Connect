/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "pipeline.h"
#include "runtime/backend/prebound_backend.h"
#include "runtime_config.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/tokenizer.h"

#include <algorithm>
#include <array>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

using PlanSectionMap = std::unordered_map<std::string, BundleSectionInfo>;

PlanSectionMap index_plan_sections(const BundleInfo& bundle_info) {
    constexpr std::array<const char*, 4> required = {
        "text_encoder_0_plan",
        "denoiser_plan",
        "vae_decoder_plan",
        "vae_decoder_first_frame_plan",
    };
    PlanSectionMap result;
    for (const char* name : required) {
        const auto section = std::find_if(
            bundle_info.sections.begin(), bundle_info.sections.end(),
            [name](const BundleSectionInfo& candidate) { return candidate.name == name; });
        if (section == bundle_info.sections.end() || section->size == 0)
            throw std::runtime_error(std::string("Wan2.2 bundle is missing or empty: ") + name);
        result.emplace(name, *section);
    }
    return result;
}

Wan22ModuleLoader make_staged_module_loader(const PipelineContext& ctx,
                                            PlanSectionMap plan_sections) {
    if (ctx.backend == nullptr)
        throw std::runtime_error("Wan2.2 requires a TensorRT backend");

    // PipelineContext is factory-owned and expires after create(). Capture
    // every value needed by generation by value. Backends are process-cached
    // by BackendLoader, so the backend pointer remains valid for the pipeline.
    const std::string bundle_path = ctx.bundle_path;
    const std::string runtime_cache_path = ctx.runtime_cache_path;
    IBackend* const backend = ctx.backend;
    const bool cuda_graphs = ctx.cuda_graphs;
    return [bundle_path, runtime_cache_path, backend, cuda_graphs,
            plan_sections = std::move(plan_sections)](
               const std::string& section_name, cudaStream_t stream,
               const std::vector<ModuleExternalBinding>& external_bindings)
               -> std::unique_ptr<ITrtModule> {
        // Only one plan payload is resident on the host. TensorRT consumes it
        // synchronously in create_module(); this vector dies before the
        // generation stage receives the module.
        const auto section = plan_sections.find(section_name);
        if (section == plan_sections.end())
            throw std::runtime_error("Wan2.2 has no plan section named " + section_name);
        auto plan = ReadBundleSection(bundle_path, section->second);
        ModuleCreateOptions options;
        options.stream = stream;
        options.runtime_cache_path = runtime_cache_path.c_str();
        options.cuda_graphs = cuda_graphs;
        std::unique_ptr<ITrtModule> module;
        if (external_bindings.empty()) {
            module = backend->create_module(plan.data(), plan.size(), options);
        } else {
            auto* prebound_backend = dynamic_cast<IPreboundBackend*>(backend);
            if (prebound_backend == nullptr) {
                throw std::runtime_error(
                    "Wan2.2 requires a TensorRT backend with external I/O prebinding");
            }
            module = prebound_backend->create_module_prebound(plan.data(), plan.size(), options,
                                                              external_bindings);
        }
        return module;
    };
}

std::unique_ptr<ITokenizer> load_tokenizer(const BundleFile& bundle) {
    const auto* tokenizer_json = find_section(bundle, "tokenizer.json");
    if (tokenizer_json == nullptr || tokenizer_json->empty())
        throw std::runtime_error("Wan2.2 bundle is missing tokenizer.json");
    auto tokenizer = CreateUnigramTokenizer(tokenizer_json->data(), tokenizer_json->size(), false);
    if (!tokenizer)
        throw std::runtime_error("Wan2.2 could not create the native UMT5 tokenizer");
    return tokenizer;
}

} // namespace

class Wan22TI2VPlugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        auto plan_sections = index_plan_sections(ctx.bundle.info);
        auto tokenizer = load_tokenizer(ctx.bundle);
        return std::make_unique<Wan22TI2VPipeline>(
            make_staged_module_loader(ctx, std::move(plan_sections)), std::move(tokenizer),
            parse_wan22_options(ctx.config_json),
            wan2_2_ti2v::resolve_runtime_config(ctx.runtime_config), ctx.bundle.info.model_id);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_wan2_2_ti2v_plugin, Wan22TI2VPlugin,
                                       "diffusion_wan2_2_ti2v");

} // namespace trtmc

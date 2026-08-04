/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "runtime/backend/prebound_backend.h"
#include "runtime/models/cosmos3/pipeline.h"
#include "trtmc/runtime/distributed_runtime.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/tokenizer.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc {
namespace {

struct ContextParallelRuntimeConfig {
    bool enabled{false};
    int32_t size{1};
    int32_t denoiser_size{1};
    int32_t classifier_free_size{1};
};

ContextParallelRuntimeConfig parse_context_parallel_config(const std::string& config_json) {
    ContextParallelRuntimeConfig config;
    auto mode = extract_json_string(config_json, "parallel_mode", "single");
    if (mode == "single")
        mode = extract_json_string(config_json, "context_parallel_mode", "single");
    config.size = extract_json_int(config_json, "context_parallel_size", 1);
    config.enabled = mode == "context_parallel" && config.size > 1;
    if (mode != "single" && mode != "context_parallel")
        throw std::invalid_argument("Cosmos3 supports single-device or context parallel runtime");
    if (config.size != 1 && config.size != 2 && config.size != 4 && config.size != 8)
        throw std::invalid_argument("Cosmos3 context_parallel_size must be 1, 2, 4, or 8");
    config.denoiser_size =
        extract_json_int(config_json, "denoiser_context_parallel_size", config.size);
    config.classifier_free_size = extract_json_int(config_json, "classifier_free_parallel_size", 1);
    if (config.denoiser_size < 1 || config.classifier_free_size < 1 ||
        config.denoiser_size * config.classifier_free_size != config.size) {
        throw std::invalid_argument(
            "Cosmos3 denoiser and classifier-free parallel sizes must multiply to "
            "context_parallel_size");
    }
    if (config.classifier_free_size != 1 && config.classifier_free_size != 2)
        throw std::invalid_argument("Cosmos3 classifier_free_parallel_size must be 1 or 2");
    return config;
}

using PlanSectionMap = std::unordered_map<std::string, BundleSectionInfo>;

const BundleSectionInfo& require_section(const BundleInfo& info, const std::string& name) {
    const auto section = std::find_if(
        info.sections.begin(), info.sections.end(),
        [&name](const BundleSectionInfo& candidate) { return candidate.name == name; });
    if (section == info.sections.end() || section->size == 0)
        throw std::runtime_error("Cosmos3 bundle is missing or empty: " + name);
    return *section;
}

PlanSectionMap index_plan_sections(const BundleInfo& bundle_info, bool context_parallel) {
    PlanSectionMap sections;
    const std::string denoiser_name = context_parallel ? "denoiser_plan_cp" : "denoiser_plan";
    sections.emplace("denoiser_plan", require_section(bundle_info, denoiser_name));
    for (const char* name : {"vae_decoder_plan", "vae_decoder_first_frame_plan"})
        sections.emplace(name, require_section(bundle_info, name));
    return sections;
}

Cosmos3ModuleLoader make_staged_loader(const PipelineContext& ctx, PlanSectionMap plan_sections,
                                       DistributedRuntimeGroup distributed_group,
                                       bool context_parallel) {
    if (ctx.backend == nullptr)
        throw std::runtime_error("Cosmos3 requires a TensorRT backend");
    const std::string bundle_path = ctx.bundle_path;
    const std::string runtime_cache_path = ctx.runtime_cache_path;
    IBackend* const backend = ctx.backend;
    const bool cuda_graphs = ctx.cuda_graphs;
    return [bundle_path, runtime_cache_path, backend, cuda_graphs,
            distributed_group = std::move(distributed_group), context_parallel,
            plan_sections = std::move(plan_sections)](
               const std::string& logical_name, cudaStream_t stream,
               const std::vector<ModuleExternalBinding>& external_bindings)
               -> std::unique_ptr<ITrtModule> {
        const auto section = plan_sections.find(logical_name);
        if (section == plan_sections.end())
            throw std::runtime_error("Cosmos3 has no plan section named " + logical_name);
        auto plan = ReadBundleSection(bundle_path, section->second);
        ModuleCreateOptions options;
        options.stream = stream;
        options.runtime_cache_path = runtime_cache_path.c_str();
        options.cuda_graphs = cuda_graphs;
        if (context_parallel && logical_name == "denoiser_plan") {
            options.distributed_communicator = distributed_group.communicator;
            options.distributed_owner = distributed_group.owner;
        }

        if (external_bindings.empty())
            return backend->create_module(plan.data(), plan.size(), options);
        auto* prebound_backend = dynamic_cast<IPreboundBackend*>(backend);
        if (prebound_backend == nullptr) {
            throw std::runtime_error(
                "Cosmos3 requires a TensorRT backend with external I/O prebinding");
        }
        return prebound_backend->create_module_prebound(plan.data(), plan.size(), options,
                                                        external_bindings);
    };
}

std::unique_ptr<ITokenizer> load_tokenizer(const BundleFile& bundle) {
    const auto* tokenizer_json = find_section(bundle, "tokenizer.json");
    const auto* tokenizer_config = find_section(bundle, "tokenizer_config.json");
    if (tokenizer_json == nullptr || tokenizer_json->empty() || tokenizer_config == nullptr ||
        tokenizer_config->empty()) {
        throw std::runtime_error(
            "Cosmos3 bundle requires tokenizer.json and tokenizer_config.json");
    }
    auto tokenizer = CreateBpeTokenizer(tokenizer_json->data(), tokenizer_json->size(), false);
    if (!tokenizer)
        throw std::runtime_error("Cosmos3 could not create its native BPE tokenizer");
    constexpr std::array<std::pair<const char*, int32_t>, 3> expected = {{
        {"<|im_start|>", 151644},
        {"<|im_end|>", 151645},
        {"<|vision_start|>", 151652},
    }};
    for (const auto& [token, id] : expected) {
        if (tokenizer->id_for_token(token) != id)
            throw std::runtime_error(std::string("Cosmos3 tokenizer has an invalid ID for ") +
                                     token);
    }
    return tokenizer;
}

} // namespace

class Cosmos3Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        const auto cp_config = parse_context_parallel_config(ctx.config_json);
        DistributedRuntimeGroup context_group;
        DistributedRuntimeGroup classifier_free_group;
        if (cp_config.enabled && cp_config.classifier_free_size == 1) {
            context_group = initialize_tensor_parallel_group(cp_config.size);
        } else if (cp_config.enabled) {
            const int32_t global_world_size = distributed_process_world_size();
            const int32_t global_rank = distributed_process_rank();
            if (global_world_size != cp_config.size || global_rank < 0 ||
                global_rank >= global_world_size) {
                throw std::runtime_error(
                    "Cosmos3 classifier-free parallel runtime requires mpirun world size to "
                    "equal context_parallel_size");
            }
            const int32_t context_group_index = global_rank / cp_config.denoiser_size;
            const int32_t context_rank = global_rank % cp_config.denoiser_size;
            context_group = initialize_distributed_subgroup(
                cp_config.denoiser_size, context_rank,
                "cosmos3_cp_" + std::to_string(context_group_index));

            const int32_t classifier_free_rank = context_group_index;
            const int32_t classifier_free_pair = context_rank;
            classifier_free_group = initialize_distributed_subgroup(
                cp_config.classifier_free_size, classifier_free_rank,
                "cosmos3_cfg_" + std::to_string(classifier_free_pair));
        }
        auto tokenizer = load_tokenizer(ctx.bundle);
        auto plans = index_plan_sections(ctx.bundle.info, cp_config.enabled);
        auto loader = make_staged_loader(ctx, std::move(plans), context_group, cp_config.enabled);
        return std::make_unique<Cosmos3Pipeline>(
            std::move(loader), std::move(tokenizer), parse_cosmos3_options(ctx.config_json),
            ctx.bundle.info.model_id, context_group, classifier_free_group);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_cosmos3_plugin, Cosmos3Plugin, "diffusion_cosmos3");

} // namespace trtmc

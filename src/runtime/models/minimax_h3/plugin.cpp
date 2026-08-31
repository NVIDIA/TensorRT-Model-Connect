/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "bundle/bundle_view.h"
#include "runtime/backend/prebound_backend.h"
#include "runtime/models/minimax_h3/pipeline.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"
#include "trtmc/tokenizer.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace trtmc {
namespace {

using SectionMap = std::unordered_map<std::string, BundleSectionInfo>;
using PlanSha256Map = std::unordered_map<std::string, std::string>;

const char* plan_filename(std::string_view section) {
    if (section == "text_encoder_plan")
        return "text_encoder.plan";
    if (section == "adaln_precompute_plan")
        return "adaln_precompute.plan";
    if (section == "denoiser_plan")
        return "denoiser.plan";
    if (section == "denoiser_head_plan")
        return "denoiser_head.plan";
    if (section == "denoiser_tail_plan")
        return "denoiser_tail.plan";
    if (section == "denoiser_finish_plan")
        return "denoiser_finish.plan";
    if (section == "vae_tile_decoder_plan")
        return "vae_tile_decoder.plan";
    throw std::runtime_error("Unknown MiniMax-H3 plan section: " + std::string(section));
}

bool is_sha256(std::string_view value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

struct RuntimeMemoryConfig {
    bool staged{false};
    std::int64_t weight_streaming_budget_bytes{-1};
};

void validate_rtx_runtime_context(const PipelineContext& ctx, std::string_view recorded_backend) {
    if (ctx.backend == nullptr || std::string_view(ctx.backend->name()) != recorded_backend)
        throw std::runtime_error(
            "MiniMax-H3 bundle backend does not match the loaded runtime backend");
    if (ctx.cuda_graphs)
        throw std::runtime_error(
            "MiniMax-H3 TensorRT-RTX weight streaming does not support CUDA graphs");
}

const nlohmann::json& require_runtime_memory_object(const nlohmann::json& root) {
    if (!root.is_object() || root.value("engine_backend", std::string{}) != "trt_rtx" ||
        !root.contains("runtime_memory") || !root.at("runtime_memory").is_object()) {
        throw std::runtime_error(
            "MiniMax-H3 TensorRT-RTX bundle is missing staged runtime metadata");
    }
    return root.at("runtime_memory");
}

std::int64_t parse_weight_streaming_budget(const nlohmann::json& memory) {
    if (memory.value("mode", std::string{}) != "staged" ||
        !memory.contains("weight_streaming_budget_bytes") ||
        !memory.at("weight_streaming_budget_bytes").is_number_integer()) {
        throw std::runtime_error(
            "MiniMax-H3 TensorRT-RTX bundle has invalid staged runtime metadata");
    }
    const auto budget = memory.at("weight_streaming_budget_bytes").get<std::int64_t>();
    if (budget < 0)
        throw std::runtime_error(
            "MiniMax-H3 TensorRT-RTX weight-streaming budget must be nonnegative");
    return budget;
}

RuntimeMemoryConfig load_runtime_memory_config(const PipelineContext& ctx) {
    const std::string recorded_backend =
        extract_json_string(ctx.config_json, "engine_backend", "trt");
    if (recorded_backend != "trt_rtx")
        return {};
    validate_rtx_runtime_context(ctx, recorded_backend);
    try {
        const auto root = nlohmann::json::parse(ctx.config_json);
        return RuntimeMemoryConfig{
            true, parse_weight_streaming_budget(require_runtime_memory_object(root))};
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("MiniMax-H3 invalid runtime-memory JSON: ") +
                                 error.what());
    }
}

PlanSha256Map load_plan_sha256(const std::string& config_json, const SectionMap& sections) {
    try {
        const auto root = nlohmann::json::parse(config_json);
        if (!root.is_object() || !root.contains("plan_sha256") ||
            !root.at("plan_sha256").is_object()) {
            throw std::runtime_error("MiniMax-H3 bundle is missing plan SHA-256 records");
        }
        const auto& records = root.at("plan_sha256");
        PlanSha256Map result;
        for (const auto& [section, _] : sections) {
            const char* filename = plan_filename(section);
            if (!records.contains(filename) || !records.at(filename).is_string())
                throw std::runtime_error("MiniMax-H3 bundle is missing plan SHA-256 for " +
                                         section);
            const std::string digest = records.at(filename).get<std::string>();
            if (!is_sha256(digest))
                throw std::runtime_error("MiniMax-H3 bundle has invalid plan SHA-256 for " +
                                         section);
            result.emplace(section, digest);
        }
        return result;
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("MiniMax-H3 invalid plan SHA-256 JSON: ") +
                                 error.what());
    }
}

struct CacheConfig {
    bool enabled{false};
    float threshold{0.025F};
};

struct HotEngineConfig {
    bool retain_engines{false};
    std::int64_t tail_weight_budget_bytes{24LL << 30};
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

HotEngineConfig load_hot_engine_config(const PipelineContext& ctx) {
    HotEngineConfig result;
    if (ctx.runtime_config == nullptr)
        return result;
    result.retain_engines =
        ctx.runtime_config->get<bool>("minimax_h3", "retain_engines");
    const auto budget_gib = ctx.runtime_config->get<std::int64_t>(
        "minimax_h3", "retained_tail_weight_budget_gib");
    if (budget_gib <= 0 ||
        budget_gib > (std::numeric_limits<std::int64_t>::max() >> 30)) {
        throw std::runtime_error(
            "MiniMax-H3 retained_tail_weight_budget_gib must be positive");
    }
    result.tail_weight_budget_bytes = budget_gib << 30;
    return result;
}

const BundleSectionInfo& require_plan_section(const SectionMap& sections, const std::string& name) {
    const auto it = sections.find(name);
    if (it == sections.end())
        throw std::runtime_error("Unknown MiniMax-H3 plan section: " + name);
    return it->second;
}

bool retain_hot_engine(std::string_view name, const HotEngineConfig& hot) {
    if (!hot.retain_engines)
        return false;
    return name == "denoiser_head_plan" || name == "denoiser_tail_plan" ||
           name == "denoiser_finish_plan" || name == "vae_tile_decoder_plan";
}

ModuleCreateOptions module_options(cudaStream_t stream, const std::string& runtime_cache,
                                   bool cuda_graphs) {
    ModuleCreateOptions options;
    options.stream = stream;
    options.runtime_cache_path = runtime_cache.c_str();
    options.cuda_graphs = cuda_graphs;
    return options;
}

std::int64_t staged_plan_budget(const std::string& name, const RuntimeMemoryConfig& memory,
                                const HotEngineConfig& hot) {
    if (retain_hot_engine(name, hot)) {
        if (name == "denoiser_head_plan" || name == "denoiser_finish_plan" ||
            name == "vae_tile_decoder_plan")
            return std::numeric_limits<std::int64_t>::max();
        if (name == "denoiser_tail_plan")
            return std::min<std::int64_t>(memory.weight_streaming_budget_bytes,
                                          hot.tail_weight_budget_bytes);
    }
    return (name == "denoiser_head_plan" || name == "denoiser_finish_plan")
               ? 0
               : memory.weight_streaming_budget_bytes;
}

std::unique_ptr<ITrtModule>
load_staged_module(const std::string& name, const BundleSectionInfo& section,
                   const std::string& bundle_path, const PlanSha256Map& plan_sha256,
                   IFileBackedBackend* file_backed_backend, const ModuleCreateOptions& options,
                   const std::vector<ModuleExternalBinding>& external_bindings,
                   const RuntimeMemoryConfig& memory, const HotEngineConfig& hot) {
    if (file_backed_backend == nullptr)
        throw std::runtime_error("MiniMax-H3 TensorRT-RTX backend lacks file-backed plan support");
    const auto digest = plan_sha256.find(name);
    if (digest == plan_sha256.end())
        throw std::runtime_error("MiniMax-H3 plan SHA-256 is missing: " + name);
    const auto range = ResolveBundleSectionFileRange(bundle_path, section);
    auto module = file_backed_backend->create_module_from_file(
        bundle_path.c_str(), range.offset, range.size, digest->second.c_str(), options,
        external_bindings, staged_plan_budget(name, memory, hot), retain_hot_engine(name, hot));
    if (!module)
        throw std::runtime_error("MiniMax-H3 backend rejected file-backed plan deserialization");
    return module;
}

std::unique_ptr<ITrtModule>
load_in_memory_module(const BundleSectionInfo& section, const std::string& bundle_path,
                      IBackend* backend, IPreboundBackend* prebound_backend,
                      const ModuleCreateOptions& options,
                      const std::vector<ModuleExternalBinding>& external_bindings) {
    auto plan = ReadBundleSection(bundle_path, section);
    if (external_bindings.empty())
        return backend->create_module(plan.data(), plan.size(), options);
    if (prebound_backend == nullptr)
        throw std::runtime_error("MiniMax-H3 backend lacks external I/O prebinding support");
    return prebound_backend->create_module_prebound(plan.data(), plan.size(), options,
                                                    external_bindings);
}

std::unique_ptr<ITrtModule> load_module(const std::string& name, cudaStream_t stream,
                                        const std::vector<ModuleExternalBinding>& external_bindings,
                                        const SectionMap& sections, const std::string& bundle_path,
                                        const std::string& runtime_cache, IBackend* backend,
                                        bool cuda_graphs, const RuntimeMemoryConfig& memory,
                                        const PlanSha256Map& plan_sha256,
                                        const HotEngineConfig& hot) {
    const auto& section = require_plan_section(sections, name);
    const auto options = module_options(stream, runtime_cache, cuda_graphs);
    auto* prebound_backend = dynamic_cast<IPreboundBackend*>(backend);
    if (memory.staged) {
        auto* file_backed_backend = dynamic_cast<IFileBackedBackend*>(backend);
        return load_staged_module(name, section, bundle_path, plan_sha256, file_backed_backend,
                                  options, external_bindings, memory, hot);
    }
    return load_in_memory_module(section, bundle_path, backend, prebound_backend, options,
                                 external_bindings);
}

MiniMaxH3ModuleLoader make_module_loader(const PipelineContext& ctx, SectionMap sections,
                                         RuntimeMemoryConfig memory, HotEngineConfig hot) {
    if (hot.retain_engines && !memory.staged) {
        throw std::runtime_error(
            "MiniMax-H3 retained engines require a staged TensorRT-RTX bundle");
    }
    const std::string bundle_path = ctx.bundle_path;
    const std::string runtime_cache = ctx.runtime_cache_path;
    IBackend* const backend = ctx.backend;
    const bool cuda_graphs = ctx.cuda_graphs;
    PlanSha256Map plan_sha256;
    if (memory.staged)
        plan_sha256 = load_plan_sha256(ctx.config_json, sections);
    return [sections = std::move(sections), bundle_path, runtime_cache, backend, cuda_graphs,
            memory, hot, plan_sha256 = std::move(plan_sha256)](
               const std::string& name, cudaStream_t stream,
               const std::vector<ModuleExternalBinding>& external_bindings) {
        return load_module(name, stream, external_bindings, sections, bundle_path, runtime_cache,
                           backend, cuda_graphs, memory, plan_sha256, hot);
    };
}

} // namespace

class MiniMaxH3Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        validate_profile(ctx);
        const CacheConfig cache = load_cache_config(ctx);
        auto sections = index_sections(ctx.bundle.info, cache.enabled);
        auto loader = make_module_loader(ctx, std::move(sections), load_runtime_memory_config(ctx),
                                         load_hot_engine_config(ctx));
        return std::make_unique<MiniMaxH3Pipeline>(std::move(loader), load_tokenizer(ctx.bundle),
                                                   ctx.bundle.info.model_id, cache.enabled,
                                                   cache.threshold);
    }
};

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_minimax_h3_plugin, MiniMaxH3Plugin,
                                       "diffusion_minimax_h3");

} // namespace trtmc

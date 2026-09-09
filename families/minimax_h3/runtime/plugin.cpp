/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/hot_engine_policy.h"
#include "families/minimax_h3/runtime/pipeline.h"
#include "families/minimax_h3/runtime/tokenizer.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc::minimax_h3_factory {
namespace {

using PlanMap = std::unordered_map<std::string, BundleSectionInfo>;

struct RuntimeMemoryConfig {
    bool staged{false};
    std::int64_t weight_streaming_budget_bytes{-1};
};

struct HotEngineConfig {
    // The validated public 5-second path is the default. runtime.json may
    // lower these values for a smaller-memory deployment.
    bool retain_engines{true};
    std::int64_t tail_weight_budget_bytes{24LL << 30};
};

std::vector<char> require_section(const BundleReader& bundle, const std::string& name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("MiniMax-H3 bundle section is missing or empty: " + name);
    return bundle.read_section(name);
}

std::unique_ptr<ITokenizer> load_tokenizer(const BundleReader& bundle) {
    const auto data = require_section(bundle, "tokenizer.json");
    auto tokenizer = CreateBpeTokenizer(data.data(), data.size(), false);
    if (!tokenizer)
        throw std::runtime_error("MiniMax-H3 tokenizer.json is not the required BPE tokenizer");
    return tokenizer;
}

bool declares_workflow(const nlohmann::json& config, const char* workflow) {
    const auto found = config.find("public_workflows");
    if (found == config.end() || !found->is_array())
        return std::string(workflow) == "t2va";
    return std::any_of(found->begin(), found->end(), [workflow](const auto& value) {
        return value.is_string() && value.template get<std::string>() == workflow;
    });
}

void require_declared_plan(const BundleReader& bundle, PlanMap& plans, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("MiniMax-H3 bundle is missing required plan: " +
                                 std::string(name));
    plans.emplace(name, *section);
}

PlanMap index_plans(const BundleReader& bundle, const nlohmann::json& config,
                    const minimax_h3::SuperResolutionConfig& sr) {
    PlanMap plans;
    for (const char* name :
         {"text_encoder_plan", "adaln_precompute_plan", "denoiser_head_plan", "denoiser_tail_plan",
          "denoiser_finish_plan", "vae_tile_decoder_plan", "audio_vae_decoder_plan"}) {
        require_declared_plan(bundle, plans, name);
    }
    if (declares_workflow(config, "fl2va")) {
        require_declared_plan(bundle, plans, "vision_encoder_plan");
        require_declared_plan(bundle, plans, "fl2va_keyframe_vae_encoder_plan");
    }
    if (declares_workflow(config, "ref2va")) {
        for (const char* name : {"ref2va_adaln_precompute_plan", "ref2va_video_vae_encoder_plan",
                                 "ref2va_audio_vae_encoder_plan"}) {
            require_declared_plan(bundle, plans, name);
        }
        const auto cache = config.value("ref2va_first_block_cache", nlohmann::json::object());
        if (cache.value("enabled", false)) {
            for (const char* name :
                 {"ref2va_dit_head_plan", "ref2va_dit_tail_plan", "ref2va_dit_finish_plan"})
                require_declared_plan(bundle, plans, name);
        } else {
            require_declared_plan(bundle, plans, "ref2va_denoiser_plan");
        }
    }
    if (sr.enabled)
        require_declared_plan(bundle, plans, sr.section.c_str());
    return plans;
}

RuntimeMemoryConfig load_runtime_memory_config(const nlohmann::json& config,
                                               const BundleReader& bundle, bool cuda_graphs) {
    if (bundle.info().backend != "trt_rtx")
        return {};
    const auto found = config.find("runtime_memory");
    if (found == config.end() || !found->is_object() ||
        found->value("mode", std::string{}) != "staged" ||
        !found->contains("weight_streaming_budget_bytes") ||
        !found->at("weight_streaming_budget_bytes").is_number_integer()) {
        throw std::runtime_error(
            "MiniMax-H3 TensorRT-RTX bundle is missing staged runtime metadata");
    }
    const auto budget = found->at("weight_streaming_budget_bytes").get<std::int64_t>();
    if (budget < 0)
        throw std::runtime_error("MiniMax-H3 weight-streaming budget must be nonnegative");
    if (cuda_graphs)
        throw std::runtime_error(
            "MiniMax-H3 TensorRT-RTX weight streaming does not support CUDA graphs");
    return {true, budget};
}

HotEngineConfig load_hot_engine_config(const nlohmann::json& config) {
    HotEngineConfig result;
    const auto memory = config.find("runtime_memory");
    if (memory == config.end() || !memory->is_object())
        return result;
    result.retain_engines = memory->value("retain_engines", result.retain_engines);
    if (memory->contains("retained_tail_weight_budget_bytes")) {
        if (!memory->at("retained_tail_weight_budget_bytes").is_number_integer())
            throw std::runtime_error("MiniMax-H3 retained tail budget must be an integer");
        result.tail_weight_budget_bytes =
            memory->at("retained_tail_weight_budget_bytes").get<std::int64_t>();
    }
    if (result.tail_weight_budget_bytes <= 0)
        throw std::runtime_error("MiniMax-H3 retained tail budget must be positive");
    return result;
}

class RuntimeCacheLease final {
  public:
    RuntimeCacheLease(IBackend& backend, const std::string& path)
        : backend_(&backend), lease_(backend.acquire_runtime_cache_lease(path.c_str())) {}

    ~RuntimeCacheLease() {
        if (lease_ == 0)
            return;
        try {
            finalize();
        } catch (const std::exception& error) {
            std::cerr << "[trtmc] Failed to persist RTX runtime cache: " << error.what() << '\n';
        }
    }

    void require_active() const {
        if (lease_ == 0)
            throw std::runtime_error("MiniMax-H3 runtime cache lease is finalized");
    }

    void finalize() {
        if (lease_ == 0)
            return;
        backend_->release_runtime_cache_lease(lease_);
        lease_ = 0;
    }

  private:
    IBackend* backend_;
    std::uint64_t lease_{0};
};

std::shared_ptr<RuntimeCacheLease> make_runtime_cache_lease(IBackend& backend,
                                                            const std::string& path, bool staged) {
    if (path.empty())
        return {};
    if (!staged)
        throw std::runtime_error("MiniMax-H3 runtime cache requires a staged TensorRT-RTX bundle");
    return std::make_shared<RuntimeCacheLease>(backend, path);
}

MiniMaxH3ModuleLoader make_loader(IBackend& backend, const BundleReader& source, PlanMap plans,
                                  RuntimeMemoryConfig memory, HotEngineConfig hot,
                                  std::string runtime_cache_path, bool cuda_graphs,
                                  const std::string& super_resolution_section,
                                  std::shared_ptr<RuntimeCacheLease> cache_lease) {
    BundleReader bundle(source.path());
    return [&backend, bundle = std::move(bundle), plans = std::move(plans), memory, hot,
            runtime_cache_path = std::move(runtime_cache_path), cuda_graphs,
            super_resolution_section, cache_lease = std::move(cache_lease)](
               const std::string& name, cudaStream_t stream,
               const std::vector<ModuleExternalBinding>& bindings, std::int32_t profile) {
        const auto found = plans.find(name);
        if (found == plans.end())
            throw std::runtime_error("MiniMax-H3 requested undeclared plan: " + name);
        if (cache_lease)
            cache_lease->require_active();

        ModuleCreateOptions options{};
        options.stream = stream;
        options.runtime_cache_path = runtime_cache_path.c_str();
        options.cuda_graphs = cuda_graphs;
        options.optimization_profile = profile;

        std::unique_ptr<ITrtModule> module;
        if (memory.staged && name != super_resolution_section) {
            const auto budget = minimax_h3::staged_plan_weight_streaming_budget(
                name, memory.weight_streaming_budget_bytes, hot.retain_engines,
                hot.tail_weight_budget_bytes);
            module = backend.create_module_from_file(
                bundle.path().c_str(), bundle.section_file_offset(name), found->second.length,
                options, bindings, budget,
                minimax_h3::should_retain_hot_engine(name, hot.retain_engines),
                minimax_h3::uses_serial_execution_context(name));
        } else {
            auto plan = require_section(bundle, name);
            module = bindings.empty() ? backend.create_module(plan.data(), plan.size(), options)
                                      : backend.create_module_prebound(plan.data(), plan.size(),
                                                                       options, bindings);
        }
        if (!module || !module->ok())
            throw std::runtime_error("MiniMax-H3 failed to load plan: " + name);
        return module;
    };
}

std::array<float, 32> read_audio_array(const nlohmann::json& config, const char* name,
                                       bool positive) {
    const auto& values = config.at(name);
    if (!values.is_array() || values.size() != 32)
        throw std::runtime_error(std::string("MiniMax-H3 runtime.json has invalid ") + name);
    std::array<float, 32> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = values.at(index).get<float>();
        if (!std::isfinite(result[index]) || (positive && result[index] <= 0.0F))
            throw std::runtime_error(std::string("MiniMax-H3 runtime.json has invalid ") + name);
    }
    return result;
}

MiniMaxH3Ref2VAConfig load_ref2va_config(const nlohmann::json& config) {
    MiniMaxH3Ref2VAConfig result;
    if (!declares_workflow(config, "ref2va"))
        return result;
    result.enabled = true;
    const auto& scheduler = config.at("ref2va_scheduler");
    result.scheduler_grid_points = scheduler.at("sigma_grid_points").get<std::int32_t>();
    result.transformer_forwards = scheduler.at("transformer_forwards").get<std::int32_t>();
    result.video_shift = scheduler.at("video_shift").get<float>();
    result.audio_shift = scheduler.at("audio_shift").get<float>();
    result.guidance_scale = scheduler.at("guidance_scale").get<float>();
    result.guidance_distilled = scheduler.at("guidance_distilled").get<bool>();
    result.denoiser_profile_count = config.value("ref2va_denoiser_profile_count", 1);
    const auto cache = config.value("ref2va_first_block_cache", nlohmann::json::object());
    result.first_block_cache = cache.value("enabled", false);
    result.first_block_cache_threshold = cache.value("threshold", 0.08F);
    result.audio_latent_mean = read_audio_array(config, "audio_latents_mean", false);
    result.audio_latent_std = read_audio_array(config, "audio_latents_std", true);
    return result;
}

minimax_h3::SuperResolutionConfig load_super_resolution_config(const nlohmann::json& config) {
    const auto found = config.find("super_resolution");
    if (found == config.end())
        return {};
    if (!found->is_object())
        throw std::runtime_error("MiniMax-H3 super_resolution metadata must be an object");
    minimax_h3::SuperResolutionConfig result;
    result.enabled = true;
    result.section = found->value("section", std::string{});
    result.input_name = found->value("input_name", std::string{});
    result.output_name = found->value("output_name", std::string{});
    result.source_height = 480;
    result.source_width = 864;
    result.target_height = 720;
    result.target_width = 1296;
    result.batch_min = 1;
    result.batch_opt = 4;
    result.batch_max = 8;
    minimax_h3::validate_super_resolution_config(result);
    return result;
}

MiniMaxH3DenoiserConfig load_denoiser_config(const nlohmann::json& config) {
    MiniMaxH3DenoiserConfig result;
    result.scheduler_grid_points = config.value("scheduler_grid_points", 50);
    result.transformer_forwards = config.value("transformer_forwards", 49);
    result.guidance_scale = config.value("guidance_scale", 1.0F);
    result.max_text_rows = config.value("text_rows_max", 2641);
    result.optimization_profile_count = config.value("denoiser_profile_count", 1);
    return result;
}

} // namespace
} // namespace trtmc::minimax_h3_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("minimax_h3 does not support --kv-cache-size");
    using namespace trtmc;
    using namespace trtmc::minimax_h3_factory;
    try {
        const auto runtime = require_section(context.reader, "runtime.json");
        const auto config = nlohmann::json::parse(runtime.begin(), runtime.end());
        if (!config.is_object() || config.value("context_parallel_size", 1) != 1 ||
            !config.value("first_block_cache", true) ||
            config.value("denoiser_cache_mode", std::string("first_block")) != "first_block") {
            throw std::runtime_error("MiniMax-H3 runtime.json declares an unsupported profile");
        }
        const float threshold = config.value("first_block_cache_threshold", 0.08F);
        if (!std::isfinite(threshold) || threshold <= 0.0F)
            throw std::runtime_error("MiniMax-H3 cache threshold must be finite and positive");
        const auto sr = load_super_resolution_config(config);
        auto plans = index_plans(context.reader, config, sr);
        const auto memory = load_runtime_memory_config(config, context.reader, context.cuda_graphs);
        auto cache_lease =
            make_runtime_cache_lease(context.backend, context.runtime_cache_path, memory.staged);
        auto loader = make_loader(context.backend, context.reader, std::move(plans), memory,
                                  load_hot_engine_config(config), context.runtime_cache_path,
                                  context.cuda_graphs, sr.section, cache_lease);
        std::function<void()> cache_finalizer;
        if (cache_lease) {
            cache_finalizer = [cache_lease = std::move(cache_lease)] { cache_lease->finalize(); };
        }
        return new MiniMaxH3Pipeline(std::move(loader), load_tokenizer(context.reader),
                                     "minimax_h3", threshold, load_denoiser_config(config),
                                     load_ref2va_config(config), sr, std::move(cache_finalizer));
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("Invalid MiniMax-H3 runtime.json: ") + error.what());
    }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/pipeline_factory.h"

#include "bundle/bundle_format.h"
#include "runtime/backend/backend_loader.h"
#include "runtime/backend/trt_version.h"
#include "runtime/core/trt_common.h"
#include "runtime/domains/text/dynamic_memory/runtime_memory_qualification.h"
#include "runtime/providers/optimized_runtime_host.h"
#include "runtime/registry/bundle_materialization.h"
#include "trtmc/config/cli_support.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/config/schema_registry.h"
#include "trtmc/runtime/pipeline_plugin.h"
#include "trtmc/runtime/pipeline_plugin_loader.h"
#include "trtmc/runtime/pipeline_pool.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"
#include "utils/data_dir.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc {

namespace detail {

namespace {

const BundleSectionInfo* find_nonempty_config_section(const BundleInfo& info) {
    const auto config_info =
        std::find_if(info.sections.begin(), info.sections.end(),
                     [](const BundleSectionInfo& entry) { return entry.name == "config.json"; });
    if (config_info == info.sections.end() || config_info->size == 0)
        return nullptr;
    return &*config_info;
}

bool contains(const std::vector<std::string>& names, const std::string& name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}

void validate_staged_loading_policy(const BundleInfo& info, const std::string& mode,
                                    const std::vector<std::string>& eager,
                                    const std::vector<std::string>& lazy) {
    if (mode != "staged" || eager.empty() || lazy.empty() || !contains(eager, "config.json") ||
        eager.size() + lazy.size() != info.sections.size()) {
        throw std::runtime_error("Invalid staged bundle_loading policy");
    }

    std::unordered_set<std::string> header_names;
    for (const auto& section : info.sections) {
        if (!header_names.insert(section.name).second ||
            std::count(eager.begin(), eager.end(), section.name) +
                    std::count(lazy.begin(), lazy.end(), section.name) !=
                1) {
            throw std::runtime_error(
                "Staged bundle_loading must partition bundle sections exactly");
        }
    }
}

} // namespace

PipelineBundleMaterialization materialize_pipeline_bundle(const std::string& bundle_path) {
    const auto info = ReadBundleHeader(bundle_path);
    const auto* config_info = find_nonempty_config_section(info);

    // Preserve compatibility with legacy bundles that did not carry a
    // config section (or carried an empty one). PipelineFactory will retain
    // its existing manifest-default strategy resolution for those bundles.
    if (config_info == nullptr) {
        return PipelineBundleMaterialization{ReadBundleFile(bundle_path), {}};
    }

    auto config_data = ReadBundleSection(bundle_path, *config_info);
    std::string config_text(config_data.begin(), config_data.end());
    const std::string policy_text = extract_json_object_text(config_text, "bundle_loading");
    if (policy_text.empty()) {
        return PipelineBundleMaterialization{ReadBundleFile(bundle_path), std::move(config_text)};
    }

    const std::string mode = extract_json_string(policy_text, "mode", "");
    const auto eager = extract_json_string_array(policy_text, "eager_sections");
    const auto lazy = extract_json_string_array(policy_text, "lazy_sections");
    validate_staged_loading_policy(info, mode, eager, lazy);

    BundleFile bundle;
    bundle.info = info;
    bundle.sections.reserve(eager.size());
    for (const auto& section_info : info.sections) {
        if (!contains(eager, section_info.name))
            continue;
        auto data = section_info.name == "config.json"
                        ? std::move(config_data)
                        : ReadBundleSection(bundle_path, section_info);
        bundle.sections.push_back(BundleSection{section_info.name, std::move(data)});
    }
    return PipelineBundleMaterialization{std::move(bundle), std::move(config_text)};
}

} // namespace detail

namespace {

std::string normalize_legacy_strategy(const std::string& strategy, const std::string& config_text) {
    auto alias = legacy_runtime_strategy_alias_target(strategy, config_text);
    return alias.value_or(strategy);
}

std::string resolve_runtime_strategy(const std::string& config_text) {
    std::string strategy = extract_json_string(config_text, "runtime_strategy", "");
    if (strategy.empty()) {
        auto fallback = default_runtime_strategy();
        if (!fallback || fallback->empty()) {
            throw std::runtime_error(
                "Bundle config missing runtime_strategy and no runtime model manifest declares "
                "default_runtime_strategy");
        }
        strategy = *fallback;
    }
    return normalize_legacy_strategy(strategy, config_text);
}

void validate_load_options_v2(const LoadOptionsV2& options) {
    if (options.struct_size < sizeof(LoadOptionsV2)) {
        throw std::invalid_argument("LoadOptionsV2.struct_size is too small for API version 2");
    }
    if (options.api_version != kLoadOptionsV2ApiVersion) {
        throw std::invalid_argument("LoadOptionsV2.api_version is not supported");
    }
    if (options.max_sequence_length_explicit > 1) {
        throw std::invalid_argument(
            "LoadOptionsV2.max_sequence_length_explicit must be zero or one");
    }

    const bool has_fraction = options.kv_cache_memory_fraction != 0.0;
    const bool has_bytes = options.kv_cache_memory_bytes != 0;
    switch (options.kv_cache_memory_policy) {
    case KvCacheMemoryPolicy::kUnspecified:
    case KvCacheMemoryPolicy::kAuto:
        if (has_fraction || has_bytes) {
            throw std::invalid_argument(
                "Automatic or unspecified KV cache policy conflicts with fraction/byte values");
        }
        break;
    case KvCacheMemoryPolicy::kFraction:
        if (!(std::isfinite(options.kv_cache_memory_fraction) &&
              options.kv_cache_memory_fraction > 0.0 && options.kv_cache_memory_fraction <= 1.0)) {
            throw std::invalid_argument("KV cache memory fraction must be in (0, 1]");
        }
        if (has_bytes) {
            throw std::invalid_argument("KV cache fraction policy requires zero bytes");
        }
        break;
    case KvCacheMemoryPolicy::kBytes:
        if (!has_bytes)
            throw std::invalid_argument("KV cache byte policy requires positive bytes");
        if (has_fraction) {
            throw std::invalid_argument("KV cache byte policy requires zero fraction");
        }
        break;
    default:
        throw std::invalid_argument("Unknown KV cache memory policy");
    }

    const bool requests_new_policy =
        options.kv_cache_memory_policy != KvCacheMemoryPolicy::kUnspecified ||
        options.max_sequence_length != 0 || options.max_sequence_length_explicit != 0;
    if (requests_new_policy && options.kv_cache_size_bytes != 0) {
        throw std::invalid_argument(
            "LoadOptionsV2 legacy kv_cache_size_bytes conflicts with runtime KV policy fields");
    }
}

bool requests_runtime_kv_policy(const LoadOptionsV2& options) {
    return options.kv_cache_memory_policy != KvCacheMemoryPolicy::kUnspecified ||
           options.max_sequence_length != 0 || options.max_sequence_length_explicit != 0;
}

void validate_runtime_kv_policy_support(const BundleInfo& header, bool requested) {
    if (!requested)
        return;
    if (header.runtime_memory.present && header.runtime_memory.contract_version == 1 &&
        header.runtime_memory.runtime_owned) {
        return;
    }
    throw std::invalid_argument("This bundle does not declare runtime_memory contract version 1; "
                                "runtime KV memory and max-sequence policies cannot be applied");
}

LoadOptions legacy_load_options(const LoadOptionsV2& options) {
    LoadOptions legacy;
    legacy.hf_python = options.hf_python;
    legacy.runtime_cache_path = options.runtime_cache_path;
    legacy.cuda_graphs = options.cuda_graphs;
    legacy.kv_cache_size_bytes = options.kv_cache_size_bytes;
    legacy.config_path = options.config_path;
    legacy.set_tokens = options.set_tokens;
    legacy.backend_search_paths = options.backend_search_paths;
    legacy.model_plugin_search_paths = options.model_plugin_search_paths;
    return legacy;
}

RuntimeMemoryPluginOptionsV1 runtime_memory_plugin_options(const LoadOptions& legacy,
                                                           const LoadOptionsV2* options) {
    RuntimeMemoryPluginOptionsV1 policy;
    if (options == nullptr) {
        if (legacy.kv_cache_size_bytes != 0) {
            policy.kv_cache_memory_policy = KvCacheMemoryPolicy::kBytes;
            policy.kv_cache_memory_fraction = 0.0;
            policy.kv_cache_memory_bytes = legacy.kv_cache_size_bytes;
        }
        return policy;
    }
    if (options->kv_cache_memory_policy == KvCacheMemoryPolicy::kUnspecified &&
        legacy.kv_cache_size_bytes != 0) {
        policy.kv_cache_memory_policy = KvCacheMemoryPolicy::kBytes;
        policy.kv_cache_memory_fraction = 0.0;
        policy.kv_cache_memory_bytes = legacy.kv_cache_size_bytes;
        policy.max_sequence_length = options->max_sequence_length;
        policy.max_sequence_length_explicit =
            options->max_sequence_length != 0 ? 1U : options->max_sequence_length_explicit;
        return policy;
    }

    policy.kv_cache_memory_policy =
        options->kv_cache_memory_policy == KvCacheMemoryPolicy::kUnspecified
            ? KvCacheMemoryPolicy::kAuto
            : options->kv_cache_memory_policy;
    policy.kv_cache_memory_fraction = policy.kv_cache_memory_policy == KvCacheMemoryPolicy::kAuto
                                          ? 0.90
                                          : options->kv_cache_memory_fraction;
    policy.kv_cache_memory_bytes = options->kv_cache_memory_bytes;
    policy.max_sequence_length = options->max_sequence_length;
    policy.max_sequence_length_explicit =
        options->max_sequence_length != 0 ? 1U : options->max_sequence_length_explicit;
    return policy;
}

IRuntimeMemoryPipelinePluginV1& require_runtime_memory_plugin(IPipelinePlugin& plugin,
                                                              const std::string& strategy) {
    auto* runtime_plugin = dynamic_cast<IRuntimeMemoryPipelinePluginV1*>(&plugin);
    if (runtime_plugin == nullptr) {
        throw std::runtime_error("runtime_memory bundle strategy '" + strategy +
                                 "' requires model plugin runtime-memory interface V1; "
                                 "the loaded model DSO is incompatible with this core");
    }
    if (runtime_plugin->runtime_memory_plugin_api_version() != kRuntimeMemoryPluginApiVersionV1) {
        throw std::runtime_error(
            "runtime_memory bundle strategy '" + strategy +
            "' reported an incompatible model plugin runtime-memory API version");
    }
    return *runtime_plugin;
}

IPipelinePlugin* lookup_plugin_or_throw(const std::string& strategy,
                                        const std::vector<std::string>& model_plugin_paths,
                                        ModelPluginAbiPolicy abi_policy) {
    load_model_plugin_for_strategy_with_abi_policy(strategy, model_plugin_paths, abi_policy);
    auto* plugin = PipelineRegistry::instance().lookup(strategy);
    if (plugin != nullptr)
        return plugin;
    std::string available;
    for (const auto& s : PipelineRegistry::instance().registered_strategies()) {
        if (!available.empty())
            available += ", ";
        available += s;
    }
    throw std::runtime_error("No plugin registered for runtime_strategy: " + strategy +
                             " (available: " + available + ")");
}

// Apply platform.* values to their process-wide sinks. Replaces the old
// TRTMC_DATA_DIR and TRTMC_TRT_LOG_{STDERR,MIN_SEVERITY} env-var reads.
// Called from try_resolve_runtime_config once a bundle has resolved.
void apply_platform_config(const config::ConfigBundle& bundle) {
    try {
        const std::string source = bundle.get<std::string>("platform", "source_dir");
        if (!source.empty())
            set_source_dir(source);
        const bool verbose_stderr = bundle.get<bool>("platform", "trt_log_stderr");
        const std::string severity = bundle.get<std::string>("platform", "trt_log_min_severity");
        configure_trt_logger(verbose_stderr, severity);
    } catch (const std::exception&) {
        // Schema absent or type mismatch — leave sinks at defaults.
    }
}

std::string bundle_trt_version_text(const BundleFile& bundle, const std::string& config_text) {
    if (!bundle.info.trt_version.empty() && bundle.info.trt_version != "unknown")
        return bundle.info.trt_version;
    return extract_json_string(config_text, "trt_version", "");
}

std::optional<TrtVersion> required_trt_version_for_bundle(const BundleFile& bundle,
                                                          const std::string& config_text,
                                                          const std::string& backend_name) {
    if (auto parsed = parse_trt_version(bundle_trt_version_text(bundle, config_text)))
        return parsed;
    if (auto parsed = parse_trt_abi_tag(extract_json_string(config_text, "trt_abi", "")))
        return parsed;
    if (auto parsed = parse_trt_abi_tag(backend_name))
        return parsed;
    return std::nullopt;
}

void throw_trt_mismatch(const std::string& bundle_path, const TrtVersion& required,
                        const TrtVersion& actual, const std::string& actual_label) {
    throw std::runtime_error("TensorRT version mismatch for bundle " + bundle_path +
                             ": bundle was built with TensorRT " + format_trt_version(required) +
                             " (ABI " + trt_abi_string(required) + "), but " + actual_label +
                             " is " + format_trt_version(actual) + " (ABI " +
                             trt_abi_string(actual) +
                             "). Rebuild the bundle with the installed TensorRT version or "
                             "install a matching TensorRT runtime/backend DSO.");
}

void enforce_trt_compatibility(const std::string& bundle_path,
                               const std::optional<TrtVersion>& required,
                               const std::optional<TrtVersion>& actual,
                               const std::string& actual_label) {
    if (!required || !actual)
        return;
    if (!trt_abi_matches(*required, *actual))
        throw_trt_mismatch(bundle_path, *required, *actual, actual_label);
}

IBackend* load_backend_for_bundle(const BundleFile& bundle, const std::string& config_text,
                                  const std::string& bundle_path, const std::string& backend_name,
                                  const std::vector<std::string>& backend_search_paths) {
    const std::string logical_backend = backend_name.empty() ? "trt" : backend_name;
    if (!is_standard_trt_backend_name(logical_backend)) {
        return BackendLoader::load(logical_backend, backend_search_paths);
    }

    const auto required = required_trt_version_for_bundle(bundle, config_text, logical_backend);
    std::string detection_diagnostics;
    std::optional<TrtVersion> installed;
    if (required) {
        auto matched =
            find_trt_library_for_version(*required, backend_search_paths, &detection_diagnostics);
        if (!matched) {
            throw std::runtime_error(
                "TensorRT runtime for bundle " + bundle_path +
                " is not available: bundle requires TensorRT ABI " + trt_abi_string(*required) +
                ". Searched candidate libnvinfer paths:\n" + detection_diagnostics);
        }
        installed = matched->version;
        if (!matched->already_loaded)
            BackendLoader::preload_dependency(matched->path);
    } else {
        installed = detect_installed_trt_version(backend_search_paths, &detection_diagnostics);
    }
    enforce_trt_compatibility(bundle_path, required, installed, "installed TensorRT runtime");

    const auto candidates = trt_backend_candidates(logical_backend, required, installed);
    std::string loaded_backend_name;
    BackendLoadMetadata metadata;
    IBackend* backend = BackendLoader::load_first_available(candidates, backend_search_paths,
                                                            &loaded_backend_name, &metadata);

    enforce_trt_compatibility(bundle_path, required, parse_trt_abi_tag(metadata.trt_abi),
                              "selected backend DSO ABI");
    enforce_trt_compatibility(bundle_path, required,
                              parse_trt_version(metadata.trt_runtime_version),
                              "selected backend TensorRT runtime");

    if (bundle.info.runtime_memory.present) {
        const auto actual =
            parse_runtime_memory_runtime_stack_json(metadata.runtime_memory_stack_json);
        validate_runtime_memory_runtime_stack(bundle.info.runtime_memory.qualified_runtime_stack,
                                              actual);
        std::cerr << "[trtmc.runtime_stack] schema=1 sm=sm" << actual.compute_capability_major
                  << actual.compute_capability_minor << " tensorrt=" << actual.trt_runtime_version
                  << " cuda_runtime=" << actual.cuda_runtime_version
                  << " cudnn_backend=" << actual.cudnn_backend_version
                  << " cudnn_frontend_revision=" << actual.cudnn_frontend_revision
                  << " nvrtc=" << actual.nvrtc_version << " driver=" << actual.driver_version
                  << std::endl;
    }

    if (required) {
        std::cerr << "[trtmc] TensorRT ABI resolved: bundle=" << trt_abi_string(*required)
                  << ", backend=" << loaded_backend_name;
        if (!metadata.trt_runtime_version.empty())
            std::cerr << ", runtime=" << metadata.trt_runtime_version;
        std::cerr << std::endl;
    }
    return backend;
}

std::optional<config::ConfigBundle>
try_resolve_runtime_config(const std::string& config_text, const std::string& bundle_path,
                           const std::string& config_path,
                           const std::vector<std::string>& set_tokens) {
    try {
        auto resolution = config::resolve_pipeline_config(config_text, config_path, set_tokens);
        config::write_effective_config_next_to(resolution.bundle, bundle_path);
        apply_platform_config(resolution.bundle);
        return std::move(resolution.bundle);
    } catch (const std::exception& e) {
        std::cerr << "[trtmc.config] Failed to resolve runtime config: " << e.what()
                  << "\n          Proceeding with schema defaults.\n";
        return std::nullopt;
    }
}

} // namespace

namespace {

std::unique_ptr<IPipeline> from_bundle_with_options(const std::string& bundle_path,
                                                    const LoadOptions& options,
                                                    const LoadOptionsV2* versioned_options) {
    const BundleInfo header = ReadBundleHeader(bundle_path);
    const bool explicit_runtime_policy =
        versioned_options != nullptr && requests_runtime_kv_policy(*versioned_options);
    validate_runtime_kv_policy_support(header, explicit_runtime_policy);
    if (header.runtime_memory.present && is_optimized_runtime_bundle(header)) {
        throw std::invalid_argument(
            "runtime_memory bundles are supported only by the native Model Connect runtime");
    }
    if (auto optimized_runtime_pipeline =
            try_make_optimized_runtime_pipeline(bundle_path, header, options)) {
        return optimized_runtime_pipeline;
    }
    auto materialized = detail::materialize_pipeline_bundle(bundle_path);
    BundleFile bundle = std::move(materialized.bundle);
    std::string config_text = std::move(materialized.config_text);
    if (bundle.sections.empty())
        throw std::runtime_error("Failed to read bundle: " + bundle_path);

    std::string strategy = resolve_runtime_strategy(config_text);

    auto* plugin = lookup_plugin_or_throw(strategy, options.model_plugin_search_paths,
                                          header.runtime_memory.present
                                              ? ModelPluginAbiPolicy::kRequireCurrent
                                              : ModelPluginAbiPolicy::kAllowLegacyUnversioned);
    IRuntimeMemoryPipelinePluginV1* runtime_plugin = nullptr;
    RuntimeMemoryPluginOptionsV1 runtime_policy;
    if (header.runtime_memory.present) {
        runtime_plugin = &require_runtime_memory_plugin(*plugin, strategy);
        runtime_policy = runtime_memory_plugin_options(options, versioned_options);
    }

    std::string backend_name = extract_json_string(config_text, "engine_backend", "trt");
    IBackend* backend = load_backend_for_bundle(bundle, config_text, bundle_path, backend_name,
                                                options.backend_search_paths);

    BaseConfig base_cfg = parse_base_config(config_text, bundle.info.max_cache_length);
    base_cfg.runtime_strategy = strategy;
    if (!base_cfg.tokenizer_add_special_tokens_present &&
        bundle.info.tokenizer_add_special_tokens_present) {
        base_cfg.tokenizer_add_special_tokens = bundle.info.tokenizer_add_special_tokens;
        base_cfg.tokenizer_add_special_tokens_present = true;
    }

    std::optional<config::ConfigBundle> resolved = try_resolve_runtime_config(
        config_text, bundle_path, options.config_path, options.set_tokens);

    PipelineContext ctx{bundle,
                        base_cfg,
                        config_text,
                        options.hf_python,
                        bundle_path,
                        backend,
                        options.runtime_cache_path,
                        options.cuda_graphs,
                        header.runtime_memory.present ? 0 : options.kv_cache_size_bytes,
                        resolved ? &*resolved : nullptr};
    auto pipeline = runtime_plugin != nullptr
                        ? runtime_plugin->create_runtime_memory(ctx, runtime_policy)
                        : plugin->create(ctx);

    std::cerr << "[trtmc] Pipeline loaded (strategy=" << strategy << ", backend=trt_new_runtime)"
              << std::endl;
    return pipeline;
}

} // namespace

std::unique_ptr<IPipeline> PipelineFactory::from_bundle(const std::string& bundle_path,
                                                        const std::string& hf_python,
                                                        const std::string& runtime_cache_path,
                                                        bool cuda_graphs) {
    LoadOptions options;
    options.hf_python = hf_python;
    options.runtime_cache_path = runtime_cache_path;
    options.cuda_graphs = cuda_graphs;
    return from_bundle_with_options(bundle_path, options, nullptr);
}

std::unique_ptr<IPipeline> PipelineFactory::from_bundle(const std::string& bundle_path,
                                                        const LoadOptions& options) {
    return from_bundle_with_options(bundle_path, options, nullptr);
}

std::unique_ptr<IPipeline> PipelineFactory::from_bundle(const std::string& bundle_path,
                                                        const LoadOptionsV2& options) {
    validate_load_options_v2(options);
    auto legacy = legacy_load_options(options);
    return from_bundle_with_options(bundle_path, legacy, &options);
}

std::unique_ptr<PipelinePool> PipelineFactory::from_bundle_pool(const std::string& bundle_path,
                                                                std::size_t pool_size,
                                                                const LoadOptions& options) {
    if (pool_size == 0)
        throw std::invalid_argument("Pipeline pool size must be positive");

    const BundleInfo header = ReadBundleHeader(bundle_path);
    if (header.runtime_memory.present) {
        throw std::invalid_argument(
            "PipelinePool does not yet support runtime-sized KV cache bundles; "
            "the beta owns one post-load KV budget per pipeline. Use "
            "PipelineFactory::from_bundle until pool-level budget partitioning is implemented.");
    }
    if (is_optimized_runtime_bundle(header))
        throw std::invalid_argument(
            "PipelineFactory::from_bundle_pool does not support optimized-runtime bundles; use "
            "from_bundle because the delegated runtime owns batching and scheduling");

    auto materialized = detail::materialize_pipeline_bundle(bundle_path);
    BundleFile bundle = std::move(materialized.bundle);
    std::string config_text = std::move(materialized.config_text);
    if (bundle.sections.empty())
        throw std::runtime_error("Failed to read bundle: " + bundle_path);

    std::string strategy = resolve_runtime_strategy(config_text);
    auto* plugin = lookup_plugin_or_throw(strategy, options.model_plugin_search_paths,
                                          ModelPluginAbiPolicy::kAllowLegacyUnversioned);
    std::string backend_name = extract_json_string(config_text, "engine_backend", "trt");
    IBackend* backend = load_backend_for_bundle(bundle, config_text, bundle_path, backend_name,
                                                options.backend_search_paths);

    BaseConfig base_cfg = parse_base_config(config_text, bundle.info.max_cache_length);
    base_cfg.runtime_strategy = strategy;
    std::optional<config::ConfigBundle> resolved = try_resolve_runtime_config(
        config_text, bundle_path, options.config_path, options.set_tokens);

    PipelineContext ctx{bundle,
                        base_cfg,
                        config_text,
                        options.hf_python,
                        bundle_path,
                        backend,
                        options.runtime_cache_path,
                        options.cuda_graphs,
                        options.kv_cache_size_bytes,
                        resolved ? &*resolved : nullptr};
    auto pipelines = plugin->create_pool(ctx, pool_size);
    auto pool = std::make_unique<PipelinePool>(std::move(pipelines));

    std::cerr << "[trtmc] Pipeline pool loaded (strategy=" << strategy << ", lanes=" << pool_size
              << ", backend=trt_new_runtime)" << std::endl;
    return pool;
}

std::unique_ptr<IPipeline> load(const std::string& bundle_path, const std::string& hf_python,
                                const std::string& runtime_cache_path, bool cuda_graphs) {
    return PipelineFactory::from_bundle(bundle_path, hf_python, runtime_cache_path, cuda_graphs);
}

std::unique_ptr<IPipeline> load(const std::string& bundle_path, const LoadOptions& options) {
    return PipelineFactory::from_bundle(bundle_path, options);
}

std::unique_ptr<IPipeline> load(const std::string& bundle_path, const LoadOptionsV2& options) {
    return PipelineFactory::from_bundle(bundle_path, options);
}

} // namespace trtmc

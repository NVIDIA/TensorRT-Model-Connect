/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/pipeline_factory.h"

#include "bundle/bundle_format.h"
#include "runtime/backend/backend_loader.h"
#include "runtime/backend/trt_version.h"
#include "runtime/core/trt_common.h"
#include "runtime/providers/optimized_runtime_host.h"
#include "runtime/registry/bundle_materialization.h"
#include "runtime/registry/runtime_config_resolution.h"
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

IPipelinePlugin* lookup_plugin_or_throw(const std::string& strategy,
                                        const std::vector<std::string>& model_plugin_paths) {
    load_model_plugin_for_strategy(strategy, model_plugin_paths);
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
// Called once a bundle's layered runtime config has resolved.
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

    if (required) {
        std::cerr << "[trtmc] TensorRT ABI resolved: bundle=" << trt_abi_string(*required)
                  << ", backend=" << loaded_backend_name;
        if (!metadata.trt_runtime_version.empty())
            std::cerr << ", runtime=" << metadata.trt_runtime_version;
        std::cerr << std::endl;
    }
    return backend;
}

} // namespace

std::optional<config::ConfigBundle>
detail::resolve_runtime_config(const std::string& config_text, const std::string& bundle_path,
                               const std::string& config_path,
                               const std::vector<std::string>& set_tokens) {
    try {
        auto resolution = config::resolve_pipeline_config(config_text, config_path, set_tokens);
        try {
            config::write_effective_config_next_to(resolution.bundle, bundle_path);
        } catch (const std::exception& e) {
            std::cerr << "[trtmc.config] Failed to write effective config sidecar: " << e.what()
                      << "\n          Resolved runtime config remains active.\n";
        }
        apply_platform_config(resolution.bundle);
        return std::move(resolution.bundle);
    } catch (const std::exception& e) {
        std::cerr << "[trtmc.config] Failed to resolve runtime config: " << e.what()
                  << "\n          Proceeding with schema defaults.\n";
        return std::nullopt;
    }
}

std::unique_ptr<IPipeline> PipelineFactory::from_bundle(const std::string& bundle_path,
                                                        const std::string& hf_python,
                                                        const std::string& runtime_cache_path,
                                                        bool cuda_graphs) {
    LoadOptions optimized_options;
    optimized_options.hf_python = hf_python;
    optimized_options.runtime_cache_path = runtime_cache_path;
    optimized_options.cuda_graphs = cuda_graphs;
    const BundleInfo header = ReadBundleHeader(bundle_path);
    if (auto optimized_runtime_pipeline =
            try_make_optimized_runtime_pipeline(bundle_path, header, optimized_options)) {
        return optimized_runtime_pipeline;
    }

    auto materialized = detail::materialize_pipeline_bundle(bundle_path);
    BundleFile bundle = std::move(materialized.bundle);
    std::string config_text = std::move(materialized.config_text);
    if (bundle.sections.empty())
        throw std::runtime_error("Failed to read bundle: " + bundle_path);

    // Parse runtime_strategy and normalize legacy strings.
    std::string strategy = resolve_runtime_strategy(config_text);

    auto* plugin = lookup_plugin_or_throw(strategy, {});

    // Load backend DSO based on bundle metadata after strategy ownership is known.
    std::string backend_name = extract_json_string(config_text, "engine_backend", "trt");
    IBackend* backend = load_backend_for_bundle(bundle, config_text, bundle_path, backend_name, {});

    // Parse base config and dispatch to plugin
    BaseConfig base_cfg = parse_base_config(config_text, bundle.info.max_cache_length);
    base_cfg.runtime_strategy = strategy; // use normalized strategy
    if (!base_cfg.tokenizer_add_special_tokens_present &&
        bundle.info.tokenizer_add_special_tokens_present) {
        base_cfg.tokenizer_add_special_tokens = bundle.info.tokenizer_add_special_tokens;
        base_cfg.tokenizer_add_special_tokens_present = true;
    }

    // Resolve the layered runtime config (BUNDLE_DEFAULT + SESSION_REQUEST).
    // Best-effort: a malformed input prints to stderr and falls back to
    // schema defaults so plugin construction isn't blocked.
    std::optional<config::ConfigBundle> resolved =
        detail::resolve_runtime_config(config_text, bundle_path, /*config_path=*/"",
                                       /*set_tokens=*/{});

    PipelineContext ctx{bundle,
                        base_cfg,
                        config_text,
                        hf_python,
                        bundle_path,
                        backend,
                        runtime_cache_path,
                        cuda_graphs,
                        /*kv_cache_size_bytes=*/0,
                        resolved ? &*resolved : nullptr};
    auto pipeline = plugin->create(ctx);

    std::cerr << "[trtmc] Pipeline loaded (strategy=" << strategy << ", backend=trt_new_runtime)"
              << std::endl;
    return pipeline;
}

std::unique_ptr<IPipeline> PipelineFactory::from_bundle(const std::string& bundle_path,
                                                        const LoadOptions& options) {
    const BundleInfo header = ReadBundleHeader(bundle_path);
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

    auto* plugin = lookup_plugin_or_throw(strategy, options.model_plugin_search_paths);

    std::string backend_name = extract_json_string(config_text, "engine_backend", "trt");
    IBackend* backend = load_backend_for_bundle(bundle, config_text, bundle_path, backend_name,
                                                options.backend_search_paths);

    BaseConfig base_cfg = parse_base_config(config_text, bundle.info.max_cache_length);
    base_cfg.runtime_strategy = strategy;

    std::optional<config::ConfigBundle> resolved = detail::resolve_runtime_config(
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
    auto pipeline = plugin->create(ctx);

    std::cerr << "[trtmc] Pipeline loaded (strategy=" << strategy << ", backend=trt_new_runtime)"
              << std::endl;
    return pipeline;
}

std::unique_ptr<PipelinePool> PipelineFactory::from_bundle_pool(const std::string& bundle_path,
                                                                std::size_t pool_size,
                                                                const LoadOptions& options) {
    if (pool_size == 0)
        throw std::invalid_argument("Pipeline pool size must be positive");

    const BundleInfo header = ReadBundleHeader(bundle_path);
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
    auto* plugin = lookup_plugin_or_throw(strategy, options.model_plugin_search_paths);
    std::string backend_name = extract_json_string(config_text, "engine_backend", "trt");
    IBackend* backend = load_backend_for_bundle(bundle, config_text, bundle_path, backend_name,
                                                options.backend_search_paths);

    BaseConfig base_cfg = parse_base_config(config_text, bundle.info.max_cache_length);
    base_cfg.runtime_strategy = strategy;
    std::optional<config::ConfigBundle> resolved = detail::resolve_runtime_config(
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

} // namespace trtmc

#include "trtmc/runtime/pipeline_factory.h"

#include "bundle/bundle_format.h"
#include "runtime/backend/backend_loader.h"
#include "runtime/backend/trt_version.h"
#include "runtime/core/trt_common.h"
#include "trtmc/config/cli_support.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/config/schema_registry.h"
#include "trtmc/runtime/pipeline_plugin.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "trtmc/runtime/trt_backend.h"
#include "utils/data_dir.h"
#include "utils/json_helpers.h"

#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc {

namespace {

// Rewrite legacy ambiguous strategy strings from old bundles into their
// unambiguous per-model equivalents. New bundles already use the split strings.
bool json_field_is_truthy(const std::string& config_text, const char* key) {
    auto pos = config_text.find(std::string("\"") + key + "\"");
    if (pos == std::string::npos)
        return false;
    auto colon = config_text.find(':', pos);
    if (colon == std::string::npos)
        return false;
    auto rest = config_text.substr(colon + 1, 20);
    return rest.find("true") != std::string::npos || rest.find('1') != std::string::npos;
}

std::string json_field_substr(const std::string& config_text, const char* key) {
    auto pos = config_text.find(std::string("\"") + key + "\"");
    if (pos == std::string::npos)
        return "";
    auto colon = config_text.find(':', pos);
    if (colon == std::string::npos)
        return "";
    return config_text.substr(colon, 40);
}

std::string normalize_legacy_strategy(const std::string& strategy, const std::string& config_text) {
    if (strategy == "text_to_audio") {
        return json_field_is_truthy(config_text, "magpie_tts") ? "text_to_audio_magpie"
                                                               : "text_to_audio_bark";
    }
    if (strategy == "diffusion") {
        auto bt = json_field_substr(config_text, "diffusion_backend_type");
        if (bt.find("flux") != std::string::npos)
            return "diffusion_flux";
        if (bt.find("z_image") != std::string::npos)
            return "diffusion_zimage";
        if (bt.find("pixart") != std::string::npos)
            return "diffusion_pixart";
        return "diffusion_wan";
    }
    // Legacy torch-trt diffusion bundles -> pixart torch-trt
    if (strategy == "torchtrt_diffusion") {
        return "diffusion_pixart_torchtrt";
    }
    return strategy;
}

IPipelinePlugin* lookup_plugin_or_throw(const std::string& strategy) {
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

bool strategy_uses_no_backend(const std::string& strategy) {
    return strategy == "diffusion_sana_wm";
}

bool backend_is_disabled(const std::string& backend_name) {
    return backend_name == "none";
}

const char* backend_log_name(IBackend* backend) {
    return backend ? "trt_new_runtime" : "none";
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

    if (required) {
        std::cerr << "[trtmc] TensorRT ABI resolved: bundle=" << trt_abi_string(*required)
                  << ", backend=" << loaded_backend_name;
        if (!metadata.trt_runtime_version.empty())
            std::cerr << ", runtime=" << metadata.trt_runtime_version;
        std::cerr << std::endl;
    }
    return backend;
}

IBackend* load_optional_backend(const BundleFile& bundle, const std::string& config_text,
                                const std::string& bundle_path, const std::string& backend_name,
                                const std::string& strategy,
                                const std::vector<std::string>& backend_search_paths) {
    if (strategy_uses_no_backend(strategy) || backend_is_disabled(backend_name))
        return nullptr;
    return load_backend_for_bundle(bundle, config_text, bundle_path, backend_name,
                                   backend_search_paths);
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

std::unique_ptr<IPipeline> PipelineFactory::from_bundle(const std::string& bundle_path,
                                                        const std::string& hf_python,
                                                        const std::string& runtime_cache_path,
                                                        bool cuda_graphs) {
    BundleFile bundle = ReadBundleFile(bundle_path);
    if (bundle.sections.empty())
        throw std::runtime_error("Failed to read bundle: " + bundle_path);

    // Extract config JSON from bundle
    std::string config_text;
    for (const auto& section : bundle.sections) {
        if (section.name == "config.json" && !section.data.empty()) {
            config_text.assign(section.data.begin(), section.data.end());
            break;
        }
    }

    // Parse runtime_strategy and normalize legacy strings
    std::string strategy = extract_json_string(config_text, "runtime_strategy", "decoder_kv_cache");
    if (strategy.empty())
        strategy = "decoder_kv_cache";
    strategy = normalize_legacy_strategy(strategy, config_text);

    // Load backend DSO based on bundle metadata. Bridge-only strategies do
    // not own TRT engine sections, so keep their context backend null.
    std::string backend_name = extract_json_string(config_text, "engine_backend", "trt");
    IBackend* backend =
        load_optional_backend(bundle, config_text, bundle_path, backend_name, strategy, {});

    auto* plugin = lookup_plugin_or_throw(strategy);

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
        try_resolve_runtime_config(config_text, bundle_path, /*config_path=*/"",
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

    std::cerr << "[trtmc] Pipeline loaded (strategy=" << strategy
              << ", backend=" << backend_log_name(backend) << ")" << std::endl;
    return pipeline;
}

std::unique_ptr<IPipeline> PipelineFactory::from_bundle(const std::string& bundle_path,
                                                        const LoadOptions& options) {
    BundleFile bundle = ReadBundleFile(bundle_path);
    if (bundle.sections.empty())
        throw std::runtime_error("Failed to read bundle: " + bundle_path);

    std::string config_text;
    for (const auto& section : bundle.sections) {
        if (section.name == "config.json" && !section.data.empty()) {
            config_text.assign(section.data.begin(), section.data.end());
            break;
        }
    }

    std::string strategy = extract_json_string(config_text, "runtime_strategy", "decoder_kv_cache");
    if (strategy.empty())
        strategy = "decoder_kv_cache";
    strategy = normalize_legacy_strategy(strategy, config_text);

    std::string backend_name = extract_json_string(config_text, "engine_backend", "trt");
    IBackend* backend = load_optional_backend(bundle, config_text, bundle_path, backend_name,
                                              strategy, options.backend_search_paths);

    auto* plugin = lookup_plugin_or_throw(strategy);

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
    auto pipeline = plugin->create(ctx);

    std::cerr << "[trtmc] Pipeline loaded (strategy=" << strategy
              << ", backend=" << backend_log_name(backend) << ")" << std::endl;
    return pipeline;
}

std::unique_ptr<IPipeline> load(const std::string& bundle_path, const std::string& hf_python,
                                const std::string& runtime_cache_path, bool cuda_graphs) {
    return PipelineFactory::from_bundle(bundle_path, hf_python, runtime_cache_path, cuda_graphs);
}

std::unique_ptr<IPipeline> load(const std::string& bundle_path, const LoadOptions& options) {
    return PipelineFactory::from_bundle(bundle_path, options);
}

} // namespace trtmc

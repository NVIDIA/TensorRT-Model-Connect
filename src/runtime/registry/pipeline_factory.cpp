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
#include "utils/sha256.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <exception>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc {

namespace detail {
namespace {

constexpr std::string_view kWan22RuntimeStrategy = "diffusion_wan2_2_ti2v";
constexpr std::string_view kWan22ArtifactSchemaV1 = "trtmc.wan2_2_ti2v.bundle-artifacts.v1";
constexpr std::string_view kWan22ArtifactSchemaV2 = "trtmc.wan2_2_ti2v.bundle-artifacts.v2";
constexpr std::array<std::string_view, 9> kWan22RequiredSections = {
    "wan2_2_umt5_cuda_plugin_so",
    "wan2_2_dit_cuda_plugin_so",
    "wan2_2_vae_cuda_plugin_so",
    "text_encoder_0_plan",
    "denoiser_plan",
    "vae_decoder_plan",
    "vae_decoder_first_frame_plan",
    "tokenizer.json",
    "config.json",
};
constexpr std::array<std::string_view, 8> kWan22ManifestSections = {
    "wan2_2_umt5_cuda_plugin_so",
    "wan2_2_dit_cuda_plugin_so",
    "wan2_2_vae_cuda_plugin_so",
    "text_encoder_0_plan",
    "denoiser_plan",
    "vae_decoder_plan",
    "vae_decoder_first_frame_plan",
    "tokenizer.json",
};

bool json_object_has_exact_keys(const nlohmann::json& object,
                                std::initializer_list<std::string_view> expected) {
    if (!object.is_object() || object.size() != expected.size())
        return false;
    for (const auto key : expected) {
        if (!object.contains(std::string(key)))
            return false;
    }
    return true;
}

bool is_lowercase_sha256(const nlohmann::json& value) {
    if (!value.is_string())
        return false;
    const auto& text = value.get_ref<const std::string&>();
    return text.size() == 64 && std::all_of(text.begin(), text.end(), [](unsigned char character) {
               return std::isdigit(character) != 0 ||
                      (character >= static_cast<unsigned char>('a') &&
                       character <= static_cast<unsigned char>('f'));
           });
}

std::uint64_t require_nonzero_size(const nlohmann::json& value, const std::string& section_name) {
    if (!value.is_number_integer() && !value.is_number_unsigned()) {
        throw std::runtime_error("Wan2.2 artifact size is not an integer for " + section_name);
    }
    if (value.is_number_integer() && value.get<std::int64_t>() <= 0) {
        throw std::runtime_error("Wan2.2 artifact size must be positive for " + section_name);
    }
    const auto result = value.get<std::uint64_t>();
    if (result == 0)
        throw std::runtime_error("Wan2.2 artifact size must be positive for " + section_name);
    return result;
}

std::string hash_bundle_section(BundleSectionReader& reader, const std::string& section_name) {
    Sha256 digest;
    reader.for_each_chunk(section_name, 4U << 20U, [&digest](const char* data, std::size_t size) {
        digest.update(data, size);
    });
    return digest.hex_digest();
}

template <std::size_t Size>
nlohmann::json string_array_json(const std::array<std::string_view, Size>& values) {
    nlohmann::json result = nlohmann::json::array();
    for (const auto value : values)
        result.push_back(value);
    return result;
}

void validate_wan22_artifact_provenance(BundleSectionReader& reader, const nlohmann::json& config,
                                        std::size_t materialized_config_size) {
    const auto& info = reader.info();
    if (info.sections.size() != kWan22RequiredSections.size()) {
        throw std::runtime_error("Wan2.2 provenance requires exactly nine bundle sections");
    }

    std::unordered_map<std::string, const BundleSectionInfo*> section_info;
    section_info.reserve(info.sections.size());
    for (const auto& section : info.sections) {
        if (section.name.empty() || !section_info.emplace(section.name, &section).second) {
            throw std::runtime_error("Duplicate or empty Wan2.2 bundle section: " + section.name);
        }
    }
    for (const auto required : kWan22RequiredSections) {
        if (section_info.find(std::string(required)) == section_info.end()) {
            throw std::runtime_error("Wan2.2 provenance is missing bundle section: " +
                                     std::string(required));
        }
    }
    if (section_info.at("config.json")->size != materialized_config_size) {
        throw std::runtime_error("Wan2.2 config.json size disagrees with bundle metadata");
    }

    const auto contract = config.find("runtime_contract");
    if (contract == config.end() || !contract->is_object() ||
        !contract->contains("required_bundle_sections") ||
        (*contract)["required_bundle_sections"] != string_array_json(kWan22RequiredSections)) {
        throw std::runtime_error(
            "Wan2.2 runtime_contract must declare the exact nine-section integrity contract");
    }

    const auto manifest_iterator = config.find("artifact_manifest");
    if (manifest_iterator == config.end() ||
        !json_object_has_exact_keys(*manifest_iterator,
                                    {"schema", "family", "profile", "runtime", "sections"})) {
        throw std::runtime_error("Wan2.2 artifact_manifest is missing or malformed");
    }
    const auto& manifest = *manifest_iterator;
    const std::string schema = manifest.value("schema", std::string{});
    const bool has_explicit_sizes = schema == kWan22ArtifactSchemaV2;
    if ((schema != kWan22ArtifactSchemaV1 && !has_explicit_sizes) ||
        manifest.value("family", std::string{}) != "wan2_2_ti2v" ||
        manifest.value("runtime", std::string{}) != "native_cpp_cuda_tensorrt") {
        throw std::runtime_error("Wan2.2 artifact_manifest identity is invalid");
    }
    if (has_explicit_sizes &&
        contract->value("artifact_integrity", std::string{}) != "sha256_size_v1") {
        throw std::runtime_error(
            "Wan2.2 v2 artifact manifest requires sha256_size_v1 integrity mode");
    }
    const nlohmann::json expected_profile = {
        {"video_width", 1280},
        {"video_height", 704},
        {"video_num_frames", 121},
        {"latent_shape", {1, 48, 31, 44, 80}},
        {"architecture",
         {{"model_type", "ti2v"},
          {"in_channels", 48},
          {"out_channels", 48},
          {"dim", 3072},
          {"ffn_dim", 14336},
          {"freq_dim", 256},
          {"num_heads", 24},
          {"num_layers", 30},
          {"head_dim", 128},
          {"text_dim", 4096},
          {"text_seq_len", 512},
          {"eps", 1e-6},
          {"patch_size", {1, 2, 2}},
          {"z_dim", 48},
          {"scale_factor_temporal", 4},
          {"scale_factor_spatial", 16},
          {"frame_rate", 24},
          {"num_inference_steps", 50},
          {"guidance_scale", 5.0},
          {"flow_shift", 5.0},
          {"train_timesteps", 1000}}},
        {"text_seq_len", 512},
        {"text_encoder_dim", 4096},
        {"text_encoder_numerics",
         {{"shape", {1, 512, 4096}},
          {"num_heads", 64},
          {"epsilon", 1e-6},
          {"source_softmax", true},
          {"source_rmsnorm", true}}},
        {"precision", "bf16"},
    };
    if (manifest["profile"] != expected_profile)
        throw std::runtime_error("Wan2.2 artifact_manifest profile is not the official profile");

    const auto& sections = manifest["sections"];
    if (!sections.is_object() || sections.size() != kWan22ManifestSections.size()) {
        throw std::runtime_error(
            "Wan2.2 artifact_manifest must describe exactly eight model-owned sections");
    }
    for (const auto expected : kWan22ManifestSections) {
        if (!sections.contains(std::string(expected))) {
            throw std::runtime_error("Wan2.2 artifact_manifest is missing section: " +
                                     std::string(expected));
        }
    }

    for (const auto section_view : kWan22ManifestSections) {
        const std::string section_name(section_view);
        const auto& entry = sections[section_name];
        const bool is_plan = section_name.size() >= 5 &&
                             section_name.compare(section_name.size() - 5, 5, "_plan") == 0;
        const bool exact_keys =
            has_explicit_sizes
                ? (is_plan ? json_object_has_exact_keys(
                                 entry, {"sha256", "size", "source_sha256", "source_inputs"})
                           : json_object_has_exact_keys(entry, {"sha256", "size"}))
                : (is_plan ? json_object_has_exact_keys(
                                 entry, {"sha256", "source_sha256", "source_inputs"})
                           : json_object_has_exact_keys(entry, {"sha256"}));
        if (!exact_keys || !is_lowercase_sha256(entry["sha256"])) {
            throw std::runtime_error("Wan2.2 artifact manifest entry is malformed for " +
                                     section_name);
        }

        const std::uint64_t header_size = section_info.at(section_name)->size;
        if (header_size == 0) {
            throw std::runtime_error("Wan2.2 artifact size must be positive for " + section_name);
        }
        if (has_explicit_sizes) {
            const std::uint64_t expected_size = require_nonzero_size(entry["size"], section_name);
            if (header_size != expected_size) {
                throw std::runtime_error("Wan2.2 artifact size mismatch for " + section_name);
            }
        }
        if (is_plan) {
            if (!is_lowercase_sha256(entry["source_sha256"]) ||
                !entry["source_inputs"].is_array() || entry["source_inputs"].empty()) {
                throw std::runtime_error("Wan2.2 source provenance is malformed for " +
                                         section_name);
            }
            for (const auto& source : entry["source_inputs"]) {
                if (!json_object_has_exact_keys(source, {"name", "sha256"}) ||
                    !source["name"].is_string() || source["name"].get<std::string>().empty() ||
                    !is_lowercase_sha256(source["sha256"])) {
                    throw std::runtime_error("Wan2.2 source input is malformed for " +
                                             section_name);
                }
            }
            const nlohmann::json source_document = {
                {"family", "wan2_2_ti2v"},
                {"component", section_name},
                {"profile", manifest["profile"]},
                {"inputs", entry["source_inputs"]},
            };
            const std::string canonical_source = source_document.dump();
            Sha256 source_digest;
            source_digest.update(canonical_source.data(), canonical_source.size());
            if (source_digest.hex_digest() != entry["source_sha256"].get<std::string>()) {
                throw std::runtime_error("Wan2.2 source identity mismatch for " + section_name);
            }
        }

        // This streams lazy plans through a bounded buffer. It authenticates
        // every byte before any model/CUDA plugin DSO is opened, while keeping
        // the plans absent from BundleFile::sections until their actual stage.
        if (hash_bundle_section(reader, section_name) != entry["sha256"].get<std::string>()) {
            throw std::runtime_error("Wan2.2 artifact SHA256 mismatch for " + section_name);
        }
    }
}

void validate_preload_provenance(BundleSectionReader& reader, const std::string& config_text,
                                 std::size_t materialized_config_size) {
    // Keep legacy bundle parsing behavior unchanged. Strict JSON parsing is
    // part of the Wan2.2 provenance contract only.
    if (extract_json_string(config_text, "runtime_strategy", "") != kWan22RuntimeStrategy)
        return;
    nlohmann::json config;
    try {
        config = nlohmann::json::parse(config_text);
    } catch (const nlohmann::json::exception& error) {
        throw std::runtime_error(std::string("Invalid bundle config.json: ") + error.what());
    }
    validate_wan22_artifact_provenance(reader, config, materialized_config_size);
}

} // namespace

PipelineBundleMaterialization materialize_pipeline_bundle(BundleSectionReader& reader) {
    const auto& info = reader.info();
    const auto config_info =
        std::find_if(info.sections.begin(), info.sections.end(),
                     [](const BundleSectionInfo& entry) { return entry.name == "config.json"; });

    // Preserve compatibility with legacy bundles that did not carry a
    // config section (or carried an empty one). PipelineFactory will retain
    // its existing manifest-default strategy resolution for those bundles.
    if (config_info == info.sections.end() || config_info->size == 0) {
        return PipelineBundleMaterialization{reader.read_all(), {}};
    }

    auto config_data = reader.read("config.json");
    std::string config_text(config_data.begin(), config_data.end());
    // This preflight runs before runtime strategy lookup, and therefore before
    // any model/CUDA plugin dlopen or TensorRT plan deserialization.
    validate_preload_provenance(reader, config_text, config_data.size());
    const std::string policy_text = extract_json_object_text(config_text, "bundle_loading");
    if (policy_text.empty()) {
        return PipelineBundleMaterialization{reader.read_all(), std::move(config_text)};
    }

    const std::string mode = extract_json_string(policy_text, "mode", "");
    if (mode != "staged") {
        throw std::runtime_error("Unsupported bundle_loading mode: " + mode);
    }
    const auto eager = extract_json_string_array(policy_text, "eager_sections");
    const auto lazy = extract_json_string_array(policy_text, "lazy_sections");
    if (eager.empty() || lazy.empty()) {
        throw std::runtime_error(
            "Staged bundle_loading requires non-empty eager_sections and lazy_sections");
    }

    std::unordered_set<std::string> declared;
    declared.reserve(eager.size() + lazy.size());
    std::unordered_set<std::string> eager_set;
    eager_set.reserve(eager.size());
    for (const auto& name : eager) {
        if (name.empty() || !declared.insert(name).second) {
            throw std::runtime_error("Duplicate or empty staged bundle section: " + name);
        }
        eager_set.insert(name);
    }
    for (const auto& name : lazy) {
        if (name.empty() || !declared.insert(name).second) {
            throw std::runtime_error("Duplicate or empty staged bundle section: " + name);
        }
    }
    if (eager_set.find("config.json") == eager_set.end()) {
        throw std::runtime_error("Staged bundle_loading must eagerly materialize config.json");
    }

    std::unordered_set<std::string> available;
    available.reserve(info.sections.size());
    for (const auto& section : info.sections) {
        if (!available.insert(section.name).second) {
            throw std::runtime_error("Duplicate section in staged bundle header: " + section.name);
        }
    }
    if (available != declared) {
        throw std::runtime_error(
            "Staged bundle_loading eager/lazy sections must partition the bundle header exactly");
    }

    BundleFile bundle;
    bundle.info = info;
    bundle.sections.reserve(eager.size());
    for (const auto& section_info : info.sections) {
        if (eager_set.find(section_info.name) == eager_set.end())
            continue;
        BundleSection section;
        section.name = section_info.name;
        if (section.name == "config.json") {
            section.data = std::move(config_data);
        } else {
            section.data = reader.read(section.name);
        }
        bundle.sections.push_back(std::move(section));
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
    LoadOptions optimized_options;
    optimized_options.hf_python = hf_python;
    optimized_options.runtime_cache_path = runtime_cache_path;
    optimized_options.cuda_graphs = cuda_graphs;
    const BundleInfo header = ReadBundleHeader(bundle_path);
    if (auto optimized_runtime_pipeline =
            try_make_optimized_runtime_pipeline(bundle_path, header, optimized_options)) {
        return optimized_runtime_pipeline;
    }

    auto bundle_reader = std::make_shared<BundleSectionReader>(bundle_path);
    auto materialized = detail::materialize_pipeline_bundle(*bundle_reader);
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
                        resolved ? &*resolved : nullptr,
                        bundle_reader};
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
    auto bundle_reader = std::make_shared<BundleSectionReader>(bundle_path);
    auto materialized = detail::materialize_pipeline_bundle(*bundle_reader);
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
                        resolved ? &*resolved : nullptr,
                        bundle_reader};
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

    auto bundle_reader = std::make_shared<BundleSectionReader>(bundle_path);
    auto materialized = detail::materialize_pipeline_bundle(*bundle_reader);
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
                        resolved ? &*resolved : nullptr,
                        bundle_reader};
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

#pragma once

#include "bundle/bundle_format.h"
#include "trtmc/config/config_bundle.h"

#include <optional>
#include <string>
#include <vector>

namespace trtmc::deployment {

inline constexpr const char* kDeploymentManifestSection = "deployment_manifest.json";

struct Artifact {
    std::string name;
    std::string kind{"bundle_section"};
    std::string section;
    std::string section_prefix;
};

struct Variant {
    std::string id;
    std::string scope;
    std::string provider;
    std::string runtime_strategy;
    std::string compatibility_json;
    std::string performance_json;
    bool fallback{false};
    std::vector<Artifact> artifacts;
};

struct Manifest {
    int schema_version{1};
    std::string default_variant;
    std::string selected_variant;
    std::string target_platform;
    std::string target_objective;
    std::vector<Variant> variants;
};

std::optional<Manifest> read_manifest(const BundleFile& bundle);

const Variant* find_variant(const Manifest& manifest, const std::string& variant_id);
const Variant* choose_variant(const Manifest& manifest,
                              const config::ConfigBundle* runtime_config);
const Variant* find_fallback_variant(const Manifest& manifest);

// Return a copy of bundle where bundle_section artifacts for variant are
// exposed under their canonical artifact names.  This lets existing native
// runtime plugins keep looking up "engine_plan" and "kernel_manifest.json".
BundleFile bundle_with_variant_artifacts(const BundleFile& bundle, const Variant& variant);

std::string inspect_text(const Manifest& manifest);

} // namespace trtmc::deployment

#include "runtime/deployment/deployment_manifest.h"

#include "bundle/bundle_view.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace trtmc::deployment {

namespace {

std::string json_string_or_empty(const nlohmann::json& obj, const char* key) {
    auto it = obj.find(key);
    if (it == obj.end() || !it->is_string())
        return {};
    return it->get<std::string>();
}

Artifact parse_artifact(const nlohmann::json& obj) {
    Artifact artifact;
    artifact.name = json_string_or_empty(obj, "name");
    artifact.kind = json_string_or_empty(obj, "kind");
    if (artifact.kind.empty())
        artifact.kind = "bundle_section";
    artifact.section = json_string_or_empty(obj, "section");
    artifact.section_prefix = json_string_or_empty(obj, "section_prefix");
    return artifact;
}

Variant parse_variant(const nlohmann::json& obj) {
    Variant variant;
    variant.id = json_string_or_empty(obj, "id");
    variant.scope = json_string_or_empty(obj, "scope");
    variant.provider = json_string_or_empty(obj, "provider");
    variant.runtime_strategy = json_string_or_empty(obj, "runtime_strategy");
    if (auto it = obj.find("compatibility"); it != obj.end() && it->is_object())
        variant.compatibility_json = it->dump();
    if (auto it = obj.find("performance"); it != obj.end() && it->is_object())
        variant.performance_json = it->dump();
    if (auto it = obj.find("fallback"); it != obj.end() && it->is_boolean())
        variant.fallback = it->get<bool>();
    if (auto it = obj.find("artifacts"); it != obj.end() && it->is_array()) {
        for (const auto& artifact_json : *it)
            if (artifact_json.is_object())
                variant.artifacts.push_back(parse_artifact(artifact_json));
    }
    return variant;
}

const BundleSection* find_bundle_section(const BundleFile& bundle, const std::string& name) {
    for (const auto& section : bundle.sections) {
        if (section.name == name)
            return &section;
    }
    return nullptr;
}

void remove_canonical_section(BundleFile& bundle, const std::string& name) {
    bundle.sections.erase(
        std::remove_if(bundle.sections.begin(), bundle.sections.end(),
                       [&](const BundleSection& section) { return section.name == name; }),
        bundle.sections.end());
}

} // namespace

std::optional<Manifest> read_manifest(const BundleFile& bundle) {
    const auto* section = find_section(bundle, kDeploymentManifestSection);
    if (section == nullptr || section->empty())
        return std::nullopt;

    const std::string text(section->begin(), section->end());
    const nlohmann::json root = nlohmann::json::parse(text);

    Manifest manifest;
    if (auto it = root.find("schema_version"); it != root.end() && it->is_number_integer())
        manifest.schema_version = it->get<int>();
    manifest.default_variant = json_string_or_empty(root, "default_variant");
    manifest.selected_variant = json_string_or_empty(root, "selected_variant");
    if (auto target = root.find("target"); target != root.end() && target->is_object()) {
        manifest.target_platform = json_string_or_empty(*target, "platform");
        manifest.target_objective = json_string_or_empty(*target, "objective");
    }
    if (auto it = root.find("variants"); it != root.end() && it->is_array()) {
        for (const auto& variant_json : *it)
            if (variant_json.is_object())
                manifest.variants.push_back(parse_variant(variant_json));
    }
    return manifest;
}

const Variant* find_variant(const Manifest& manifest, const std::string& variant_id) {
    if (variant_id.empty())
        return nullptr;
    for (const auto& variant : manifest.variants) {
        if (variant.id == variant_id)
            return &variant;
    }
    return nullptr;
}

const Variant* find_fallback_variant(const Manifest& manifest) {
    if (const auto* by_default = find_variant(manifest, manifest.default_variant))
        return by_default;
    for (const auto& variant : manifest.variants) {
        if (variant.fallback)
            return &variant;
    }
    return nullptr;
}

const Variant* choose_variant(const Manifest& manifest,
                              const config::ConfigBundle* runtime_config) {
    if (runtime_config != nullptr) {
        try {
            const bool force_fallback = runtime_config->get<bool>("deployment", "force_fallback");
            if (force_fallback) {
                if (const auto* fallback = find_fallback_variant(manifest))
                    return fallback;
            }
            const std::string requested = runtime_config->get<std::string>("deployment", "variant");
            if (!requested.empty()) {
                if (const auto* variant = find_variant(manifest, requested))
                    return variant;
                throw std::runtime_error("Requested deployment variant not found: " + requested);
            }
        } catch (const std::out_of_range&) {
        }
    }
    if (const auto* selected = find_variant(manifest, manifest.selected_variant))
        return selected;
    if (const auto* fallback = find_fallback_variant(manifest))
        return fallback;
    return manifest.variants.empty() ? nullptr : &manifest.variants.front();
}

BundleFile bundle_with_variant_artifacts(const BundleFile& bundle, const Variant& variant) {
    BundleFile out = bundle;
    for (const auto& artifact : variant.artifacts) {
        if (artifact.kind != "bundle_section")
            continue;
        const std::string source_name = artifact.section.empty() ? artifact.name : artifact.section;
        if (source_name.empty() || artifact.name.empty())
            continue;
        const auto* source = find_bundle_section(bundle, source_name);
        if (source == nullptr)
            throw std::runtime_error("Deployment artifact section not found: " + source_name);
        remove_canonical_section(out, artifact.name);
        BundleSection remapped;
        remapped.name = artifact.name;
        remapped.data = source->data;
        out.sections.push_back(std::move(remapped));
    }
    if (!variant.runtime_strategy.empty())
        out.info.runtime_strategy = variant.runtime_strategy;
    return out;
}

std::string inspect_text(const Manifest& manifest) {
    std::ostringstream out;
    out << "Deployment:\n";
    out << "  target: " << (manifest.target_platform.empty() ? "<unspecified>"
                                                              : manifest.target_platform)
        << '\n';
    if (!manifest.target_objective.empty())
        out << "  objective: " << manifest.target_objective << '\n';
    out << "  selected_variant: " << manifest.selected_variant << '\n';
    out << "  default_variant: " << manifest.default_variant << '\n';
    out << "  variants:\n";
    for (const auto& variant : manifest.variants) {
        out << "    - id: " << variant.id << '\n';
        out << "      scope: " << variant.scope << '\n';
        out << "      provider: " << variant.provider << '\n';
        if (!variant.runtime_strategy.empty())
            out << "      runtime_strategy: " << variant.runtime_strategy << '\n';
        if (!variant.compatibility_json.empty())
            out << "      compatibility: " << variant.compatibility_json << '\n';
        if (!variant.performance_json.empty())
            out << "      performance: " << variant.performance_json << '\n';
        out << "      fallback: " << (variant.fallback ? "true" : "false") << '\n';
        if (!variant.artifacts.empty()) {
            out << "      artifacts:\n";
            for (const auto& artifact : variant.artifacts) {
                out << "        - " << artifact.name << " (" << artifact.kind << ")";
                if (!artifact.section.empty())
                    out << " section=" << artifact.section;
                if (!artifact.section_prefix.empty())
                    out << " section_prefix=" << artifact.section_prefix;
                out << '\n';
            }
        }
    }
    return out.str();
}

} // namespace trtmc::deployment

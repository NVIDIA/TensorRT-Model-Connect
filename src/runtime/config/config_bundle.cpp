/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/config/config_bundle.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace trtmc::config {

const char* layer_name(Layer layer) {
    switch (layer) {
    case Layer::SchemaDefault:
        return "schema_default";
    case Layer::BuildTime:
        return "build_time";
    case Layer::BundleDefault:
        return "bundle_default";
    case Layer::PlatformProfile:
        return "platform_profile";
    case Layer::SessionRequest:
        return "session_request";
    }
    return "unknown";
}

namespace {

// Higher value in the Layer enum = higher priority. See schema_registry.h.
bool higher_priority(Layer a, Layer b) {
    return static_cast<std::uint8_t>(a) > static_cast<std::uint8_t>(b);
}

const ConfigField* find_field(const Schema& schema, const std::string& name) {
    for (const auto& field : schema.fields) {
        if (field.name == name)
            return &field;
    }
    return nullptr;
}

// Walk all contributions and fail fast on any authoring or layer error.
void validate_contributions(const std::vector<LayerContribution>& contributions,
                            const SchemaRegistry& registry) {
    for (const auto& contrib : contributions) {
        for (const auto& ns_entry : contrib.values) {
            const std::string& ns = ns_entry.first;
            const Schema* schema = registry.lookup(ns);
            if (schema == nullptr) {
                throw std::invalid_argument(std::string("Layer ") + layer_name(contrib.layer) +
                                            " contributed value for unregistered namespace: " + ns);
            }
            for (const auto& f_entry : ns_entry.second) {
                const std::string& field_name = f_entry.first;
                const ConfigField* field = find_field(*schema, field_name);
                if (field == nullptr) {
                    throw std::invalid_argument(std::string("Layer ") + layer_name(contrib.layer) +
                                                " contributed value for unknown field: " + ns +
                                                "." + field_name);
                }
                if (field->allowed_layers.count(contrib.layer) == 0) {
                    throw std::invalid_argument(std::string("Layer ") + layer_name(contrib.layer) +
                                                " is not permitted to set " + ns + "." +
                                                field_name);
                }
                if (field->validator && !field->validator(f_entry.second)) {
                    throw std::invalid_argument(std::string("Validator rejected value for ") + ns +
                                                "." + field_name + " from layer " +
                                                layer_name(contrib.layer));
                }
            }
        }
    }
}

// For a given (namespace, field), find the highest-priority layer that
// contributed a value. Returns nullptr if nothing contributed.
const std::any* pick_highest_priority(const std::vector<LayerContribution>& contributions,
                                      const std::string& ns, const std::string& field,
                                      Layer& out_source) {
    const std::any* best = nullptr;
    Layer best_layer = Layer::SchemaDefault;
    for (const auto& contrib : contributions) {
        auto ns_it = contrib.values.find(ns);
        if (ns_it == contrib.values.end())
            continue;
        auto f_it = ns_it->second.find(field);
        if (f_it == ns_it->second.end())
            continue;
        if (best == nullptr || higher_priority(contrib.layer, best_layer)) {
            best = &f_it->second;
            best_layer = contrib.layer;
        }
    }
    if (best != nullptr)
        out_source = best_layer;
    return best;
}

} // namespace

ConfigBundle ConfigBundle::build(const std::vector<LayerContribution>& contributions,
                                 const SchemaRegistry& registry) {
    validate_contributions(contributions, registry);

    ConfigBundle bundle;
    for (const std::string& ns : registry.registered_namespaces()) {
        const Schema* schema = registry.lookup(ns);
        for (const auto& field : schema->fields) {
            Layer source = Layer::SchemaDefault;
            const std::any* picked = pick_highest_priority(contributions, ns, field.name, source);
            ResolvedValue rv;
            if (picked != nullptr) {
                rv.value = *picked;
                rv.source = source;
            } else {
                rv.value = field.default_value;
                rv.source = Layer::SchemaDefault;
            }
            bundle.resolved_[ns][field.name] = rv;
        }
    }
    return bundle;
}

const std::any& ConfigBundle::get_any(const std::string& namespace_name,
                                      const std::string& field) const {
    auto ns_it = resolved_.find(namespace_name);
    if (ns_it == resolved_.end())
        throw std::out_of_range("ConfigBundle: unknown namespace: " + namespace_name);
    auto f_it = ns_it->second.find(field);
    if (f_it == ns_it->second.end()) {
        throw std::out_of_range("ConfigBundle: unknown field: " + namespace_name + "." + field);
    }
    return f_it->second.value;
}

Layer ConfigBundle::source_of(const std::string& namespace_name, const std::string& field) const {
    auto ns_it = resolved_.find(namespace_name);
    if (ns_it == resolved_.end())
        throw std::out_of_range("ConfigBundle: unknown namespace: " + namespace_name);
    auto f_it = ns_it->second.find(field);
    if (f_it == ns_it->second.end()) {
        throw std::out_of_range("ConfigBundle: unknown field: " + namespace_name + "." + field);
    }
    return f_it->second.source;
}

} // namespace trtmc::config

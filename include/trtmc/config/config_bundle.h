/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// ConfigBundle: the resolved, immutable, per-session configuration.
//
// Built by merging layer contributions against the currently-registered
// schemas in priority order:
//
//   SessionRequest > PlatformProfile > BundleDefault > BuildTime > SchemaDefault
//
// The merge enforces each field's allowed_layers — a layer that attempts to
// set a field outside its allowlist fails fast with namespace.field in the
// message. Validators run after layer selection on the resolved value.
//
// Plugins never read layers directly. They receive a ConfigBundle via the
// PipelineContext and ask for their own namespace:
//
//   auto kv_budget = bundle.get<int32_t>("triattention", "kv_budget");
//
// Provenance (which layer contributed each resolved value) is preserved so
// effective_config.json can record it.

#include "trtmc/config/schema_registry.h"

#include <any>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace trtmc::config {

// One layer's contribution: a nested map namespace -> field -> value.
// Constructed by the CLI/profile loader; fed into ConfigBundle::build.
struct LayerContribution {
    Layer layer;
    std::unordered_map<std::string, std::unordered_map<std::string, std::any>> values;
};

// A resolved value, tagged with the layer that produced it. The SchemaDefault
// layer is used when no other layer contributed a value for the field.
struct ResolvedValue {
    std::any value;
    Layer source;
};

// Immutable resolved configuration for one session.
class ConfigBundle {
  public:
    using NamespaceMap = std::unordered_map<std::string, ResolvedValue>;
    using ResolvedMap = std::unordered_map<std::string, NamespaceMap>;

    ConfigBundle() = default;

    // Merge contributions against the given registry (default: singleton).
    // Throws std::invalid_argument on allowlist violations, unknown
    // namespace/field contributions, or validator failures.
    static ConfigBundle build(const std::vector<LayerContribution>& contributions,
                              const SchemaRegistry& registry = SchemaRegistry::instance());

    // Typed access. Throws std::out_of_range if namespace/field unknown,
    // std::bad_any_cast if the requested type does not match the stored type.
    template <typename T>
    T get(const std::string& namespace_name, const std::string& field) const {
        const std::any& val = get_any(namespace_name, field);
        try {
            return std::any_cast<T>(val);
        } catch (const std::bad_any_cast&) {
            throw std::bad_any_cast();
        }
    }

    // Untyped access — used by effective_config.json serialization.
    const std::any& get_any(const std::string& namespace_name, const std::string& field) const;
    Layer source_of(const std::string& namespace_name, const std::string& field) const;

    // All resolved values. Key order is unspecified; serialization code
    // sorts as needed for stable output.
    const ResolvedMap& all() const { return resolved_; }

  private:
    ResolvedMap resolved_;
};

// Human-readable name for a layer. Used in error messages and
// effective_config.json provenance.
const char* layer_name(Layer layer);

} // namespace trtmc::config

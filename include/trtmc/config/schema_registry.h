/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Config schema registry: singleton that maps namespace strings to schema
// definitions. Built-in schemas register through the generated CMake manifest
// source. Mirrors the PipelineRegistry pattern so parallel agents can add
// features in isolation without touching any shared dispatcher, CLI parser, or
// registry-of-registries.
//
// Design contract (see website/docs/context/config-registry-status.md):
//   - Registration: catalogs *schemas* (field metadata + defaults). No values.
//   - ConfigBundle supports the general precedence session > platform >
//     bundle defaults > build time > schema defaults. The current native
//     pipeline factory passes the materialized config.json section to the
//     resolver. A top-level "defaults" object in that section can contribute
//     BundleDefault, and LoadOptions can contribute SessionRequest. The
//     factory does not pass BundleInfo.defaults from the binary header to the
//     resolver, nor does it inject BuildTime or PlatformProfile.
//   - LoadOptions.config_path and CLI --config/--set all contribute to the
//     current SessionRequest layer before the bundle is attached to
//     PipelineContext.
//   - Native plugin create(ctx): queries ctx.runtime_config for its own
//     namespace only.
//   - A higher-priority layer overrides a lower-priority value only when the
//     field's allowed_layers contract permits that source.
//   - Optimized-runtime bundles bypass native ConfigBundle/plugin dispatch.
//     Their embedded implementation receives LoadOptions through its private
//     factory request and owns support or rejection of those options.

#include <any>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc::config {

// Which config layers are permitted to set a field. Schema declarations use
// this to express intent like "kv_budget may come from session, platform, or
// bundle defaults." Enforcement
// happens during layer merge: a layer attempting to set a field outside its
// allowlist is a fail-fast error with the exact namespace/field name.
enum class Layer : std::uint8_t {
    SchemaDefault = 0,   // Hard-coded fallback baked into the schema.
    BuildTime = 1,       // General resolver layer; not factory-wired today.
    BundleDefault = 2,   // Factory reads config.json's top-level defaults object.
    PlatformProfile = 3, // General resolver layer; not factory-wired today.
    SessionRequest = 4,  // Factory config_path / CLI --config or --set.
};

// Metadata for one field inside a namespaced schema. The validator is
// optional; if present it returns true for valid values and false otherwise.
// Type is a string tag ("int32", "int64", "float", "bool", "string",
// "list<string>") for diagnostic messages and cross-language schema
// comparison; runtime type checking uses std::any's held type directly.
struct ConfigField {
    std::string name;
    std::string type;
    std::any default_value;
    std::set<Layer> allowed_layers;
    std::function<bool(const std::any&)> validator;
};

// A schema bundles a namespace string with its field list. Registering a
// schema more than once for the same namespace is a fail-fast error — that
// would indicate two agents tried to claim the same namespace.
struct Schema {
    std::string namespace_name;
    std::vector<ConfigField> fields;
};

// Singleton registry. Lookups by namespace are O(1). The registry itself
// holds no values — it is a metadata catalog. Value resolution is handled by
// the implemented ConfigBundle::build path in config_bundle.cpp; the pipeline
// factory resolves that bundle and attaches it to PipelineContext.
class SchemaRegistry {
  public:
    static SchemaRegistry& instance();

    // Register a schema for a namespace. Built-in schemas arrive through
    // generated manifest registrar calls. Throws on duplicate namespace, empty
    // namespace, or empty field list.
    void register_schema(Schema schema);

    // Look up the schema for a namespace. Returns nullptr if the namespace
    // was never registered.
    const Schema* lookup(const std::string& namespace_name) const;

    // All registered namespace names, sorted. For diagnostics and the
    // scalability test.
    std::vector<std::string> registered_namespaces() const;

    // Clear the registry. Test-only: tests that register throwaway schemas
    // call this between cases. Not part of the production lifecycle.
    void clear_for_testing();

  private:
    SchemaRegistry() = default;
    std::unordered_map<std::string, Schema> schemas_;
};

// Legacy helper for ad hoc tests or local extensions that still rely on
// file-scope registration. Built-in schemas use manifest registration below.
struct SchemaRegistrar {
    SchemaRegistrar(Schema schema) {
        SchemaRegistry::instance().register_schema(std::move(schema));
    }
};

// Legacy macro: register a schema for a namespace at process startup. Usage
// (at file scope, typically in a generated header or a hand-written schema
// source):
//
//   REGISTER_CONFIG_SCHEMA(::trtmc::config::Schema{
//       "triattention",
//       {
//           {"kv_budget", "int32", std::any{int32_t{6144}},
//            {Layer::BundleDefault, Layer::PlatformProfile,
//             Layer::SessionRequest}, nullptr},
//           // ... more fields
//       }
//   });
//
// The schema object is a temporary; the registrar copies/moves it into the
// registry. Duplicate namespaces throw at process start, which surfaces a
// registration collision between parallel agents immediately.
#define REGISTER_CONFIG_SCHEMA(schema_literal)                                                     \
    namespace {                                                                                    \
    static ::trtmc::config::SchemaRegistrar                                                        \
        g_config_schema_registrar_##__COUNTER__(schema_literal);                                   \
    }

// Define a schema factory registration function consumed by the generated
// manifest source. The function name must match cmake/trtmc_config_schemas.cmake.
#define REGISTER_CONFIG_SCHEMA_FACTORY_WITH_MANIFEST(RegisterFunction, FactoryFn)                  \
    void RegisterFunction(::trtmc::config::SchemaRegistry& registry) {                             \
        registry.register_schema(FactoryFn());                                                     \
    }

} // namespace trtmc::config

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-CFG-REG-01
// Architecture:   ARCH-CFG-001
// Unit Design:    UD-CFG-REG-01
// Intent:         SchemaRegistry and ConfigBundle — registration rules, layer
//                 priority merge, allowlist enforcement, provenance tracking.
// Preconditions:  None (no GPU, no TRT, no filesystem).
// Postconditions: Registry accepts valid schemas, rejects malformed ones;
//                 bundle merges layers by priority; provenance is preserved.
// =============================================================================

#include "trtmc/config/config_bundle.h"
#include "trtmc/config/schema_registry.h"

#include <any>
#include <cstdint>
#include <exception>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++g_failures;
    }
}

template <typename Fn>
void expect_throws(Fn fn, const char* substring_match, const char* test_name) {
    try {
        fn();
        std::cerr << "FAIL: " << test_name << " (no exception thrown)\n";
        ++g_failures;
    } catch (const std::exception& e) {
        if (substring_match == nullptr ||
            std::string(e.what()).find(substring_match) != std::string::npos)
            return;
        std::cerr << "FAIL: " << test_name << " (message missing '" << substring_match
                  << "'): " << e.what() << '\n';
        ++g_failures;
    }
}

using trtmc::config::ConfigBundle;
using trtmc::config::ConfigField;
using trtmc::config::Layer;
using trtmc::config::LayerContribution;
using trtmc::config::Schema;
using trtmc::config::SchemaRegistry;

// Convenience factory for a valid field usable by most tests.
ConfigField int_field(const std::string& name, std::int32_t default_value, std::set<Layer> layers) {
    return ConfigField{name, "int32", std::any{default_value}, std::move(layers), nullptr};
}

ConfigField bool_field(const std::string& name, bool default_value, std::set<Layer> layers) {
    return ConfigField{name, "bool", std::any{default_value}, std::move(layers), nullptr};
}

// Install one fresh schema in an empty registry. Returns a SchemaRegistry
// reference to the singleton (already cleared).
SchemaRegistry& fresh_registry_with(const std::string& ns, std::vector<ConfigField> fields) {
    SchemaRegistry& reg = SchemaRegistry::instance();
    reg.clear_for_testing();
    reg.register_schema(Schema{ns, std::move(fields)});
    return reg;
}

// --- Registry: registration rules ---------------------------------------------

void test_register_and_lookup() {
    auto& reg = fresh_registry_with(
        "ns_a", {int_field("budget", 6144, {Layer::SessionRequest, Layer::BundleDefault})});
    const Schema* schema = reg.lookup("ns_a");
    check(schema != nullptr, "register_and_lookup: schema present");
    check(schema && schema->fields.size() == 1, "register_and_lookup: field count");
    check(reg.lookup("ns_b") == nullptr, "register_and_lookup: missing returns nullptr");
}

void test_duplicate_namespace_throws() {
    SchemaRegistry& reg = SchemaRegistry::instance();
    reg.clear_for_testing();
    reg.register_schema(Schema{"dup", {int_field("f", 0, {Layer::SessionRequest})}});
    expect_throws(
        [&] { reg.register_schema(Schema{"dup", {int_field("f", 0, {Layer::SessionRequest})}}); },
        "Duplicate", "duplicate_namespace_throws");
}

void test_empty_namespace_throws() {
    SchemaRegistry& reg = SchemaRegistry::instance();
    reg.clear_for_testing();
    expect_throws(
        [&] { reg.register_schema(Schema{"", {int_field("f", 0, {Layer::SessionRequest})}}); },
        "empty namespace", "empty_namespace_throws");
}

void test_empty_fields_throws() {
    SchemaRegistry& reg = SchemaRegistry::instance();
    reg.clear_for_testing();
    expect_throws([&] { reg.register_schema(Schema{"ns", {}}); }, "no fields",
                  "empty_fields_throws");
}

void test_schema_default_in_allowlist_throws() {
    SchemaRegistry& reg = SchemaRegistry::instance();
    reg.clear_for_testing();
    expect_throws(
        [&] {
            reg.register_schema(
                Schema{"ns", {int_field("f", 0, {Layer::SchemaDefault, Layer::SessionRequest})}});
        },
        "SchemaDefault", "schema_default_in_allowlist_throws");
}

void test_empty_allowlist_throws() {
    SchemaRegistry& reg = SchemaRegistry::instance();
    reg.clear_for_testing();
    expect_throws([&] { reg.register_schema(Schema{"ns", {int_field("f", 0, {})}}); },
                  "empty allowed_layers", "empty_allowlist_throws");
}

void test_registered_namespaces_sorted() {
    SchemaRegistry& reg = SchemaRegistry::instance();
    reg.clear_for_testing();
    reg.register_schema(Schema{"zeta", {int_field("f", 0, {Layer::SessionRequest})}});
    reg.register_schema(Schema{"alpha", {int_field("f", 0, {Layer::SessionRequest})}});
    reg.register_schema(Schema{"mike", {int_field("f", 0, {Layer::SessionRequest})}});
    auto names = reg.registered_namespaces();
    check(names.size() == 3, "registered_namespaces: count");
    check(names[0] == "alpha" && names[1] == "mike" && names[2] == "zeta",
          "registered_namespaces: sorted");
}

// --- Bundle: merge ------------------------------------------------------------

LayerContribution layer(Layer which, const std::string& ns, const std::string& field,
                        std::any value) {
    LayerContribution c;
    c.layer = which;
    c.values[ns][field] = std::move(value);
    return c;
}

void test_merge_session_beats_platform() {
    fresh_registry_with("ns",
                        {int_field("k", 100, {Layer::SessionRequest, Layer::PlatformProfile})});
    auto bundle = ConfigBundle::build({
        layer(Layer::PlatformProfile, "ns", "k", std::int32_t{200}),
        layer(Layer::SessionRequest, "ns", "k", std::int32_t{300}),
    });
    check(bundle.get<std::int32_t>("ns", "k") == 300, "merge_session_beats_platform: value");
    check(bundle.source_of("ns", "k") == Layer::SessionRequest,
          "merge_session_beats_platform: source");
}

void test_merge_platform_beats_bundle() {
    fresh_registry_with("ns",
                        {int_field("k", 100, {Layer::BundleDefault, Layer::PlatformProfile})});
    auto bundle = ConfigBundle::build({
        layer(Layer::BundleDefault, "ns", "k", std::int32_t{200}),
        layer(Layer::PlatformProfile, "ns", "k", std::int32_t{250}),
    });
    check(bundle.get<std::int32_t>("ns", "k") == 250, "merge_platform_beats_bundle: value");
    check(bundle.source_of("ns", "k") == Layer::PlatformProfile,
          "merge_platform_beats_bundle: source");
}

void test_merge_bundle_beats_build() {
    fresh_registry_with("ns", {int_field("k", 100, {Layer::BuildTime, Layer::BundleDefault})});
    auto bundle = ConfigBundle::build({
        layer(Layer::BuildTime, "ns", "k", std::int32_t{200}),
        layer(Layer::BundleDefault, "ns", "k", std::int32_t{250}),
    });
    check(bundle.get<std::int32_t>("ns", "k") == 250, "merge_bundle_beats_build: value");
    check(bundle.source_of("ns", "k") == Layer::BundleDefault, "merge_bundle_beats_build: source");
}

void test_merge_fallback_to_schema_default() {
    fresh_registry_with("ns", {int_field("k", 100, {Layer::SessionRequest})});
    auto bundle = ConfigBundle::build({});
    check(bundle.get<std::int32_t>("ns", "k") == 100, "fallback_to_schema_default: value");
    check(bundle.source_of("ns", "k") == Layer::SchemaDefault,
          "fallback_to_schema_default: source");
}

void test_merge_allowlist_violation_throws() {
    fresh_registry_with("ns", {int_field("k", 100, {Layer::BundleDefault})}); // session NOT allowed
    expect_throws(
        [&] {
            ConfigBundle::build({
                layer(Layer::SessionRequest, "ns", "k", std::int32_t{300}),
            });
        },
        "not permitted", "allowlist_violation: message mentions permission");
}

void test_merge_unknown_namespace_throws() {
    fresh_registry_with("ns", {int_field("k", 0, {Layer::SessionRequest})});
    expect_throws(
        [&] {
            ConfigBundle::build({
                layer(Layer::SessionRequest, "other_ns", "k", std::int32_t{1}),
            });
        },
        "unregistered namespace", "unknown_namespace_throws");
}

void test_merge_unknown_field_throws() {
    fresh_registry_with("ns", {int_field("k", 0, {Layer::SessionRequest})});
    expect_throws(
        [&] {
            ConfigBundle::build({
                layer(Layer::SessionRequest, "ns", "other_field", std::int32_t{1}),
            });
        },
        "unknown field", "unknown_field_throws");
}

void test_merge_validator_rejection_throws() {
    SchemaRegistry& reg = SchemaRegistry::instance();
    reg.clear_for_testing();
    ConfigField field{
        "k", "int32", std::any{std::int32_t{0}}, {Layer::SessionRequest}, [](const std::any& v) {
            try {
                return std::any_cast<std::int32_t>(v) > 0;
            } catch (...) {
                return false;
            }
        }};
    reg.register_schema(Schema{"ns", {field}});
    expect_throws(
        [&] {
            ConfigBundle::build({
                layer(Layer::SessionRequest, "ns", "k", std::int32_t{-1}),
            });
        },
        "Validator rejected", "validator_rejection_throws");
}

// --- Bundle: typed access -----------------------------------------------------

void test_bundle_get_typed_multiple_kinds() {
    SchemaRegistry& reg = SchemaRegistry::instance();
    reg.clear_for_testing();
    reg.register_schema(Schema{"ns",
                               {
                                   int_field("budget", 6144, {Layer::SessionRequest}),
                                   bool_field("protect", true, {Layer::SessionRequest}),
                               }});
    auto bundle = ConfigBundle::build({});
    check(bundle.get<std::int32_t>("ns", "budget") == 6144, "typed_access: int default");
    check(bundle.get<bool>("ns", "protect") == true, "typed_access: bool default");
}

void test_bundle_type_mismatch_throws() {
    fresh_registry_with("ns", {int_field("k", 100, {Layer::SessionRequest})});
    auto bundle = ConfigBundle::build({});
    expect_throws([&] { (void)bundle.get<bool>("ns", "k"); }, nullptr, "type_mismatch_throws");
}

void test_bundle_get_any_unknown_namespace_throws() {
    fresh_registry_with("ns", {int_field("k", 100, {Layer::SessionRequest})});
    auto bundle = ConfigBundle::build({});
    expect_throws([&] { (void)bundle.get_any("missing", "k"); }, "unknown namespace",
                  "get_any_unknown_namespace_throws");
}

void test_bundle_get_any_unknown_field_throws() {
    fresh_registry_with("ns", {int_field("k", 100, {Layer::SessionRequest})});
    auto bundle = ConfigBundle::build({});
    expect_throws([&] { (void)bundle.get_any("ns", "missing"); }, "unknown field",
                  "get_any_unknown_field_throws");
}

// --- Provenance ---------------------------------------------------------------

void test_bundle_all_includes_every_field() {
    SchemaRegistry& reg = SchemaRegistry::instance();
    reg.clear_for_testing();
    reg.register_schema(Schema{"ns_a",
                               {
                                   int_field("f1", 1, {Layer::SessionRequest}),
                                   int_field("f2", 2, {Layer::SessionRequest}),
                               }});
    reg.register_schema(Schema{"ns_b",
                               {
                                   int_field("f3", 3, {Layer::SessionRequest}),
                               }});
    auto bundle = ConfigBundle::build({
        layer(Layer::SessionRequest, "ns_a", "f2", std::int32_t{20}),
    });
    const auto& all = bundle.all();
    check(all.size() == 2, "all: two namespaces");
    check(all.at("ns_a").size() == 2, "all: ns_a has both fields");
    check(all.at("ns_b").size() == 1, "all: ns_b has one field");
    check(all.at("ns_a").at("f1").source == Layer::SchemaDefault,
          "all: ns_a.f1 came from schema default");
    check(all.at("ns_a").at("f2").source == Layer::SessionRequest,
          "all: ns_a.f2 came from session");
    check(all.at("ns_b").at("f3").source == Layer::SchemaDefault,
          "all: ns_b.f3 came from schema default");
}

// --- layer_name diagnostic ----------------------------------------------------

void test_layer_name_stable_strings() {
    using trtmc::config::layer_name;
    check(std::string(layer_name(Layer::SchemaDefault)) == "schema_default", "layer_name: schema");
    check(std::string(layer_name(Layer::BuildTime)) == "build_time", "layer_name: build");
    check(std::string(layer_name(Layer::BundleDefault)) == "bundle_default", "layer_name: bundle");
    check(std::string(layer_name(Layer::PlatformProfile)) == "platform_profile",
          "layer_name: platform");
    check(std::string(layer_name(Layer::SessionRequest)) == "session_request",
          "layer_name: session");
}

} // namespace

int main() {
    test_register_and_lookup();
    test_duplicate_namespace_throws();
    test_empty_namespace_throws();
    test_empty_fields_throws();
    test_schema_default_in_allowlist_throws();
    test_empty_allowlist_throws();
    test_registered_namespaces_sorted();

    test_merge_session_beats_platform();
    test_merge_platform_beats_bundle();
    test_merge_bundle_beats_build();
    test_merge_fallback_to_schema_default();
    test_merge_allowlist_violation_throws();
    test_merge_unknown_namespace_throws();
    test_merge_unknown_field_throws();
    test_merge_validator_rejection_throws();

    test_bundle_get_typed_multiple_kinds();
    test_bundle_type_mismatch_throws();
    test_bundle_get_any_unknown_namespace_throws();
    test_bundle_get_any_unknown_field_throws();

    test_bundle_all_includes_every_field();
    test_layer_name_stable_strings();

    // Leave the registry clean so test order/discovery doesn't spill.
    SchemaRegistry::instance().clear_for_testing();

    if (g_failures != 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    return 0;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-CFG-CLI-CPP-01
// Architecture:   ARCH-CFG-001
// Unit Design:    UD-CFG-CLI-01
// Intent:         C++ mirror of --set / --config parsing, type coercion, and
//                 merge-into-LayerContribution. Guards same semantics as the
//                 Python side (tests/builder/test_config_cli_support.py).
// Preconditions:  None (no GPU, no TRT, no network).
// Postconditions: Parser rejects malformed tokens; coercion follows
//                 type_tag; merge produces a single SESSION_REQUEST
//                 contribution with --set winning over --config; JSON
//                 profile parser accepts the scoped shape.
// =============================================================================

#include "runtime/registry/runtime_config_resolution.h"
#include "test_helpers.h"
#include "trtmc/config/cli_support.h"
#include "trtmc/config/config_bundle.h"
#include "trtmc/config/schema_registry.h"

#include <any>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
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
void expect_throws(Fn fn, const char* substring, const char* test_name) {
    try {
        fn();
        std::cerr << "FAIL: " << test_name << " (no exception thrown)\n";
        ++g_failures;
    } catch (const std::exception& e) {
        if (substring == nullptr || std::string(e.what()).find(substring) != std::string::npos)
            return;
        std::cerr << "FAIL: " << test_name << " (message missing '" << substring
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

ConfigField int_field(const std::string& name, std::int32_t default_value, std::set<Layer> layers) {
    return ConfigField{name, "int32", std::any{default_value}, std::move(layers), nullptr};
}
ConfigField bool_field(const std::string& name, bool default_value, std::set<Layer> layers) {
    return ConfigField{name, "bool", std::any{default_value}, std::move(layers), nullptr};
}
ConfigField string_field(const std::string& name, const std::string& default_value,
                         std::set<Layer> layers) {
    return ConfigField{name, "string", std::any{default_value}, std::move(layers), nullptr};
}

void register_demo_schema() {
    SchemaRegistry& reg = SchemaRegistry::instance();
    reg.clear_for_testing();
    reg.register_schema(Schema{
        "triattention",
        {
            int_field("kv_budget", 6144,
                      {Layer::SessionRequest, Layer::PlatformProfile, Layer::BundleDefault}),
            bool_field("protect_prefill", true, {Layer::SessionRequest, Layer::BundleDefault}),
            string_field("dump_scores_path", "", {Layer::SessionRequest}),
        }});
}

// ---- parse_set_token -------------------------------------------------------

void test_parse_set_token_basic() {
    auto t = trtmc::config::parse_set_token("ns.field=42");
    check(t.namespace_name == "ns" && t.field_name == "field" && t.raw_value == "42",
          "parse_set_token_basic");
}

void test_parse_set_token_equals_in_value() {
    auto t = trtmc::config::parse_set_token("ns.field=a=b=c");
    check(t.raw_value == "a=b=c", "parse_set_token_equals_in_value");
}

void test_parse_set_token_missing_equals() {
    expect_throws([] { trtmc::config::parse_set_token("ns.field42"); }, "missing '='",
                  "parse_set_token_missing_equals");
}

void test_parse_set_token_missing_dot() {
    expect_throws([] { trtmc::config::parse_set_token("nsfield=42"); }, "missing '.'",
                  "parse_set_token_missing_dot");
}

void test_parse_set_token_empty_parts() {
    expect_throws([] { trtmc::config::parse_set_token(".field=42"); }, "empty",
                  "parse_set_token_empty_ns");
    expect_throws([] { trtmc::config::parse_set_token("ns.=42"); }, "empty",
                  "parse_set_token_empty_field");
}

// ---- coerce_scalar --------------------------------------------------------

void test_coerce_int() {
    auto v = trtmc::config::coerce_scalar("42", "int32", "x.y");
    check(std::any_cast<std::int32_t>(v) == 42, "coerce_int");
    auto v64 = trtmc::config::coerce_scalar("-7", "int64", "x.y");
    check(std::any_cast<std::int64_t>(v64) == -7, "coerce_int64");
}

void test_coerce_int_rejects_float_text() {
    expect_throws([] { trtmc::config::coerce_scalar("3.14", "int32", "x.y"); }, "expected integer",
                  "coerce_int_rejects_float_text");
}

void test_coerce_float() {
    auto v = trtmc::config::coerce_scalar("3.14", "float", "x.y");
    check(std::any_cast<float>(v) > 3.13F && std::any_cast<float>(v) < 3.15F, "coerce_float");
}

void test_coerce_bool_vocab() {
    for (const std::string& t : {"true", "True", "TRUE", "1", "yes", "on"}) {
        auto v = trtmc::config::coerce_scalar(t, "bool", "x.y");
        check(std::any_cast<bool>(v) == true, ("coerce_bool_true:" + t).c_str());
    }
    for (const std::string& t : {"false", "False", "FALSE", "0", "no", "off"}) {
        auto v = trtmc::config::coerce_scalar(t, "bool", "x.y");
        check(std::any_cast<bool>(v) == false, ("coerce_bool_false:" + t).c_str());
    }
}

void test_coerce_bool_rejects_unknown() {
    expect_throws([] { trtmc::config::coerce_scalar("maybe", "bool", "x.y"); }, "expected bool",
                  "coerce_bool_rejects_unknown");
}

void test_coerce_string_identity() {
    auto v = trtmc::config::coerce_scalar("hello", "string", "x.y");
    check(std::any_cast<std::string>(v) == "hello", "coerce_string_identity");
}

void test_coerce_unknown_type_tag_raises() {
    expect_throws([] { trtmc::config::coerce_scalar("x", "list<int>", "x.y"); },
                  "unsupported type_tag", "coerce_unknown_type_tag_raises");
}

// ---- parse_layered_json ---------------------------------------------------

void test_parse_empty_object() {
    auto out = trtmc::config::parse_layered_json("{}");
    check(out.empty(), "parse_empty_object");
}

void test_parse_simple_object() {
    const char* text = R"({ "triattention": { "kv_budget": 4096, "protect_prefill": true } })";
    auto out = trtmc::config::parse_layered_json(text);
    check(out.size() == 1, "parse_simple_object: count");
    check(out.count("triattention") == 1, "parse_simple_object: ns");
    const auto& body = out.at("triattention");
    check(body.size() == 2, "parse_simple_object: field count");
    check(std::any_cast<std::int64_t>(body.at("kv_budget")) == 4096,
          "parse_simple_object: kv_budget");
    check(std::any_cast<bool>(body.at("protect_prefill")) == true,
          "parse_simple_object: protect_prefill");
}

void test_parse_scalar_kinds() {
    const char* text = R"({ "ns": {
        "i": 42, "neg": -17, "f": 3.14, "b_t": true, "b_f": false, "n": null, "s": "hi"
    }})";
    auto out = trtmc::config::parse_layered_json(text);
    const auto& body = out.at("ns");
    check(std::any_cast<std::int64_t>(body.at("i")) == 42, "parse_scalar:int");
    check(std::any_cast<std::int64_t>(body.at("neg")) == -17, "parse_scalar:neg");
    check(std::any_cast<double>(body.at("f")) > 3.13, "parse_scalar:float");
    check(std::any_cast<bool>(body.at("b_t")) == true, "parse_scalar:bool_true");
    check(std::any_cast<bool>(body.at("b_f")) == false, "parse_scalar:bool_false");
    check(!body.at("n").has_value(), "parse_scalar:null");
    check(std::any_cast<std::string>(body.at("s")) == "hi", "parse_scalar:string");
}

void test_parse_rejects_malformed() {
    expect_throws([] { trtmc::config::parse_layered_json("{ \"ns\" { } }"); }, "expected ':'",
                  "parse_missing_colon");
    expect_throws([] { trtmc::config::parse_layered_json("{ \"ns\": [1,2] }"); }, "expected '{'",
                  "parse_rejects_array");
    // Empty input yields an empty map (no throw) — treated as "no profile".
    auto empty = trtmc::config::parse_layered_json("");
    check(empty.empty(), "parse_empty_string_returns_empty");
}

// ---- build_cli_contribution + merge ---------------------------------------

LayerContribution session_with(const std::string& ns, const std::string& field, std::any value) {
    LayerContribution c;
    c.layer = Layer::SessionRequest;
    c.values[ns][field] = std::move(value);
    return c;
}

void test_build_cli_contribution_from_config_only() {
    register_demo_schema();
    trtmc::config::LayeredFileValues file_values;
    file_values["triattention"]["kv_budget"] = std::any{std::int64_t{4096}};
    file_values["triattention"]["protect_prefill"] = std::any{false};

    auto contrib = trtmc::config::build_cli_contribution(file_values, {});
    check(contrib.layer == Layer::SessionRequest, "cli_contrib: layer");
    check(contrib.values.at("triattention").size() == 2, "cli_contrib: fields merged");
    check(std::any_cast<std::int64_t>(contrib.values.at("triattention").at("kv_budget")) == 4096,
          "cli_contrib: kv_budget");
    check(std::any_cast<bool>(contrib.values.at("triattention").at("protect_prefill")) == false,
          "cli_contrib: protect_prefill");
}

void test_build_cli_contribution_coerces_set_tokens() {
    register_demo_schema();
    auto contrib = trtmc::config::build_cli_contribution(
        {}, {"triattention.kv_budget=8192", "triattention.protect_prefill=false",
             "triattention.dump_scores_path=/tmp/x.pt"});
    check(std::any_cast<std::int32_t>(contrib.values.at("triattention").at("kv_budget")) == 8192,
          "cli_contrib:set:int32");
    check(std::any_cast<bool>(contrib.values.at("triattention").at("protect_prefill")) == false,
          "cli_contrib:set:bool");
    check(std::any_cast<std::string>(contrib.values.at("triattention").at("dump_scores_path")) ==
              "/tmp/x.pt",
          "cli_contrib:set:string");
}

void test_set_overrides_config_within_session_layer() {
    register_demo_schema();
    trtmc::config::LayeredFileValues file_values;
    file_values["triattention"]["kv_budget"] = std::any{std::int64_t{4096}};
    auto contrib =
        trtmc::config::build_cli_contribution(file_values, {"triattention.kv_budget=8192"});
    check(std::any_cast<std::int32_t>(contrib.values.at("triattention").at("kv_budget")) == 8192,
          "set_overrides_config: value");
}

void test_unknown_namespace_raises() {
    register_demo_schema();
    trtmc::config::LayeredFileValues bad;
    bad["missing"]["x"] = std::any{std::int64_t{1}};
    expect_throws([&] { trtmc::config::build_cli_contribution(bad, {}); },
                  "unknown namespace 'missing'", "cli_contrib:unknown_ns");
}

void test_unknown_field_raises() {
    register_demo_schema();
    expect_throws([] { trtmc::config::build_cli_contribution({}, {"triattention.nope=1"}); },
                  "unknown field 'nope'", "cli_contrib:unknown_field");
}

void test_coercion_error_surfaces_field() {
    register_demo_schema();
    expect_throws(
        [] { trtmc::config::build_cli_contribution({}, {"triattention.kv_budget=not_a_number"}); },
        "triattention.kv_budget", "cli_contrib:coerce_err_field");
}

// ---- resolve_cli_config (end-to-end) --------------------------------------

std::string make_temp_json(const std::string& body) {
    namespace fs = std::filesystem;
    fs::path dir = fs::temp_directory_path() / "test_config_cli_support";
    fs::create_directories(dir);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "profile_%d.json", std::rand());
    const std::string p = (dir / buf).string();
    std::ofstream out(p);
    out << body;
    return p;
}

void test_resolve_builds_bundle_from_file_and_set() {
    register_demo_schema();
    const std::string path =
        make_temp_json(R"({"triattention":{"kv_budget":4096,"protect_prefill":false}})");
    auto bundle = trtmc::config::resolve_cli_config(path, {"triattention.kv_budget=8192"});
    check(bundle.get<std::int32_t>("triattention", "kv_budget") == 8192, "resolve: set wins");
    check(bundle.get<bool>("triattention", "protect_prefill") == false,
          "resolve: config value retained");
    check(bundle.source_of("triattention", "kv_budget") == Layer::SessionRequest,
          "resolve: source");
    // dump_scores_path untouched → schema default
    check(bundle.get<std::string>("triattention", "dump_scores_path") == "",
          "resolve: schema default");
    std::filesystem::remove(path);
}

void test_resolve_session_beats_platform() {
    register_demo_schema();
    LayerContribution platform;
    platform.layer = Layer::PlatformProfile;
    platform.values["triattention"]["kv_budget"] = std::any{std::int32_t{10240}};
    auto bundle =
        trtmc::config::resolve_cli_config("", {"triattention.kv_budget=8192"}, {platform});
    check(bundle.get<std::int32_t>("triattention", "kv_budget") == 8192,
          "resolve: session beats platform");
    check(bundle.source_of("triattention", "kv_budget") == Layer::SessionRequest,
          "resolve: source = session");
}

// ---- effective_config JSON serialization ---------------------------------

void test_bundle_to_effective_json_contains_source() {
    register_demo_schema();
    auto bundle = trtmc::config::resolve_cli_config("", {"triattention.kv_budget=8192"});
    std::string json = trtmc::config::bundle_to_effective_json(bundle);
    check(json.find("\"triattention\"") != std::string::npos, "effective_json: ns present");
    check(json.find("\"kv_budget\"") != std::string::npos, "effective_json: field present");
    check(json.find("\"source\": \"session_request\"") != std::string::npos,
          "effective_json: source");
    check(json.find("8192") != std::string::npos, "effective_json: value");
    // Schema default source is preserved too.
    check(json.find("\"source\": \"schema_default\"") != std::string::npos,
          "effective_json: default source");
}

void test_write_effective_config_next_to_places_file(std::string tmp_dir) {
    namespace fs = std::filesystem;
    register_demo_schema();
    auto bundle = trtmc::config::resolve_cli_config("", {"triattention.kv_budget=8192"});
    fs::path bundle_path = fs::path(tmp_dir) / "some" / "bundle.bundle";
    fs::create_directories(bundle_path.parent_path());
    std::string written =
        trtmc::config::write_effective_config_next_to(bundle, bundle_path.string());
    check(fs::exists(written), "write_effective: file exists");
    check(fs::path(written).filename() == "bundle.effective_config.json",
          "write_effective: sibling filename");
}

void test_try_write_effective_config_reports_unwritable_sidecar() {
    register_demo_schema();
    auto bundle = trtmc::config::resolve_cli_config("", {"triattention.kv_budget=8192"});

    const auto result =
        trtmc::config::try_write_effective_config_next_to(bundle, "/dev/null/bundle.bundle");

    check(!result.path.has_value(), "try_write_effective: unwritable path is non-fatal");
    check(!result.error.empty(), "try_write_effective: write error remains observable");
}

void test_runtime_resolution_survives_unwritable_effective_config_sidecar() {
    register_demo_schema();
    auto resolved = trtmc::detail::resolve_runtime_config(
        R"({"defaults":{"triattention":{"kv_budget":4096}}})", "/dev/null/bundle.bundle", "",
        {"triattention.kv_budget=8192"});

    check(resolved.has_value(), "runtime resolution: unwritable sidecar retains config");
    if (resolved) {
        check(resolved->get<std::int32_t>("triattention", "kv_budget") == 8192,
              "runtime resolution: session override remains active");
    }
}

void test_runtime_resolution_sidecar_policy(std::string tmp_dir) {
    namespace fs = std::filesystem;
    register_demo_schema();
    const fs::path bundle_path = fs::path(tmp_dir) / "locked.bundle";
    const fs::path sidecar_path = fs::path(tmp_dir) / "locked.effective_config.json";
    fs::remove(sidecar_path);
    auto resolved = trtmc::detail::resolve_runtime_config(
        R"({"defaults":{"triattention":{"kv_budget":4096}}})", bundle_path.string(), "",
        {"triattention.kv_budget=8192"});
    check(resolved.has_value(), "runtime sidecar policy: resolution succeeds");
#if defined(TRTMC_LOCKED_H3_RUNTIME)
    check(!fs::exists(sidecar_path), "locked runtime sidecar policy: no file is created");
#else
    check(fs::is_regular_file(sidecar_path),
          "normal runtime sidecar policy: effective config is created");
#endif
}

void test_runtime_resolution_failure_policy() {
    register_demo_schema();
    const auto resolve_invalid_override = [] {
        return trtmc::detail::resolve_runtime_config(
            R"({"defaults":{"triattention":{"kv_budget":4096}}})", "bundle.bundle", "",
            {"triattention.kv_budget=not_a_number"});
    };
#if defined(TRTMC_LOCKED_H3_RUNTIME)
    expect_throws([&] { (void)resolve_invalid_override(); }, "triattention.kv_budget",
                  "locked runtime resolution: invalid override fails closed");
#else
    check(!resolve_invalid_override().has_value(),
          "normal runtime resolution: invalid override retains best-effort fallback");
#endif
}

// ---- bundle defaults: block ------------------------------------------------

void test_extract_bundle_defaults_finds_block() {
    std::string header = R"({
        "model_id": "demo",
        "vocab_size": 100,
        "defaults": {
            "triattention": { "kv_budget": 4096, "protect_prefill": true }
        },
        "sections": {}
    })";
    auto out = trtmc::config::extract_bundle_defaults(header);
    check(out.size() == 1, "extract_defaults: one namespace");
    check(std::any_cast<std::int64_t>(out.at("triattention").at("kv_budget")) == 4096,
          "extract_defaults: kv_budget");
    check(std::any_cast<bool>(out.at("triattention").at("protect_prefill")) == true,
          "extract_defaults: protect_prefill");
}

void test_extract_bundle_defaults_absent_block() {
    std::string header = R"({ "model_id": "x", "sections": {} })";
    auto out = trtmc::config::extract_bundle_defaults(header);
    check(out.empty(), "extract_defaults: absent => empty");
}

void test_extract_bundle_defaults_key_in_string_not_confused() {
    // "defaults" appears inside a string literal; must be ignored.
    std::string header = R"({
        "comment": "default key: defaults",
        "defaults": { "ns": { "f": 1 } }
    })";
    auto out = trtmc::config::extract_bundle_defaults(header);
    check(out.size() == 1 && out.count("ns") == 1,
          "extract_defaults: skips key-like string literal");
}

void test_filter_drops_unregistered_namespaces() {
    SchemaRegistry& reg = SchemaRegistry::instance();
    reg.clear_for_testing();
    reg.register_schema(Schema{"known", {int_field("f", 0, {Layer::BundleDefault})}});

    LayerContribution contrib;
    contrib.layer = Layer::BundleDefault;
    contrib.values["known"]["f"] = std::any{std::int64_t{1}};
    contrib.values["stranger_danger"]["f"] = std::any{std::int64_t{2}};

    auto dropped = trtmc::config::filter_to_registered_namespaces(contrib, reg);
    check(dropped.size() == 1 && dropped.front() == "stranger_danger",
          "filter: dropped unknown namespace");
    check(contrib.values.count("known") == 1 && contrib.values.count("stranger_danger") == 0,
          "filter: known kept, unknown removed");
}

void test_resolve_pipeline_config_merges_bundle_and_session(std::string tmp_dir) {
    namespace fs = std::filesystem;
    register_demo_schema();
    const std::string header = R"({"defaults": {"triattention": {
        "kv_budget": 4096, "protect_prefill": false
    }}})";

    // Profile file supplies a platform-ish override here, layered as session.
    fs::path profile = fs::path(tmp_dir) / "profile.json";
    std::ofstream(profile) << R"({"triattention": {"dump_scores_path": "/tmp/x"}})";

    auto res = trtmc::config::resolve_pipeline_config(header, profile.string(),
                                                      {"triattention.kv_budget=8192"});

    // bundle_default + session contributions both land
    check(res.contributions.size() == 2, "resolve: two contributions");
    // Session beats bundle default; bundle default preserved where session silent.
    check(res.bundle.get<std::int32_t>("triattention", "kv_budget") == 8192,
          "resolve: session wins");
    check(res.bundle.get<bool>("triattention", "protect_prefill") == false,
          "resolve: bundle default preserved");
    check(res.bundle.get<std::string>("triattention", "dump_scores_path") == "/tmp/x",
          "resolve: session field routed");
    check(res.bundle.source_of("triattention", "kv_budget") == Layer::SessionRequest,
          "resolve: source kv_budget");
    check(res.bundle.source_of("triattention", "protect_prefill") == Layer::BundleDefault,
          "resolve: source protect_prefill");
}

void test_resolve_pipeline_config_tolerates_unknown_defaults() {
    SchemaRegistry& reg = SchemaRegistry::instance();
    reg.clear_for_testing();
    reg.register_schema(Schema{"known", {int_field("f", 5, {Layer::BundleDefault})}});
    const std::string header = R"({"defaults": {
        "known": {"f": 10},
        "not_yet_migrated": {"old": 1}
    }})";
    auto res = trtmc::config::resolve_pipeline_config(header, "", {});
    // Known namespace retained; unknown dropped at filter step.
    check(res.bundle.get<std::int64_t>("known", "f") == 10, "resolve: known ns kept");
    check(res.contributions.size() == 1, "resolve: only bundle_default layer survives");
}

void test_bundle_defaults_contribution_produces_bundle_default_layer() {
    register_demo_schema();
    std::string header = R"({ "defaults": { "triattention": { "kv_budget": 4096 } } })";
    auto contrib = trtmc::config::bundle_defaults_contribution(header);
    check(contrib.layer == Layer::BundleDefault, "contrib: layer");
    check(std::any_cast<std::int64_t>(contrib.values.at("triattention").at("kv_budget")) == 4096,
          "contrib: kv_budget");

    // Merge: session beats bundle default; bundle default fills gaps.
    LayerContribution session;
    session.layer = Layer::SessionRequest;
    session.values["triattention"]["kv_budget"] = std::any{std::int32_t{8192}};
    auto merged = trtmc::config::ConfigBundle::build({contrib, session});
    check(merged.get<std::int32_t>("triattention", "kv_budget") == 8192, "merge: session wins");
    check(merged.source_of("triattention", "kv_budget") == Layer::SessionRequest,
          "merge: session source");

    auto without_session = trtmc::config::ConfigBundle::build({contrib});
    check(without_session.source_of("triattention", "kv_budget") == Layer::BundleDefault,
          "merge: bundle default fills gap");
}

} // namespace

int main() {
    test_parse_set_token_basic();
    test_parse_set_token_equals_in_value();
    test_parse_set_token_missing_equals();
    test_parse_set_token_missing_dot();
    test_parse_set_token_empty_parts();

    test_coerce_int();
    test_coerce_int_rejects_float_text();
    test_coerce_float();
    test_coerce_bool_vocab();
    test_coerce_bool_rejects_unknown();
    test_coerce_string_identity();
    test_coerce_unknown_type_tag_raises();

    test_parse_empty_object();
    test_parse_simple_object();
    test_parse_scalar_kinds();
    test_parse_rejects_malformed();

    test_build_cli_contribution_from_config_only();
    test_build_cli_contribution_coerces_set_tokens();
    test_set_overrides_config_within_session_layer();
    test_unknown_namespace_raises();
    test_unknown_field_raises();
    test_coercion_error_surfaces_field();

    test_resolve_builds_bundle_from_file_and_set();
    test_resolve_session_beats_platform();

    test_bundle_to_effective_json_contains_source();
    test_try_write_effective_config_reports_unwritable_sidecar();
    test_runtime_resolution_survives_unwritable_effective_config_sidecar();
    test_runtime_resolution_failure_policy();

    test_extract_bundle_defaults_finds_block();
    test_extract_bundle_defaults_absent_block();
    test_extract_bundle_defaults_key_in_string_not_confused();
    test_bundle_defaults_contribution_produces_bundle_default_layer();
    test_filter_drops_unregistered_namespaces();
    test_resolve_pipeline_config_tolerates_unknown_defaults();
    {
        namespace fs = std::filesystem;
        fs::path tmp = fs::temp_directory_path() / "test_resolve_pipeline";
        trtmc_test::remove_all_safe(tmp.string());
        fs::create_directories(tmp);
        test_resolve_pipeline_config_merges_bundle_and_session(tmp.string());
    }

    // Writing to a temp dir — resolve at runtime.
    {
        namespace fs = std::filesystem;
        fs::path tmp = fs::temp_directory_path() / "test_config_cli_support_write";
        trtmc_test::remove_all_safe(tmp.string());
        fs::create_directories(tmp);
        test_write_effective_config_next_to_places_file(tmp.string());
        test_runtime_resolution_sidecar_policy(tmp.string());
    }

    SchemaRegistry::instance().clear_for_testing();
    if (g_failures != 0) {
        std::cerr << g_failures << " test(s) failed\n";
        return 1;
    }
    return 0;
}

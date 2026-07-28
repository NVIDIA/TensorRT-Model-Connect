/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// CLI support helpers mirroring tensorrt_model_connect.runtime_config.cli_support.
//
// Same responsibilities, same fixed surface:
//   --config <file.json>         → one layer's worth of values, loaded as
//                                  nested {namespace: {field: scalar}}.
//   --set <ns.field=value>       → repeatable override within the session.
//
// The CLI never grows per-knob flags. Features that need runtime
// configuration declare a schema with REGISTER_CONFIG_SCHEMA and their
// values flow in through these two flags.
//
// JSON is the canonical wire format on the C++ side: load_layered_file()
// accepts .json profiles only because the container does not ship with
// yaml-cpp. An orchestration layer may parse YAML and convert it before
// invoking the C++ config APIs, but C++ CLI --config does not do that.

#include "trtmc/config/config_bundle.h"
#include "trtmc/config/schema_registry.h"

#include <any>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace trtmc::config {

// Parse one --set token into its three parts. Throws std::invalid_argument
// on malformed input, with the raw token quoted in the message.
//
// Note: the first '=' splits key from value, so values can legitimately
// contain further '=' characters.
struct SetToken {
    std::string namespace_name;
    std::string field_name;
    std::string raw_value;
};
SetToken parse_set_token(std::string_view token);

// Coerce a raw string value to the schema's declared scalar type. ``where``
// appears in error messages for diagnostics (e.g. "triattention.kv_budget").
// Supported type_tags: int/int32/int64, float/double, bool, string/str/path.
std::any coerce_scalar(const std::string& raw, const std::string& type_tag,
                       const std::string& where);

// Minimal JSON-subset parser scoped to the config profile shape:
// a top-level object whose values are objects whose values are scalars
// (null/bool/int/float/string). Arrays and nested objects beyond that
// depth are not supported yet and raise std::invalid_argument.
//
// Return: nested map namespace → field → value (as std::any holding one
// of: bool, int64_t, double, std::string, std::nullptr_t).
using LayeredFileValues =
    std::unordered_map<std::string, std::unordered_map<std::string, std::any>>;

LayeredFileValues parse_layered_json(std::string_view text);

// File-based variant: reads the file and calls parse_layered_json.
// Throws on IO failures or parse errors, with the filesystem path in
// the message.
LayeredFileValues load_layered_file(const std::string& path);

// Merge --config file values and --set tokens into a single
// LayerContribution. Within `layer`, --set wins on collision — the
// helper resolves them before producing the contribution so
// ConfigBundle::build never sees same-layer collisions.
//
// Each namespace/field is validated against the registered schemas;
// unknown namespace or field fails fast. --set values are coerced from
// strings according to the field's declared type_tag.
LayerContribution
build_cli_contribution(const LayeredFileValues& config_file_values,
                       const std::vector<std::string>& set_tokens,
                       Layer layer = Layer::SessionRequest,
                       const SchemaRegistry& registry = SchemaRegistry::instance());

// End-to-end helper: parse/load flags → merge → build a ConfigBundle.
// `extra_contributions` lets callers inject platform / bundle-default /
// build-time layers alongside the session layer.
ConfigBundle resolve_cli_config(const std::string& config_path,             // empty → skip file
                                const std::vector<std::string>& set_tokens, // may be empty
                                const std::vector<LayerContribution>& extra_contributions = {},
                                const SchemaRegistry& registry = SchemaRegistry::instance());

// Write the effective-config JSON alongside an artifact. For artifact
// ``foo/bar.trtfb`` and default suffix the output is
// ``foo/bar.effective_config.json``. Returns the written path.
std::string write_effective_config_next_to(const ConfigBundle& bundle,
                                           const std::string& artifact_path,
                                           const std::string& suffix = ".effective_config.json");

// Best-effort variant for runtime paths where the artifact may be read-only.
// Resolution remains authoritative even when the diagnostic sidecar cannot
// be written; callers can surface ``error`` without changing behavior.
struct EffectiveConfigWriteResult {
    std::optional<std::string> path;
    std::string error;
};
EffectiveConfigWriteResult
try_write_effective_config_next_to(const ConfigBundle& bundle, const std::string& artifact_path,
                                   const std::string& suffix = ".effective_config.json");

// Low-level: serialize a bundle to JSON text. Stable field/namespace
// ordering so two identical bundles produce byte-identical output.
std::string bundle_to_effective_json(const ConfigBundle& bundle);

// Parse the materialized ``config.json`` section and return the direct
// top-level ``"defaults": { ... }`` object as LayeredFileValues. Returns an
// empty map when the document is malformed, the root is not an object, or the
// direct key is absent or not object-valued. Nested keys named ``defaults`` do
// not contribute runtime defaults. Duplicate object keys are rejected before
// the JSON DOM can collapse them; invalid defaults values retain the fail-fast
// behavior of parse_layered_json.
LayeredFileValues extract_bundle_defaults(const std::string& config_json);

// Convenience: wrap ``extract_bundle_defaults`` as a BundleDefault layer.
LayerContribution bundle_defaults_contribution(const std::string& config_json);

// Drop namespaces from ``contrib`` that aren't known to ``registry`` —
// typically used for the BundleDefault layer so old bundles carrying
// defaults for clusters that haven't migrated yet don't fail-fast.
//
// Emits one stderr line per dropped namespace (informational — not an
// error). Mutates ``contrib`` in place and returns the list of
// dropped namespace names for diagnostics.
std::vector<std::string>
filter_to_registered_namespaces(LayerContribution& contrib,
                                const SchemaRegistry& registry = SchemaRegistry::instance());

// High-level helper used by PipelineFactory. It receives the materialized
// ``config.json`` section plus session-layer CLI inputs and produces a merged
// ConfigBundle ready to attach to PipelineContext. A top-level ``defaults``
// object contributes BundleDefault; ``--config``/``--set`` contribute
// SessionRequest. The current factory does not pass ``BundleInfo.defaults``
// from the binary header or inject separate BuildTime or PlatformProfile
// contributions. Also returns the full contribution list so callers can feed
// it into ``write_effective_config_next_to``.
struct PipelineConfigResolution {
    ConfigBundle bundle;
    std::vector<LayerContribution> contributions;
};
PipelineConfigResolution
resolve_pipeline_config(const std::string& config_json, const std::string& config_path,
                        const std::vector<std::string>& set_tokens,
                        const SchemaRegistry& registry = SchemaRegistry::instance());

} // namespace trtmc::config

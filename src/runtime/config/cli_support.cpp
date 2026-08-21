/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/config/cli_support.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include <nlohmann/json.hpp>

namespace trtmc::config {

namespace {

std::string tolower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string strip(std::string_view sv) {
    const auto is_ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    std::size_t a = 0;
    std::size_t b = sv.size();
    while (a < b && is_ws(sv[a]))
        ++a;
    while (b > a && is_ws(sv[b - 1]))
        --b;
    return std::string(sv.substr(a, b - a));
}

// ---- JSON conversion helpers --------------------------------------------

std::any json_to_any(const nlohmann::json& j) {
    if (j.is_null()) return std::any{};
    if (j.is_boolean()) return std::any{j.get<bool>()};
    if (j.is_number_integer()) return std::any{j.get<std::int64_t>()};
    if (j.is_number_unsigned()) return std::any{static_cast<std::int64_t>(j.get<std::uint64_t>())};
    if (j.is_number_float()) return std::any{j.get<double>()};
    if (j.is_string()) return std::any{j.get<std::string>()};
    throw std::invalid_argument("unsupported json type");
}

std::string c_string_or_empty(const char* value) {
    if (value == nullptr)
        return {};
    return value;
}

nlohmann::json any_to_json(const std::any& v) {
    if (!v.has_value()) return nullptr;
    if (v.type() == typeid(bool)) return std::any_cast<bool>(v);
    if (v.type() == typeid(std::int32_t)) return std::any_cast<std::int32_t>(v);
    if (v.type() == typeid(std::int64_t)) return std::any_cast<std::int64_t>(v);
    if (v.type() == typeid(int)) return std::any_cast<int>(v);
    if (v.type() == typeid(double)) return std::any_cast<double>(v);
    if (v.type() == typeid(float)) return std::any_cast<float>(v);
    if (v.type() == typeid(std::string))
        return std::any_cast<const std::string&>(v);
    if (v.type() == typeid(const char*))
        return c_string_or_empty(std::any_cast<const char*>(v));
    return "<unrepresentable>";
}

// ---- Field lookup shared by --set and --config paths --------------------

// Returns a pointer into registry-owned memory. Pointer stays valid as
// long as the registry is not mutated. We deliberately return a raw
// pointer rather than a reference so callers can't bind it to a rvalue
// and trigger the -Wdangling-reference false positive.
const ConfigField* lookup_field_or_throw(const SchemaRegistry& registry, const std::string& ns,
                                         const std::string& field_name, const std::string& origin) {
    const Schema* schema = registry.lookup(ns);
    if (schema == nullptr) {
        std::string known;
        for (const auto& n : registry.registered_namespaces()) {
            if (!known.empty())
                known += ", ";
            known += n;
        }
        throw std::invalid_argument(origin + ": unknown namespace '" + ns +
                                    "'. Known: " + (known.empty() ? "<none registered>" : known));
    }
    for (const auto& cfg_field : schema->fields) {
        if (cfg_field.name == field_name)
            return &cfg_field;
    }
    std::string known;
    for (const auto& f : schema->fields) {
        if (!known.empty())
            known += ", ";
        known += f.name;
    }
    throw std::invalid_argument(origin + ": unknown field '" + field_name + "' in namespace '" +
                                ns + "'. Known: " + known);
}

} // namespace

SetToken parse_set_token(std::string_view token) {
    auto eq = token.find('=');
    if (eq == std::string_view::npos) {
        throw std::invalid_argument(std::string("--set expects 'ns.field=value' (got ") +
                                    std::string(token) + "; missing '=')");
    }
    std::string key = strip(token.substr(0, eq));
    std::string raw_value(token.substr(eq + 1));
    auto dot = key.find('.');
    if (dot == std::string::npos) {
        throw std::invalid_argument(std::string("--set expects 'ns.field=value' (got ") +
                                    std::string(token) + "; missing '.')");
    }
    SetToken out;
    out.namespace_name = strip(std::string_view(key).substr(0, dot));
    out.field_name = strip(std::string_view(key).substr(dot + 1));
    out.raw_value = raw_value;
    if (out.namespace_name.empty() || out.field_name.empty()) {
        throw std::invalid_argument(std::string("--set expects 'ns.field=value' (got ") +
                                    std::string(token) + "; empty namespace or field)");
    }
    return out;
}

namespace {

std::any coerce_integer(const std::string& raw, const std::string& tag, const std::string& where) {
    try {
        std::size_t consumed = 0;
        std::int64_t v = std::stoll(raw, &consumed);
        if (consumed != raw.size())
            throw std::invalid_argument("trailing chars");
        if (tag == "int64")
            return std::any{v};
        return std::any{static_cast<std::int32_t>(v)};
    } catch (const std::exception&) {
        throw std::invalid_argument(where + ": expected integer, got '" + raw + "'");
    }
}

std::any coerce_floating(const std::string& raw, const std::string& tag, const std::string& where) {
    try {
        std::size_t consumed = 0;
        double d = std::stod(raw, &consumed);
        if (consumed != raw.size())
            throw std::invalid_argument("trailing chars");
        if (tag == "double")
            return std::any{d};
        return std::any{static_cast<float>(d)};
    } catch (const std::exception&) {
        throw std::invalid_argument(where + ": expected float, got '" + raw + "'");
    }
}

std::any coerce_boolean(const std::string& raw, const std::string& where) {
    const std::string low = tolower_copy(strip(raw));
    static const std::string_view kTrue[] = {"true", "1", "yes", "on"};
    static const std::string_view kFalse[] = {"false", "0", "no", "off"};
    for (auto tok : kTrue)
        if (low == tok)
            return std::any{true};
    for (auto tok : kFalse)
        if (low == tok)
            return std::any{false};
    throw std::invalid_argument(where + ": expected bool (true/false), got '" + raw + "'");
}

bool is_integer_tag(const std::string& tag) {
    return tag == "int" || tag == "int32" || tag == "int64";
}

bool is_floating_tag(const std::string& tag) {
    return tag == "float" || tag == "double";
}

bool is_string_tag(const std::string& tag) {
    return tag == "string" || tag == "str" || tag == "path";
}

} // namespace

std::any coerce_scalar(const std::string& raw, const std::string& type_tag,
                       const std::string& where) {
    const std::string tag = tolower_copy(type_tag);
    if (is_integer_tag(tag))
        return coerce_integer(raw, tag, where);
    if (is_floating_tag(tag))
        return coerce_floating(raw, tag, where);
    if (tag == "bool")
        return coerce_boolean(raw, where);
    if (is_string_tag(tag))
        return std::any{raw};
    throw std::invalid_argument(where + ": schema declares unsupported type_tag '" + type_tag +
                                "' for --set coercion");
}

LayeredFileValues parse_layered_json(std::string_view text) {
    if (text.empty()) return {};
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(text);
    } catch (const nlohmann::json::parse_error& e) {
        throw std::invalid_argument(std::string("expected ':' ") + e.what());
    }
    if (j.is_null()) return {};
    if (!j.is_object()) {
        throw std::invalid_argument("expected '{'");
    }
    LayeredFileValues out;
    for (auto& [ns, fields] : j.items()) {
        if (!fields.is_object()) {
            throw std::invalid_argument("expected '{'");
        }
        for (auto& [field, value] : fields.items()) {
            out[ns][field] = json_to_any(value);
        }
    }
    return out;
}

LayeredFileValues load_layered_file(const std::string& path) {
    namespace fs = std::filesystem;
    if (!fs::exists(path))
        throw std::invalid_argument("--config file not found: " + path);
    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in)
        throw std::invalid_argument("--config file cannot be opened: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string body = ss.str();

    const std::string ext = fs::path(path).extension().string();
    const std::string ext_low = tolower_copy(ext);
    if (ext_low == ".json")
        return parse_layered_json(body);
    if (ext_low == ".yaml" || ext_low == ".yml") {
        throw std::invalid_argument(
            "--config " + path +
            ": YAML is not supported by the C++ loader; convert to JSON or "
            "load via a wrapper (tensorrt_model_connect/cli.py accepts YAML).");
    }
    throw std::invalid_argument("--config " + path + ": unsupported extension '" + ext +
                                "' (expected .json)");
}

LayerContribution build_cli_contribution(const LayeredFileValues& config_file_values,
                                         const std::vector<std::string>& set_tokens, Layer layer,
                                         const SchemaRegistry& registry) {
    LayerContribution out;
    out.layer = layer;

    for (const auto& ns_entry : config_file_values) {
        const std::string& ns = ns_entry.first;
        for (const auto& f_entry : ns_entry.second) {
            (void)lookup_field_or_throw(registry, ns, f_entry.first, "--config"); // validate

            out.values[ns][f_entry.first] = f_entry.second;
        }
    }

    // --set wins within this layer; later tokens win on repeated key.
    for (const auto& token : set_tokens) {
        SetToken st = parse_set_token(token);
        const ConfigField* field =
            lookup_field_or_throw(registry, st.namespace_name, st.field_name, "--set");
        std::any coerced = coerce_scalar(st.raw_value, field->type,
                                         "--set " + st.namespace_name + "." + st.field_name);
        out.values[st.namespace_name][st.field_name] = coerced;
    }
    return out;
}

ConfigBundle resolve_cli_config(const std::string& config_path,
                                const std::vector<std::string>& set_tokens,
                                const std::vector<LayerContribution>& extra_contributions,
                                const SchemaRegistry& registry) {
    LayeredFileValues file_values;
    if (!config_path.empty())
        file_values = load_layered_file(config_path);

    std::vector<LayerContribution> contribs = extra_contributions;
    contribs.push_back(
        build_cli_contribution(file_values, set_tokens, Layer::SessionRequest, registry));
    return ConfigBundle::build(contribs, registry);
}

std::string bundle_to_effective_json(const ConfigBundle& bundle) {
    nlohmann::json out = nlohmann::json::object();
    for (const auto& [ns, fields] : bundle.all()) {
        nlohmann::json ns_json = nlohmann::json::object();
        for (const auto& [fname, rv] : fields) {
            nlohmann::json field_json;
            field_json["value"] = any_to_json(rv.value);
            field_json["source"] = layer_name(rv.source);
            ns_json[fname] = field_json;
        }
        out[ns] = ns_json;
    }
    return out.dump(2) + "\n";
}

LayeredFileValues extract_bundle_defaults(const std::string& header_json) {
    if (header_json.empty()) return {};
    auto j = nlohmann::json::parse(header_json);
    if (!j.contains("defaults") || j["defaults"].is_null()) {
        return {};
    }
    LayeredFileValues out;
    for (auto& [ns, fields] : j["defaults"].items()) {
        for (auto& [field, value] : fields.items()) {
            out[ns][field] = json_to_any(value);
        }
    }
    return out;
}

LayerContribution bundle_defaults_contribution(const std::string& header_json) {
    LayerContribution out;
    out.layer = Layer::BundleDefault;
    out.values = extract_bundle_defaults(header_json);
    return out;
}

std::vector<std::string> filter_to_registered_namespaces(LayerContribution& contrib,
                                                         const SchemaRegistry& registry) {
    std::vector<std::string> dropped;
    for (auto it = contrib.values.begin(); it != contrib.values.end();) {
        if (registry.lookup(it->first) == nullptr) {
            std::cerr << "[trtmc.config] dropping bundle default for "
                      << "unregistered namespace: " << it->first << " (layer "
                      << layer_name(contrib.layer) << ")\n";
            dropped.push_back(it->first);
            it = contrib.values.erase(it);
        } else {
            ++it;
        }
    }
    return dropped;
}

PipelineConfigResolution resolve_pipeline_config(const std::string& header_json,
                                                 const std::string& config_path,
                                                 const std::vector<std::string>& set_tokens,
                                                 const SchemaRegistry& registry) {
    PipelineConfigResolution out;

    // BundleDefault layer. Filter unknown namespaces so old bundles whose
    // clusters haven't been migrated yet don't fail-fast at load time.
    LayerContribution bundle_defaults = bundle_defaults_contribution(header_json);
    filter_to_registered_namespaces(bundle_defaults, registry);
    if (!bundle_defaults.values.empty())
        out.contributions.push_back(bundle_defaults);

    // SessionRequest layer (only if the caller supplied something).
    if (!config_path.empty() || !set_tokens.empty()) {
        LayeredFileValues file_values;
        if (!config_path.empty())
            file_values = load_layered_file(config_path);
        LayerContribution session =
            build_cli_contribution(file_values, set_tokens, Layer::SessionRequest, registry);
        out.contributions.push_back(session);
    }

    out.bundle = ConfigBundle::build(out.contributions, registry);
    return out;
}

std::string write_effective_config_next_to(const ConfigBundle& bundle,
                                           const std::string& artifact_path,
                                           const std::string& suffix) {
    namespace fs = std::filesystem;
    fs::path p(artifact_path);
    p.replace_extension(suffix);
    fs::create_directories(p.parent_path());
    std::ofstream out(p);
    if (!out)
        throw std::invalid_argument("cannot write effective_config to: " + p.string());
    out << bundle_to_effective_json(bundle);
    return p.string();
}

EffectiveConfigWriteResult try_write_effective_config_next_to(const ConfigBundle& bundle,
                                                              const std::string& artifact_path,
                                                              const std::string& suffix) {
    try {
        return EffectiveConfigWriteResult{
            write_effective_config_next_to(bundle, artifact_path, suffix), ""};
    } catch (const std::exception& e) {
        return EffectiveConfigWriteResult{std::nullopt, e.what()};
    }
}

} // namespace trtmc::config

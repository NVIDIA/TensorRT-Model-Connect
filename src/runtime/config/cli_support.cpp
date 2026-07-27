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

// ---- Tiny JSON parser (scoped to config profile shape) -------------------

struct Parser {
    std::string_view text;
    std::size_t i = 0;

    [[noreturn]] void fail(const std::string& msg) const {
        // Crude line/col for diagnostics. Counting the prefix is O(n) but
        // this runs once at session start — not hot.
        std::size_t line = 1;
        std::size_t col = 1;
        for (std::size_t j = 0; j < i && j < text.size(); ++j) {
            if (text[j] == '\n') {
                ++line;
                col = 1;
            } else
                ++col;
        }
        throw std::invalid_argument("parse_layered_json: " + msg + " at line " +
                                    std::to_string(line) + " col " + std::to_string(col));
    }

    bool is_space(char c) const { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
    bool is_line_comment_start(std::size_t at) const {
        return at + 1 < text.size() && text[at] == '/' && text[at + 1] == '/';
    }
    void skip_line_comment() {
        while (i < text.size() && text[i] != '\n')
            ++i;
    }

    void skip_ws() {
        while (i < text.size()) {
            if (is_space(text[i])) {
                ++i;
                continue;
            }
            if (is_line_comment_start(i)) {
                skip_line_comment();
                continue;
            }
            break;
        }
    }

    char peek() {
        skip_ws();
        if (i >= text.size())
            fail("unexpected end of input");
        return text[i];
    }

    void expect(char c) {
        if (peek() != c) {
            std::string msg = "expected '";
            msg.push_back(c);
            msg += "' got '";
            msg.push_back(text[i]);
            msg += "'";
            fail(msg);
        }
        ++i;
    }

    char decode_string_escape(char esc) {
        // JSON string-escape vocabulary. Unknown escapes fail the parse.
        static constexpr std::pair<char, char> table[] = {
            {'"', '"'},  {'\\', '\\'}, {'/', '/'},  {'n', '\n'},
            {'t', '\t'}, {'r', '\r'},  {'b', '\b'}, {'f', '\f'},
        };
        for (const auto& entry : table)
            if (entry.first == esc)
                return entry.second;
        fail("unsupported escape in string");
    }

    std::string parse_string() {
        skip_ws();
        if (i >= text.size() || text[i] != '"')
            fail("expected string");
        ++i;
        std::string out;
        while (i < text.size()) {
            char c = text[i++];
            if (c == '"')
                return out;
            if (c == '\\' && i < text.size()) {
                out.push_back(decode_string_escape(text[i++]));
                continue;
            }
            out.push_back(c);
        }
        fail("unterminated string");
    }

    std::any parse_bool_literal() {
        if (text.compare(i, 4, "true") == 0) {
            i += 4;
            return std::any{true};
        }
        if (text.compare(i, 5, "false") == 0) {
            i += 5;
            return std::any{false};
        }
        fail("expected bool literal");
    }

    std::any parse_null_literal() {
        // JSON null → empty std::any. has_value() is the canonical "null
        // was present in the profile" check downstream.
        if (text.compare(i, 4, "null") == 0) {
            i += 4;
            return std::any{};
        }
        fail("expected null literal");
    }

    static bool is_digit(char c) { return c >= '0' && c <= '9'; }
    static bool is_dot_or_exp(char c) { return c == '.' || c == 'e' || c == 'E'; }
    static bool is_sign_char(char c) { return c == '-' || c == '+'; }
    bool prev_was_exp(std::size_t at) const {
        if (at == 0)
            return false;
        char p = text[at - 1];
        return p == 'e' || p == 'E';
    }

    bool is_number_continuation(std::size_t at, bool& out_has_dot_or_exp) const {
        char ch = text[at];
        if (is_digit(ch))
            return true;
        if (is_dot_or_exp(ch)) {
            out_has_dot_or_exp = true;
            return true;
        }
        if (is_sign_char(ch))
            return prev_was_exp(at);
        return false;
    }

    std::any parse_number_literal() {
        std::size_t start = i;
        if (text[i] == '-' || text[i] == '+')
            ++i;
        bool has_dot_or_exp = false;
        while (i < text.size() && is_number_continuation(i, has_dot_or_exp))
            ++i;
        if (start == i)
            fail("expected value");
        std::string num_text(text.substr(start, i - start));
        try {
            if (has_dot_or_exp)
                return std::any{std::stod(num_text)};
            return std::any{static_cast<std::int64_t>(std::stoll(num_text))};
        } catch (const std::exception&) {
            fail("invalid number: " + num_text);
        }
    }

    std::any parse_scalar() {
        skip_ws();
        if (i >= text.size())
            fail("expected scalar");
        char c = text[i];
        if (c == '"')
            return std::any{parse_string()};
        if (c == 't' || c == 'f')
            return parse_bool_literal();
        if (c == 'n')
            return parse_null_literal();
        return parse_number_literal();
    }

    // One inner object: {field: scalar, ...}
    std::unordered_map<std::string, std::any> parse_inner_object() {
        expect('{');
        std::unordered_map<std::string, std::any> out;
        skip_ws();
        if (i < text.size() && text[i] == '}') {
            ++i;
            return out;
        }
        while (true) {
            std::string key = parse_string();
            skip_ws();
            expect(':');
            if (out.count(key) != 0)
                fail("duplicate field key: " + key);
            out[key] = parse_scalar();
            skip_ws();
            if (i < text.size() && text[i] == ',') {
                ++i;
                continue;
            }
            if (i < text.size() && text[i] == '}') {
                ++i;
                break;
            }
            fail("expected ',' or '}' in object");
        }
        return out;
    }

    LayeredFileValues parse_outer_object() {
        expect('{');
        LayeredFileValues out;
        skip_ws();
        if (i < text.size() && text[i] == '}') {
            ++i;
            return out;
        }
        while (true) {
            std::string ns = parse_string();
            skip_ws();
            expect(':');
            if (out.count(ns) != 0)
                fail("duplicate namespace key: " + ns);
            out[ns] = parse_inner_object();
            skip_ws();
            if (i < text.size() && text[i] == ',') {
                ++i;
                continue;
            }
            if (i < text.size() && text[i] == '}') {
                ++i;
                break;
            }
            fail("expected ',' or '}' at outer level");
        }
        return out;
    }
};

// ---- std::any serialization for effective_config.json -------------------

void append_json_escaped_string(std::ostringstream& os, const std::string& s) {
    os << '"';
    for (char c : s) {
        switch (c) {
        case '"':
            os << "\\\"";
            break;
        case '\\':
            os << "\\\\";
            break;
        case '\n':
            os << "\\n";
            break;
        case '\t':
            os << "\\t";
            break;
        case '\r':
            os << "\\r";
            break;
        case '\b':
            os << "\\b";
            break;
        case '\f':
            os << "\\f";
            break;
        default:
            os << c;
        }
    }
    os << '"';
}

bool try_append_numeric(std::ostringstream& os, const std::any& v) {
    if (v.type() == typeid(std::int32_t)) {
        os << std::any_cast<std::int32_t>(v);
        return true;
    }
    if (v.type() == typeid(std::int64_t)) {
        os << std::any_cast<std::int64_t>(v);
        return true;
    }
    if (v.type() == typeid(int)) {
        os << std::any_cast<int>(v);
        return true;
    }
    if (v.type() == typeid(double)) {
        os << std::any_cast<double>(v);
        return true;
    }
    if (v.type() == typeid(float)) {
        os << std::any_cast<float>(v);
        return true;
    }
    return false;
}

void append_json_scalar(std::ostringstream& os, const std::any& v) {
    if (!v.has_value()) {
        os << "null";
        return;
    }
    if (v.type() == typeid(bool)) {
        os << (std::any_cast<bool>(v) ? "true" : "false");
        return;
    }
    if (try_append_numeric(os, v))
        return;
    if (v.type() == typeid(std::string)) {
        append_json_escaped_string(os, std::any_cast<const std::string&>(v));
        return;
    }
    if (v.type() == typeid(const char*)) {
        const char* s = std::any_cast<const char*>(v);
        append_json_escaped_string(os, s ? std::string(s) : std::string());
        return;
    }
    os << "\"<unrepresentable>\"";
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
    Parser p{text, 0};
    p.skip_ws();
    if (p.i >= text.size())
        return {};
    LayeredFileValues out = p.parse_outer_object();
    p.skip_ws();
    if (p.i != text.size())
        p.fail("unexpected trailing content");
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
    std::ostringstream os;
    os << "{\n";
    // Sort namespaces for stable output.
    std::vector<std::string> ns_names;
    for (const auto& entry : bundle.all())
        ns_names.push_back(entry.first);
    std::sort(ns_names.begin(), ns_names.end());

    for (std::size_t ni = 0; ni < ns_names.size(); ++ni) {
        const auto& ns = ns_names[ni];
        os << "  \"" << ns << "\": {\n";
        std::vector<std::string> field_names;
        for (const auto& f : bundle.all().at(ns))
            field_names.push_back(f.first);
        std::sort(field_names.begin(), field_names.end());

        for (std::size_t fi = 0; fi < field_names.size(); ++fi) {
            const auto& fname = field_names[fi];
            const ResolvedValue& rv = bundle.all().at(ns).at(fname);
            os << "    \"" << fname << "\": {\n";
            os << "      \"value\": ";
            append_json_scalar(os, rv.value);
            os << ",\n";
            os << "      \"source\": \"" << layer_name(rv.source) << "\"\n";
            os << "    }";
            if (fi + 1 < field_names.size())
                os << ",";
            os << "\n";
        }
        os << "  }";
        if (ni + 1 < ns_names.size())
            os << ",";
        os << "\n";
    }
    os << "}\n";
    return os.str();
}

namespace {

bool is_json_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

std::size_t skip_json_ws(const std::string& text, std::size_t p) {
    while (p < text.size() && is_json_space(text[p]))
        ++p;
    return p;
}

// Starting at a '{' at index `start`, scan forward honoring string literals
// and return the index just past the matching close '}'. Returns
// std::string::npos on unbalanced or missing close.
std::size_t match_object_end(const std::string& text, std::size_t start) {
    int depth = 0;
    bool in_string = false;
    for (std::size_t p = start; p < text.size(); ++p) {
        char c = text[p];
        if (in_string) {
            if (c == '\\' && p + 1 < text.size()) {
                ++p;
                continue;
            }
            if (c == '"')
                in_string = false;
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '{')
            ++depth;
        else if (c == '}' && --depth == 0)
            return p + 1;
    }
    return std::string::npos;
}

// Confirm that a "<key>" match at `pattern_end` is actually followed by
// a colon and then an object open-brace. Returns the index of the '{' on
// success, std::string::npos otherwise.
std::size_t find_object_open_after_key(const std::string& text, std::size_t pattern_end) {
    std::size_t p = skip_json_ws(text, pattern_end);
    if (p >= text.size() || text[p] != ':')
        return std::string::npos;
    p = skip_json_ws(text, p + 1);
    if (p >= text.size() || text[p] != '{')
        return std::string::npos;
    return p;
}

// Locate the object value for ``"<key>":`` in a JSON text and return the
// substring from the opening '{' to its matching close '}' (inclusive).
// Returns empty string if the key is absent or the value isn't an object.
// Honors string-literal escaping so that '{' or '}' inside quoted values
// don't confuse brace matching.
std::string find_object_value_for_key(const std::string& text, const std::string& key) {
    const std::string pattern = "\"" + key + "\"";
    std::size_t pos = 0;
    while ((pos = text.find(pattern, pos)) != std::string::npos) {
        std::size_t open = find_object_open_after_key(text, pos + pattern.size());
        if (open == std::string::npos) {
            // Not a key (or value isn't an object). Skip past and keep
            // looking — handles the case where the literal appears inside
            // another string value.
            pos += pattern.size();
            continue;
        }
        std::size_t end = match_object_end(text, open);
        if (end == std::string::npos)
            return {};
        return text.substr(open, end - open);
    }
    return {};
}

} // namespace

LayeredFileValues extract_bundle_defaults(const std::string& config_json) {
    std::string sub = find_object_value_for_key(config_json, "defaults");
    if (sub.empty())
        return {};
    return parse_layered_json(sub);
}

LayerContribution bundle_defaults_contribution(const std::string& config_json) {
    LayerContribution out;
    out.layer = Layer::BundleDefault;
    out.values = extract_bundle_defaults(config_json);
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

PipelineConfigResolution resolve_pipeline_config(const std::string& config_json,
                                                 const std::string& config_path,
                                                 const std::vector<std::string>& set_tokens,
                                                 const SchemaRegistry& registry) {
    PipelineConfigResolution out;

    // BundleDefault layer. Filter unknown namespaces so old bundles whose
    // clusters haven't been migrated yet don't fail-fast at load time.
    LayerContribution bundle_defaults = bundle_defaults_contribution(config_json);
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

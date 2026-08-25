/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/pipeline_plugin_loader.h"

#include "runtime/platform/dynamic_library.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

namespace trtmc {

namespace {

namespace fs = std::filesystem;

using RegisterModelPluginFn = void (*)(PipelineRegistry*);
using ModelPluginIdFn = const char* (*)();

struct ModelPluginCandidate {
    fs::path path;
    void* handle{nullptr};
    RegisterModelPluginFn register_fn{nullptr};
};

std::vector<void*>& loaded_handles() {
    static std::vector<void*> handles;
    return handles;
}

std::unordered_set<std::string>& loaded_model_ids() {
    static std::unordered_set<std::string> ids;
    return ids;
}

std::string exe_dir() {
    return internal::current_executable_path().parent_path().string();
}

void append_split_paths(std::vector<std::string>& paths, const char* raw) {
    if (raw == nullptr || raw[0] == '\0')
        return;
    std::string text(raw);
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = text.find(internal::path_list_separator(), start);
        auto item = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!item.empty())
            paths.push_back(std::move(item));
        if (end == std::string::npos)
            break;
        start = end + 1;
    }
}

void append_python_package_model_dirs(const fs::path& root, std::vector<std::string>& dirs) {
    std::error_code ec;
    for (const auto& lib_dir : {root / "lib", root / "lib64"}) {
        if (!fs::is_directory(lib_dir, ec)) {
            ec.clear();
            continue;
        }
        for (fs::directory_iterator it(lib_dir, ec), end; !ec && it != end; it.increment(ec)) {
            if (!it->is_directory(ec)) {
                ec.clear();
                continue;
            }
            const std::string name = it->path().filename().string();
            if (name.rfind("python", 0) != 0)
                continue;
            const fs::path candidate =
                it->path() / "site-packages" / "tensorrt_model_connect" / "bin";
            if (fs::is_directory(candidate, ec))
                dirs.push_back(candidate.string());
            ec.clear();
        }
        ec.clear();
    }
}

void append_installed_model_plugin_dirs(std::vector<std::string>& dirs) {
    const std::string bin_dir = exe_dir();
    if (bin_dir.empty())
        return;

    dirs.push_back(bin_dir);

    const fs::path exe_bin_dir(bin_dir);
    if (exe_bin_dir.filename() == "bin") {
        const fs::path prefix = exe_bin_dir.parent_path();
        dirs.push_back((prefix / "bin" / "trtmc" / "models").string());
        dirs.push_back((prefix / "lib" / "trtmc" / "models").string());
        dirs.push_back((prefix / "lib64" / "trtmc" / "models").string());
        append_python_package_model_dirs(prefix, dirs);
    }
}

bool strict_model_plugin_loading() {
    const char* value = std::getenv("TRTMC_MODEL_PLUGIN_STRICT");
    if (value == nullptr)
        return false;
    const std::string text(value);
    return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
}

std::vector<std::string> model_plugin_search_paths(const std::vector<std::string>& explicit_paths) {
    std::vector<std::string> paths = explicit_paths;
    append_split_paths(paths, std::getenv("TRTMC_MODEL_PLUGIN_DIR"));
    if (strict_model_plugin_loading())
        return paths;
    append_installed_model_plugin_dirs(paths);

#ifdef TRTMC_BINARY_DIR
    paths.emplace_back(TRTMC_BINARY_DIR "/models");
#endif

    paths.emplace_back(".");
    return paths;
}

std::filesystem::path plugin_path_in_dir(const std::string& dir, const std::string& model_id,
                                         const std::string& library_name) {
    std::filesystem::path root(dir);
    auto nested = root / model_id / library_name;
    if (std::filesystem::exists(nested))
        return nested;
    return root / library_name;
}

std::unordered_set<std::string> strategy_set(const std::vector<std::string>& strategies) {
    return std::unordered_set<std::string>(strategies.begin(), strategies.end());
}

std::unordered_set<std::string> expected_strategies_for_model(const std::string& model_id) {
    std::unordered_set<std::string> strategies;
    for (const auto& entry : runtime_model_plugin_index()) {
        if (entry.model_id != nullptr && model_id == entry.model_id &&
            entry.runtime_strategy != nullptr) {
            strategies.insert(entry.runtime_strategy);
        }
    }
    return strategies;
}

void close_model_plugin_candidate(ModelPluginCandidate& candidate) {
    if (candidate.handle == nullptr)
        return;
    internal::close_dynamic_library(candidate.handle);
    candidate.handle = nullptr;
}

bool model_plugin_id_matches(const fs::path& candidate, void* handle, const std::string& model_id,
                             std::vector<std::string>& errors) {
    auto* id_sym = internal::dynamic_library_symbol(handle, "trtmc_model_plugin_id");
    if (id_sym == nullptr) {
        errors.push_back(candidate.string() + ": missing trtmc_model_plugin_id");
        return false;
    }

    const char* actual_model_id = reinterpret_cast<ModelPluginIdFn>(id_sym)();
    if (actual_model_id != nullptr && model_id == actual_model_id)
        return true;

    errors.push_back(candidate.string() + ": plugin id mismatch, expected '" + model_id +
                     "' but got '" + (actual_model_id ? actual_model_id : "<null>") + "'");
    return false;
}

std::optional<ModelPluginCandidate> open_model_plugin_candidate(const fs::path& path,
                                                                const std::string& model_id,
                                                                std::vector<std::string>& errors) {
    std::string error;
    void* handle =
        internal::open_dynamic_library(path, internal::DynamicLibraryVisibility::local, &error);
    if (handle == nullptr) {
        errors.push_back(path.string() + ": " + error);
        return std::nullopt;
    }

    ModelPluginCandidate candidate{path, handle, nullptr};
    if (!model_plugin_id_matches(path, handle, model_id, errors)) {
        close_model_plugin_candidate(candidate);
        return std::nullopt;
    }

    auto* sym = internal::dynamic_library_symbol(handle, "trtmc_register_model_plugin");
    if (sym == nullptr) {
        errors.push_back(path.string() + ": missing trtmc_register_model_plugin");
        close_model_plugin_candidate(candidate);
        return std::nullopt;
    }

    candidate.register_fn = reinterpret_cast<RegisterModelPluginFn>(sym);
    return candidate;
}

std::optional<std::string>
unexpected_registered_strategy(const std::vector<std::string>& after,
                               const std::unordered_set<std::string>& before,
                               const std::unordered_set<std::string>& expected) {
    for (const auto& registered : after) {
        if (before.find(registered) != before.end())
            continue;
        if (expected.find(registered) == expected.end())
            return registered;
    }
    return std::nullopt;
}

bool register_model_plugin_candidate(ModelPluginCandidate& candidate, const std::string& model_id,
                                     const std::string& strategy,
                                     std::vector<std::string>& errors) {
    auto& registry = PipelineRegistry::instance();
    const auto before = strategy_set(registry.registered_strategies());
    const auto expected = expected_strategies_for_model(model_id);

    candidate.register_fn(&registry);

    const auto after = registry.registered_strategies();
    if (const auto unexpected = unexpected_registered_strategy(after, before, expected)) {
        errors.push_back(candidate.path.string() + ": plugin '" + model_id +
                         "' registered unexpected strategy '" + *unexpected + "'");
        close_model_plugin_candidate(candidate);
        return false;
    }

    if (registry.lookup(strategy) != nullptr)
        return true;

    errors.push_back(candidate.path.string() + ": plugin '" + model_id +
                     "' did not register requested strategy '" + strategy + "'");
    close_model_plugin_candidate(candidate);
    return false;
}

std::string alias_string(const char* value) {
    return value == nullptr ? std::string{} : std::string(value);
}

std::string json_field_value_window(const std::string& config_text, const std::string& key) {
    if (key.empty() || key == "_")
        return "";
    const auto pos = config_text.find("\"" + key + "\"");
    if (pos == std::string::npos)
        return "";
    const auto colon = config_text.find(':', pos);
    if (colon == std::string::npos)
        return "";
    return config_text.substr(colon + 1, 160);
}

bool json_field_is_truthy(const std::string& config_text, const std::string& key) {
    const auto value = json_field_value_window(config_text, key);
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return false;
    return value.compare(first, 4, "true") == 0 || value[first] == '1';
}

bool legacy_alias_matches(const LegacyStrategyAlias& alias, const std::string& config_text) {
    const std::string op = alias_string(alias.match_op);
    const std::string key = alias_string(alias.config_key);
    const std::string value = alias_string(alias.match_value);

    if (op == "default")
        return true;
    if (op == "truthy")
        return json_field_is_truthy(config_text, key);
    if (op == "not_truthy")
        return !json_field_is_truthy(config_text, key);
    if (op == "contains")
        return json_field_value_window(config_text, key).find(value) != std::string::npos;
    if (op == "equals") {
        const auto window = json_field_value_window(config_text, key);
        return window.find("\"" + value + "\"") != std::string::npos ||
               window.find(value) != std::string::npos;
    }
    return false;
}

[[noreturn]] void throw_load_error(const std::string& model_id, const std::string& library_name,
                                   const std::vector<std::string>& paths,
                                   const std::vector<std::string>& errors) {
    std::ostringstream msg;
    msg << "Unable to load model plugin " << model_id << " (" << library_name << "). Searched:";
    for (const auto& path : paths)
        msg << "\n  " << path;
    if (!errors.empty()) {
        msg << "\nLoad errors:";
        for (const auto& error : errors)
            msg << "\n  " << error;
    }
    throw std::runtime_error(msg.str());
}

} // namespace

std::optional<std::string> model_plugin_id_for_strategy(const std::string& strategy) {
    for (const auto& entry : runtime_model_plugin_index()) {
        if (entry.runtime_strategy != nullptr && strategy == entry.runtime_strategy)
            return std::string(entry.model_id);
    }
    return std::nullopt;
}

std::string model_plugin_library_name(const std::string& model_id) {
    for (const auto& entry : runtime_model_plugin_index()) {
        if (entry.model_id != nullptr && model_id == entry.model_id &&
            entry.library_name != nullptr)
            return std::string(entry.library_name);
    }
    return internal::dynamic_library_filename("trtmc_model_" + model_id);
}

std::optional<std::string> legacy_runtime_strategy_alias_target(const std::string& strategy,
                                                                const std::string& config_text) {
    const LegacyStrategyAlias* fallback = nullptr;
    for (const auto& alias : legacy_runtime_strategy_alias_index()) {
        if (strategy != alias_string(alias.legacy_strategy))
            continue;
        if (alias_string(alias.match_op) == "default") {
            if (fallback == nullptr)
                fallback = &alias;
            continue;
        }
        if (legacy_alias_matches(alias, config_text))
            return alias_string(alias.target_strategy);
    }
    if (fallback != nullptr)
        return alias_string(fallback->target_strategy);
    return std::nullopt;
}

void load_model_plugin_for_strategy(const std::string& strategy,
                                    const std::vector<std::string>& search_paths) {
    const auto model_id = model_plugin_id_for_strategy(strategy);
    if (!model_id)
        throw std::runtime_error("No plugin registered for runtime_strategy: " + strategy);

    if (loaded_model_ids().find(*model_id) != loaded_model_ids().end())
        return;

    const auto library_name = model_plugin_library_name(*model_id);
    const auto paths = model_plugin_search_paths(search_paths);
    if (strict_model_plugin_loading() && paths.empty()) {
        throw std::runtime_error(
            "TRTMC_MODEL_PLUGIN_STRICT requires an explicit model plugin search path");
    }
    std::vector<std::string> errors;

    for (const auto& dir : paths) {
        const auto candidate = plugin_path_in_dir(dir, *model_id, library_name);
        if (!std::filesystem::exists(candidate))
            continue;

        auto plugin = open_model_plugin_candidate(candidate, *model_id, errors);
        if (!plugin)
            continue;

        if (!register_model_plugin_candidate(*plugin, *model_id, strategy, errors))
            continue;

        loaded_handles().push_back(plugin->handle);
        loaded_model_ids().insert(*model_id);
        return;
    }

    throw_load_error(*model_id, library_name, paths, errors);
}

} // namespace trtmc

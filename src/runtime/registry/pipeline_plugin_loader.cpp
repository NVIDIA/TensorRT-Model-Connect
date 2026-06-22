#include "trtmc/runtime/pipeline_plugin_loader.h"

#include "trtmc/runtime/pipeline_registry.h"

#include <dlfcn.h>

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <unordered_set>
#include <vector>

namespace trtmc {

namespace {

namespace fs = std::filesystem;

using RegisterModelPluginFn = void (*)(PipelineRegistry*);
using ModelPluginIdFn = const char* (*)();

std::vector<void*>& loaded_handles() {
    static std::vector<void*> handles;
    return handles;
}

std::unordered_set<std::string>& loaded_model_ids() {
    static std::unordered_set<std::string> ids;
    return ids;
}

std::string exe_dir() {
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0)
        return "";
    buf[len] = '\0';
    std::string path(buf);
    auto pos = path.rfind('/');
    return (pos != std::string::npos) ? path.substr(0, pos) : "";
}

void append_split_paths(std::vector<std::string>& paths, const char* raw) {
    if (raw == nullptr || raw[0] == '\0')
        return;
    std::string text(raw);
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = text.find(':', start);
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
        dirs.push_back((prefix / "lib" / "trtmc" / "models").string());
        dirs.push_back((prefix / "lib64" / "trtmc" / "models").string());
        append_python_package_model_dirs(prefix, dirs);
    }
}

std::vector<std::string> model_plugin_search_paths(const std::vector<std::string>& explicit_paths) {
    std::vector<std::string> paths = explicit_paths;
    append_split_paths(paths, std::getenv("TRTMC_MODEL_PLUGIN_DIR"));
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
        if (entry.model_id != nullptr && model_id == entry.model_id && entry.library_name != nullptr)
            return std::string(entry.library_name);
    }
    return "libtrtmc_model_" + model_id + ".so";
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
    std::vector<std::string> errors;

    for (const auto& dir : paths) {
        const auto candidate = plugin_path_in_dir(dir, *model_id, library_name);
        if (!std::filesystem::exists(candidate))
            continue;

        dlerror();
        void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            const char* err = dlerror();
            errors.push_back(candidate.string() + ": " + (err ? err : "unknown dlopen error"));
            continue;
        }

        dlerror();
        auto* id_sym = dlsym(handle, "trtmc_model_plugin_id");
        const char* id_err = dlerror();
        if (id_err != nullptr || id_sym == nullptr) {
            errors.push_back(candidate.string() + ": missing trtmc_model_plugin_id");
            dlclose(handle);
            continue;
        }
        const char* actual_model_id = reinterpret_cast<ModelPluginIdFn>(id_sym)();
        if (actual_model_id == nullptr || *model_id != actual_model_id) {
            errors.push_back(candidate.string() + ": plugin id mismatch, expected '" + *model_id +
                             "' but got '" + (actual_model_id ? actual_model_id : "<null>") +
                             "'");
            dlclose(handle);
            continue;
        }

        dlerror();
        auto* sym = dlsym(handle, "trtmc_register_model_plugin");
        const char* err = dlerror();
        if (err != nullptr || sym == nullptr) {
            errors.push_back(candidate.string() + ": missing trtmc_register_model_plugin");
            dlclose(handle);
            continue;
        }

        auto& registry = PipelineRegistry::instance();
        const auto before = strategy_set(registry.registered_strategies());
        const auto expected = expected_strategies_for_model(*model_id);

        reinterpret_cast<RegisterModelPluginFn>(sym)(&PipelineRegistry::instance());
        const auto after = registry.registered_strategies();
        bool valid_registration = true;
        for (const auto& registered : after) {
            if (before.find(registered) != before.end())
                continue;
            if (expected.find(registered) == expected.end()) {
                errors.push_back(candidate.string() + ": plugin '" + *model_id +
                                 "' registered unexpected strategy '" + registered + "'");
                valid_registration = false;
                break;
            }
        }
        if (!valid_registration) {
            dlclose(handle);
            continue;
        }
        if (registry.lookup(strategy) == nullptr) {
            errors.push_back(candidate.string() + ": plugin '" + *model_id +
                             "' did not register requested strategy '" + strategy + "'");
            dlclose(handle);
            continue;
        }

        loaded_handles().push_back(handle);
        loaded_model_ids().insert(*model_id);
        return;
    }

    throw_load_error(*model_id, library_name, paths, errors);
}

} // namespace trtmc

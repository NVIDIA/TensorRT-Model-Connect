/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/backend/trt_version.h"

#include "runtime/platform/dynamic_library.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

namespace trtmc {

namespace {

using VersionFn = int32_t (*)() noexcept;
namespace fs = std::filesystem;

struct LoadedNvinfer {
    std::string path;
    TrtVersion version;
};

struct TrtLibrarySearchStep {
    std::optional<TrtLibraryMatch> match;
    bool definitive = false;
};

bool is_digit(char c) {
    return std::isdigit(static_cast<unsigned char>(c)) != 0;
}

bool consume_digits(const std::string& text, std::size_t* pos) {
    const std::size_t start = *pos;
    while (*pos < text.size() && is_digit(text[*pos]))
        ++(*pos);
    return *pos > start;
}

std::size_t first_digit_pos(const std::string& text) {
    std::size_t pos = 0;
    while (pos < text.size() && !is_digit(text[pos]))
        ++pos;
    return pos;
}

std::optional<int> parse_int_component(const std::string& text, std::size_t* pos) {
    if (*pos >= text.size() || !is_digit(text[*pos]))
        return std::nullopt;

    const std::size_t start = *pos;
    consume_digits(text, pos);
    try {
        return std::stoi(text.substr(start, *pos - start));
    } catch (...) {
        return std::nullopt;
    }
}

bool consume_dot_separator(const std::string& text, std::size_t* pos) {
    if (*pos >= text.size() || text[*pos] != '.')
        return false;
    ++(*pos);
    return true;
}

std::string exe_dir() {
    return internal::current_executable_path().parent_path().string();
}

std::string join_path(const std::string& dir, const std::string& filename) {
    if (dir.empty())
        return filename;
    return (fs::path(dir) / filename).string();
}

std::vector<std::string> split_path_list(const char* value) {
    std::vector<std::string> out;
    if (value == nullptr || value[0] == '\0')
        return out;

    std::string text(value);
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = text.find(internal::path_list_separator(), start);
        std::string item =
            text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!item.empty())
            out.push_back(std::move(item));
        if (end == std::string::npos)
            break;
        start = end + 1;
    }
    return out;
}

std::vector<std::string> unique_preserving_order(std::vector<std::string> values) {
    std::vector<std::string> out;
    std::unordered_set<std::string> seen;
    for (auto& value : values) {
        if (value.empty())
            continue;
        if (seen.insert(value).second)
            out.push_back(std::move(value));
    }
    return out;
}

std::optional<fs::path> python_tensorrt_lib_dir(const fs::directory_entry& entry,
                                                std::error_code& ec) {
    if (!entry.is_directory(ec))
        return std::nullopt;
    const std::string name = entry.path().filename().string();
    if (name.rfind("python", 0) != 0)
        return std::nullopt;
    const fs::path candidate = entry.path() / "site-packages" / "tensorrt_libs";
    if (!fs::is_directory(candidate, ec))
        return std::nullopt;
    return candidate;
}

void append_python_lib_tensorrt_dirs(const fs::path& lib_dir, std::vector<std::string>& dirs,
                                     std::error_code& ec) {
    if (!fs::is_directory(lib_dir, ec))
        return;
    for (const auto& entry : fs::directory_iterator(lib_dir, ec)) {
        if (ec)
            break;
        if (auto candidate = python_tensorrt_lib_dir(entry, ec))
            dirs.push_back(candidate->string());
    }
}

void append_python_tensorrt_lib_dirs(const char* root_env, std::vector<std::string>& dirs) {
    if (root_env == nullptr || root_env[0] == '\0')
        return;

    std::error_code ec;
    const fs::path root(root_env);
#if defined(_WIN32)
    const fs::path windows_candidate = root / "Lib" / "site-packages" / "tensorrt_libs";
    if (fs::is_directory(windows_candidate, ec))
        dirs.push_back(windows_candidate.string());
    ec.clear();
#endif
    append_python_lib_tensorrt_dirs(root / "lib", dirs, ec);
    append_python_lib_tensorrt_dirs(root / "lib64", dirs, ec);
}

void append_packaged_tensorrt_lib_dir(std::vector<std::string>& dirs) {
    const std::string bin_dir = exe_dir();
    if (bin_dir.empty())
        return;

    std::error_code ec;
    const fs::path site_packages = fs::path(bin_dir).parent_path().parent_path();
    const fs::path candidate = site_packages / "tensorrt_libs";
    if (fs::is_directory(candidate, ec))
        dirs.push_back(candidate.string());
}

void append_installed_prefix_tensorrt_lib_dirs(std::vector<std::string>& dirs) {
    const std::string bin_dir = exe_dir();
    if (bin_dir.empty())
        return;

    const fs::path exe_bin_dir(bin_dir);
    if (exe_bin_dir.filename() != "bin")
        return;

    const std::string prefix = exe_bin_dir.parent_path().string();
    append_python_tensorrt_lib_dirs(prefix.c_str(), dirs);
}

std::optional<TrtVersion> version_from_symbol_scope(void* handle, const std::string& source) {
    auto major_fn = reinterpret_cast<VersionFn>(
        internal::dynamic_library_symbol(handle, "getInferLibMajorVersion"));
    auto minor_fn = reinterpret_cast<VersionFn>(
        internal::dynamic_library_symbol(handle, "getInferLibMinorVersion"));
    auto patch_fn = reinterpret_cast<VersionFn>(
        internal::dynamic_library_symbol(handle, "getInferLibPatchVersion"));
    auto build_fn = reinterpret_cast<VersionFn>(
        internal::dynamic_library_symbol(handle, "getInferLibBuildVersion"));
    if (major_fn == nullptr || minor_fn == nullptr) {
        return std::nullopt;
    }

    TrtVersion version;
    version.major = major_fn();
    version.minor = minor_fn();
    version.patch = patch_fn ? patch_fn() : -1;
    version.build = build_fn ? build_fn() : -1;
    version.source = source;
    return version;
}

bool is_nvinfer_library_name(const fs::path& path) {
    if (path.empty())
        return false;

    std::string name = path.filename().string();
#if defined(_WIN32)
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return name == "nvinfer.dll" || (name.rfind("nvinfer_", 0) == 0 && name.size() > 12 &&
                                     name.substr(name.size() - 4) == ".dll");
#else
    return name == "libnvinfer.so" || name.rfind("libnvinfer.so.", 0) == 0;
#endif
}

std::vector<LoadedNvinfer> loaded_nvinfer_versions() {
    std::vector<LoadedNvinfer> loaded;
    for (const fs::path& path : internal::loaded_dynamic_library_paths()) {
        if (!is_nvinfer_library_name(path))
            continue;
        auto handle = internal::open_dynamic_library(path);
        if (handle == nullptr)
            continue;
        auto version = version_from_symbol_scope(handle, path.string());
        internal::close_dynamic_library(handle);
        if (version)
            loaded.push_back(LoadedNvinfer{path.string(), *version});
    }
    return loaded;
}

std::vector<std::string> nvinfer_candidates(const std::vector<std::string>& search_dirs) {
#if defined(_WIN32)
    const std::vector<std::string> names = {
        "nvinfer.dll",
        "nvinfer_11.dll",
        "nvinfer_10.dll",
    };
#else
    const std::vector<std::string> names = {
        "libnvinfer.so",
        "libnvinfer.so.11",
        "libnvinfer.so.10",
    };
#endif

    std::vector<std::string> dirs;
    const std::string bin_dir = exe_dir();
    if (!bin_dir.empty())
        dirs.push_back(bin_dir);
    append_packaged_tensorrt_lib_dir(dirs);
    append_installed_prefix_tensorrt_lib_dirs(dirs);
    dirs.insert(dirs.end(), search_dirs.begin(), search_dirs.end());
    if (const char* trt_dir = std::getenv("TRTMC_TRT_LIBRARY_DIR"))
        dirs.push_back(trt_dir);
    append_python_tensorrt_lib_dirs(std::getenv("VIRTUAL_ENV"), dirs);
    append_python_tensorrt_lib_dirs(std::getenv("CONDA_PREFIX"), dirs);
    auto loader_dirs =
        split_path_list(std::getenv(internal::dynamic_library_search_path_environment()));
    dirs.insert(dirs.end(), loader_dirs.begin(), loader_dirs.end());
    dirs = unique_preserving_order(std::move(dirs));

    std::vector<std::string> candidates;
    for (const auto& dir : dirs) {
        for (const auto& name : names)
            candidates.push_back(join_path(dir, name));
    }
    for (const auto& name : names)
        candidates.push_back(name);
    return unique_preserving_order(std::move(candidates));
}

bool looks_like_versioned_trt_backend(const std::string& backend_name) {
    constexpr const char* prefix = "trt_";
    if (backend_name.rfind(prefix, 0) != 0)
        return false;
    if (backend_name == "trt_rtx")
        return false;

    std::size_t pos = std::strlen(prefix);
    if (!consume_digits(backend_name, &pos))
        return false;
    if (pos >= backend_name.size() || backend_name[pos] != '_')
        return false;
    ++pos;
    return consume_digits(backend_name, &pos) && pos == backend_name.size();
}

void append_diagnostic(std::string* diagnostics, std::string message) {
    if (diagnostics == nullptr)
        return;
    *diagnostics += "  " + std::move(message) + "\n";
}

void append_dynamic_load_error(std::string* diagnostics, const std::string& candidate,
                               const std::string& error) {
    append_diagnostic(diagnostics,
                      candidate + ": " + (error.empty() ? "unknown dynamic-loader error" : error));
}

TrtLibrarySearchStep match_loaded_trt_library(const TrtVersion& required_version,
                                              std::string* diagnostics) {
    const auto loaded = loaded_nvinfer_versions();
    if (loaded.empty())
        return {};

    std::optional<TrtLibraryMatch> match;
    bool all_match = true;
    for (const auto& lib : loaded) {
        const bool abi_matches = trt_abi_matches(required_version, lib.version);
        std::string message = "already loaded " + lib.path + ": TensorRT " +
                              format_trt_version(lib.version) + " (ABI " +
                              trt_abi_string(lib.version) + ")";
        if (!abi_matches) {
            message += " does not match required ABI " + trt_abi_string(required_version);
            all_match = false;
        } else if (!match) {
            match = TrtLibraryMatch{lib.version, lib.path, true};
        }
        append_diagnostic(diagnostics, std::move(message));
    }
    return {all_match ? match : std::nullopt, true};
}

std::optional<TrtVersion> version_from_process_symbol_scope() {
    auto major_fn = reinterpret_cast<VersionFn>(
        internal::dynamic_library_symbol_in_process("getInferLibMajorVersion"));
    auto minor_fn = reinterpret_cast<VersionFn>(
        internal::dynamic_library_symbol_in_process("getInferLibMinorVersion"));
    auto patch_fn = reinterpret_cast<VersionFn>(
        internal::dynamic_library_symbol_in_process("getInferLibPatchVersion"));
    auto build_fn = reinterpret_cast<VersionFn>(
        internal::dynamic_library_symbol_in_process("getInferLibBuildVersion"));
    if (major_fn == nullptr || minor_fn == nullptr)
        return std::nullopt;
    TrtVersion version;
    version.major = major_fn();
    version.minor = minor_fn();
    version.patch = patch_fn ? patch_fn() : -1;
    version.build = build_fn ? build_fn() : -1;
    version.source = "process symbol scope";
    return version;
}

TrtLibrarySearchStep match_process_scope_trt_library(const TrtVersion& required_version,
                                                     std::string* diagnostics) {
    auto default_version = version_from_process_symbol_scope();
    if (!default_version)
        return {};
    if (trt_abi_matches(required_version, *default_version))
        return {TrtLibraryMatch{*default_version, "", true}, true};

    append_diagnostic(diagnostics, "process symbol scope: already loaded TensorRT " +
                                       format_trt_version(*default_version) + " (ABI " +
                                       trt_abi_string(*default_version) +
                                       ") does not match required ABI " +
                                       trt_abi_string(required_version));
    return {std::nullopt, true};
}

std::optional<TrtLibraryMatch>
match_candidate_trt_libraries(const TrtVersion& required_version,
                              const std::vector<std::string>& search_dirs,
                              std::string* diagnostics) {
    for (const auto& candidate : nvinfer_candidates(search_dirs)) {
        std::string error;
        void* handle = internal::open_dynamic_library(
            fs::path(candidate), internal::DynamicLibraryVisibility::local, &error);
        if (!handle) {
            append_dynamic_load_error(diagnostics, candidate, error);
            continue;
        }

        auto version = version_from_symbol_scope(handle, candidate);
        internal::close_dynamic_library(handle);
        if (!version) {
            append_diagnostic(diagnostics, candidate + ": missing TensorRT version symbols");
            continue;
        }

        if (trt_abi_matches(required_version, *version))
            return TrtLibraryMatch{*version, candidate, false};

        append_diagnostic(diagnostics, candidate + ": TensorRT " + format_trt_version(*version) +
                                           " (ABI " + trt_abi_string(*version) +
                                           ") does not match required ABI " +
                                           trt_abi_string(required_version));
    }
    return std::nullopt;
}

} // namespace

std::optional<TrtVersion> parse_trt_version(const std::string& text) {
    std::size_t pos = first_digit_pos(text);
    if (pos == text.size())
        return std::nullopt;

    std::vector<int> components;
    while (pos < text.size() && components.size() < 4) {
        auto component = parse_int_component(text, &pos);
        if (!component)
            break;
        components.push_back(*component);
        if (!consume_dot_separator(text, &pos))
            break;
    }

    if (components.size() < 2)
        return std::nullopt;

    TrtVersion version;
    version.major = components[0];
    version.minor = components[1];
    version.patch = components.size() > 2 ? components[2] : -1;
    version.build = components.size() > 3 ? components[3] : -1;
    return version;
}

std::optional<TrtVersion> parse_trt_abi_tag(const std::string& text) {
    if (auto parsed = parse_trt_version(text))
        return parsed;

    const std::string prefix = "trt_";
    std::string tag = text;
    if (tag.rfind(prefix, 0) == 0)
        tag = tag.substr(prefix.size());

    const std::size_t sep = tag.find('_');
    if (sep == std::string::npos)
        return std::nullopt;

    try {
        TrtVersion version;
        version.major = std::stoi(tag.substr(0, sep));
        version.minor = std::stoi(tag.substr(sep + 1));
        return version;
    } catch (...) {
        return std::nullopt;
    }
}

std::string format_trt_version(const TrtVersion& version) {
    if (version.major < 0 || version.minor < 0)
        return "unknown";
    std::ostringstream oss;
    oss << version.major << "." << version.minor;
    if (version.patch >= 0)
        oss << "." << version.patch;
    if (version.build >= 0)
        oss << "." << version.build;
    return oss.str();
}

std::string trt_abi_string(const TrtVersion& version) {
    if (version.major < 0 || version.minor < 0)
        return "";
    return std::to_string(version.major) + "." + std::to_string(version.minor);
}

std::string trt_abi_suffix(const TrtVersion& version) {
    if (version.major < 0 || version.minor < 0)
        return "";
    return std::to_string(version.major) + "_" + std::to_string(version.minor);
}

std::string trt_backend_name_for_abi(const TrtVersion& version) {
    const std::string suffix = trt_abi_suffix(version);
    return suffix.empty() ? "trt" : "trt_" + suffix;
}

bool trt_abi_matches(const TrtVersion& lhs, const TrtVersion& rhs) {
    return lhs.major >= 0 && lhs.minor >= 0 && lhs.major == rhs.major && lhs.minor == rhs.minor;
}

bool is_standard_trt_backend_name(const std::string& backend_name) {
    return backend_name == "trt" || looks_like_versioned_trt_backend(backend_name);
}

std::optional<TrtVersion> detect_installed_trt_version(const std::vector<std::string>& search_dirs,
                                                       std::string* diagnostics) {
    if (diagnostics)
        diagnostics->clear();

    const auto loaded = loaded_nvinfer_versions();
    if (!loaded.empty())
        return loaded.front().version;

    if (auto loaded = version_from_process_symbol_scope())
        return loaded;

    for (const auto& candidate : nvinfer_candidates(search_dirs)) {
        std::string error;
        void* handle = internal::open_dynamic_library(
            fs::path(candidate), internal::DynamicLibraryVisibility::local, &error);
        if (!handle) {
            append_dynamic_load_error(diagnostics, candidate, error);
            continue;
        }

        auto version = version_from_symbol_scope(handle, candidate);
        internal::close_dynamic_library(handle);
        if (version)
            return version;

        if (diagnostics)
            *diagnostics += "  " + candidate + ": missing TensorRT version symbols\n";
    }

    return std::nullopt;
}

std::optional<TrtLibraryMatch>
find_trt_library_for_version(const TrtVersion& required_version,
                             const std::vector<std::string>& search_dirs,
                             std::string* diagnostics) {
    if (diagnostics)
        diagnostics->clear();

    const auto loaded = match_loaded_trt_library(required_version, diagnostics);
    if (loaded.definitive)
        return loaded.match;

    const auto default_scope = match_process_scope_trt_library(required_version, diagnostics);
    if (default_scope.definitive)
        return default_scope.match;

    return match_candidate_trt_libraries(required_version, search_dirs, diagnostics);
}

std::vector<std::string>
trt_backend_candidates(const std::string& backend_name,
                       const std::optional<TrtVersion>& required_version,
                       const std::optional<TrtVersion>& installed_version) {
    const std::string logical_name = backend_name.empty() ? "trt" : backend_name;
    if (!is_standard_trt_backend_name(logical_name))
        return {logical_name};
    if (looks_like_versioned_trt_backend(logical_name))
        return {logical_name};

    std::vector<std::string> candidates;
    if (required_version) {
        candidates.push_back(trt_backend_name_for_abi(*required_version));
    } else if (installed_version) {
        candidates.push_back(trt_backend_name_for_abi(*installed_version));
    }
    candidates.push_back("trt");
    return unique_preserving_order(std::move(candidates));
}

} // namespace trtmc

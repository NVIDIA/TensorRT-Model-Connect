/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_h3/native_plugin_loader.h"

#include "bundle/bundle_view.h"
#include "utils/json_helpers.h"
#include "utils/sha256.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#if !defined(_WIN32)
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace trtmc {
namespace {

constexpr const char* kPluginSection = "minimax_h3_native_plugin_so";
constexpr const char* kPluginIdentity = "trtmc.minimax_h3.native_plugin:aten-ops:1";
constexpr std::uint32_t kPluginAbi = 1U;

bool is_lower_sha256(const std::string& value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

std::string sha256_bytes(const std::vector<char>& bytes) {
    internal::Sha256 digest;
    if (!bytes.empty())
        digest.update(bytes.data(), bytes.size());
    return digest.hex_digest();
}

#if !defined(_WIN32)

std::filesystem::path private_cache_directory() {
    const char* configured = std::getenv("TRTMC_MINIMAX_H3_NATIVE_PLUGIN_CACHE_DIR");
    const auto directory = configured != nullptr && configured[0] != '\0'
                               ? std::filesystem::path(configured)
                               : std::filesystem::temp_directory_path() /
                                     ("trtmc-minimax-h3-native-plugin-" +
                                      std::to_string(static_cast<unsigned long>(geteuid())));
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error)
        throw std::runtime_error("Unable to create MiniMax-H3 native plugin cache: " +
                                 error.message());
    struct stat status{};
    if (lstat(directory.c_str(), &status) != 0 || !S_ISDIR(status.st_mode) ||
        status.st_uid != geteuid()) {
        throw std::runtime_error("MiniMax-H3 native plugin cache is not a private owned directory");
    }
    if (chmod(directory.c_str(), S_IRWXU) != 0)
        throw std::runtime_error("Unable to secure MiniMax-H3 native plugin cache");
    return directory;
}

bool cache_matches(const std::filesystem::path& output, const std::vector<char>& bytes) {
    struct stat status{};
    if (lstat(output.c_str(), &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_uid != geteuid() || (status.st_mode & 0077) != 0 ||
        static_cast<std::uintmax_t>(status.st_size) != bytes.size()) {
        return false;
    }
    std::ifstream stream(output, std::ios::binary);
    if (!stream)
        return false;
    std::vector<char> cached(bytes.size());
    if (!cached.empty())
        stream.read(cached.data(), static_cast<std::streamsize>(cached.size()));
    return stream.good() && cached == bytes;
}

void write_all(int descriptor, const std::vector<char>& bytes) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto count = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0)
            throw std::runtime_error("Unable to write MiniMax-H3 native plugin cache file");
        offset += static_cast<std::size_t>(count);
    }
}

void publish_cache_file(const std::filesystem::path& output, const std::vector<char>& bytes) {
    if (cache_matches(output, bytes))
        return;
    const auto temporary =
        std::filesystem::path(output.string() + ".tmp." + std::to_string(getpid()));
    struct stat stale{};
    if (lstat(temporary.c_str(), &stale) == 0) {
        if (!S_ISREG(stale.st_mode) || stale.st_uid != geteuid() ||
            ::unlink(temporary.c_str()) != 0) {
            throw std::runtime_error("Unsafe stale MiniMax-H3 native plugin cache file");
        }
    }
    const int descriptor =
        ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (descriptor < 0)
        throw std::runtime_error("Unable to create MiniMax-H3 native plugin cache file: " +
                                 std::string(std::strerror(errno)));
    bool descriptor_open = true;
    try {
        write_all(descriptor, bytes);
        if (::fsync(descriptor) != 0)
            throw std::runtime_error("Unable to sync MiniMax-H3 native plugin cache file");
        const int close_result = ::close(descriptor);
        descriptor_open = false;
        if (close_result != 0)
            throw std::runtime_error("Unable to close MiniMax-H3 native plugin cache file");
    } catch (...) {
        if (descriptor_open)
            ::close(descriptor);
        ::unlink(temporary.c_str());
        throw;
    }
    if (::rename(temporary.c_str(), output.c_str()) != 0) {
        ::unlink(temporary.c_str());
        if (!cache_matches(output, bytes))
            throw std::runtime_error("Unable to publish MiniMax-H3 native plugin cache file");
    }
    if (!cache_matches(output, bytes))
        throw std::runtime_error("Published MiniMax-H3 native plugin cache file is invalid");
}

std::filesystem::path materialize_plugin(const PipelineContext& ctx, std::string& plugin_sha256) {
    const auto* bytes = find_section(ctx.bundle, kPluginSection);
    if (bytes == nullptr || bytes->empty())
        throw std::runtime_error("MiniMax-H3 Ref2VA bundle is missing its native plugin DSO");
    const std::string assets = extract_json_object_text(ctx.config_json, "asset_sha256");
    const std::string expected = extract_json_string(assets, kPluginSection, "");
    if (!is_lower_sha256(expected))
        throw std::runtime_error("MiniMax-H3 native plugin has an invalid expected SHA256");
    plugin_sha256 = sha256_bytes(*bytes);
    if (plugin_sha256 != expected)
        throw std::runtime_error("MiniMax-H3 native plugin SHA256 does not match bundle config");
    const auto output =
        private_cache_directory() / ("libtrtmc_minimax_h3_native_plugin_" + plugin_sha256 + ".so");
    publish_cache_file(output, *bytes);
    return output;
}

template <typename Function>
Function load_symbol(void* handle, const char* name) {
    dlerror();
    void* symbol = dlsym(handle, name);
    const char* message = dlerror();
    if (message != nullptr || symbol == nullptr)
        throw std::runtime_error(std::string("MiniMax-H3 native plugin is missing ") + name);
    return reinterpret_cast<Function>(symbol);
}

void validate_plugin_identity(void* handle) {
    using StringFunction = const char* (*)();
    using AbiFunction = std::uint32_t (*)();
    const char* identity =
        load_symbol<StringFunction>(handle, "trtmc_minimax_h3_native_plugin_identity")();
    if (identity == nullptr || std::string(identity) != kPluginIdentity)
        throw std::runtime_error("MiniMax-H3 native plugin identity mismatch");
    if (load_symbol<AbiFunction>(handle, "trtmc_minimax_h3_native_plugin_abi_version")() !=
        kPluginAbi)
        throw std::runtime_error("MiniMax-H3 native plugin ABI mismatch");
    const char* build_identity =
        load_symbol<StringFunction>(handle, "trtmc_minimax_h3_native_plugin_build_identity")();
    if (build_identity == nullptr || build_identity[0] == '\0')
        throw std::runtime_error("MiniMax-H3 native plugin has an empty build identity");
    using RegistryFunction = bool (*)();
    if (!load_symbol<RegistryFunction>(handle, "trtmc_minimax_h3_native_plugin_registry_matches")())
        throw std::runtime_error(
            "MiniMax-H3 native plugin creators do not own their TensorRT registry entries");
}

#endif

} // namespace

void load_minimax_h3_native_plugin(const PipelineContext& ctx) {
#if defined(_WIN32)
    (void)ctx;
    throw std::runtime_error(
        "MiniMax-H3 Ref2VA ATen plugins are currently qualified only on Linux");
#else
    static std::mutex mutex;
    static std::string loaded_sha256;
    static void* loaded_handle = nullptr;
    static void* failed_handle = nullptr;
    const std::lock_guard<std::mutex> lock(mutex);
    if (failed_handle != nullptr)
        throw std::runtime_error("A previous MiniMax-H3 native plugin load poisoned this process");
    std::string plugin_sha256;
    const auto path = materialize_plugin(ctx, plugin_sha256);
    if (loaded_handle != nullptr) {
        if (loaded_sha256 != plugin_sha256)
            throw std::runtime_error(
                "A different MiniMax-H3 native plugin is already loaded in this process");
        return;
    }
    dlerror();
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) {
        const char* message = dlerror();
        throw std::runtime_error(
            std::string("Unable to load MiniMax-H3 native plugin and its libtorch dependencies: ") +
            (message != nullptr ? message : path.string()));
    }
    try {
        validate_plugin_identity(handle);
    } catch (...) {
        // Creator registration runs during dlopen and TensorRT does not
        // deregister those objects on dlclose. Keep the rejected DSO mapped
        // and poison future loads instead of leaving dangling registry state.
        failed_handle = handle;
        throw;
    }
    loaded_sha256 = plugin_sha256;
    loaded_handle = handle;
#endif
}

} // namespace trtmc

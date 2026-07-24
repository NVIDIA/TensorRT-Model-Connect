/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/backend/backend_loader.h"

#include "runtime/backend/prebound_backend.h"
#include "runtime/backend/runtime_memory_backend.h"

#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc {

IPreboundBackend::~IPreboundBackend() = default;
IRuntimeMemoryEngineIntrospectionV1::~IRuntimeMemoryEngineIntrospectionV1() = default;
IRuntimeMemoryTransferLedgerV1::~IRuntimeMemoryTransferLedgerV1() = default;
IRuntimeMemoryModuleV1::~IRuntimeMemoryModuleV1() = default;
IRuntimeMemoryBackendV1::~IRuntimeMemoryBackendV1() = default;

RuntimeMemoryTransferDeltaV1
runtime_memory_transfer_delta(const RuntimeMemoryTransferSnapshotV1& before,
                              const RuntimeMemoryTransferSnapshotV1& after) {
    if (before.api_version != kRuntimeMemoryBackendApiVersionCurrent ||
        after.api_version != kRuntimeMemoryBackendApiVersionCurrent ||
        before.struct_size != sizeof(RuntimeMemoryTransferSnapshotV1) ||
        after.struct_size != sizeof(RuntimeMemoryTransferSnapshotV1)) {
        throw std::invalid_argument("runtime-memory transfer snapshot ABI mismatch");
    }
    if (after.event_sequence < before.event_sequence)
        throw std::logic_error("runtime-memory transfer event sequence moved backwards");

    std::unordered_map<std::string, RuntimeMemoryTransferCounterV1> baseline;
    baseline.reserve(before.counters.size());
    for (const auto& counter : before.counters)
        baseline.emplace(counter.tensor_name, counter);

    RuntimeMemoryTransferDeltaV1 delta;
    for (const auto& current : after.counters) {
        const auto found = baseline.find(current.tensor_name);
        const RuntimeMemoryTransferCounterV1 empty{};
        const auto& previous = found == baseline.end() ? empty : found->second;
        if (current.device_to_host_bytes < previous.device_to_host_bytes ||
            current.device_to_device_bytes < previous.device_to_device_bytes ||
            current.device_to_host_events < previous.device_to_host_events ||
            current.device_to_device_events < previous.device_to_device_events) {
            throw std::logic_error("runtime-memory transfer counter moved backwards for " +
                                   current.tensor_name);
        }
        if (!(current.runtime_kv_binding || previous.runtime_kv_binding))
            continue;
        delta.runtime_kv_device_to_host_bytes +=
            current.device_to_host_bytes - previous.device_to_host_bytes;
        delta.runtime_kv_device_to_device_bytes +=
            current.device_to_device_bytes - previous.device_to_device_bytes;
        delta.runtime_kv_device_to_host_events +=
            current.device_to_host_events - previous.device_to_host_events;
        delta.runtime_kv_device_to_device_events +=
            current.device_to_device_events - previous.device_to_device_events;
    }
    return delta;
}

namespace {

namespace fs = std::filesystem;

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

struct CachedBackend {
    void* dl_handle{nullptr};
    IBackend* backend{nullptr};
    BackendLoadMetadata metadata;
};

std::mutex g_mu;
std::unordered_map<std::string, CachedBackend> g_cache;
std::unordered_map<std::string, void*> g_preloaded_dependencies;

void cleanup_backends();

void register_cleanup_once() {
    static bool registered = false;
    if (!registered) {
        std::atexit(cleanup_backends);
        registered = true;
    }
}

void cleanup_backends() {
    for (auto& [name, entry] : g_cache) {
        if (entry.backend) {
            auto destroy = reinterpret_cast<void (*)(IBackend*)>(
                dlsym(entry.dl_handle, "trtmc_destroy_backend"));
            if (destroy)
                destroy(entry.backend);
            entry.backend = nullptr;
        }
        if (entry.dl_handle) {
            dlclose(entry.dl_handle);
            entry.dl_handle = nullptr;
        }
    }
    for (auto& [path, handle] : g_preloaded_dependencies) {
        if (handle)
            dlclose(handle);
    }
    g_preloaded_dependencies.clear();
}

void append_load_error(std::string& tried, const std::string& label) {
    const char* error = dlerror();
    tried += "  " + label + ": " + (error ? error : "unknown dlopen error") + "\n";
}

void* try_open_backend_dso(const std::string& path, const std::string& label, std::string& tried) {
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        append_load_error(tried, label);
    }
    return handle;
}

std::string join_path(const std::string& dir, const std::string& dso_name) {
    if (dir.empty()) {
        return dso_name;
    }
    if (dir.back() == '/') {
        return dir + dso_name;
    }
    return dir + "/" + dso_name;
}

void append_python_package_backend_dirs(const fs::path& root, std::vector<std::string>& dirs) {
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

std::vector<std::string> installed_package_backend_dirs() {
    std::vector<std::string> dirs;
    const std::string bin_dir = exe_dir();
    if (bin_dir.empty())
        return dirs;

    const fs::path exe_bin_dir(bin_dir);
    if (exe_bin_dir.filename() == "bin")
        append_python_package_backend_dirs(exe_bin_dir.parent_path(), dirs);
    return dirs;
}

void* open_backend_dso(const std::string& dso_name, const std::vector<std::string>& search_dirs,
                       std::string& tried) {
    const std::string exe_path = exe_dir();
    if (!exe_path.empty()) {
        void* handle =
            try_open_backend_dso(exe_path + "/" + dso_name, exe_path + "/" + dso_name, tried);
        if (handle) {
            return handle;
        }
    }

    for (const std::string& dir : installed_package_backend_dirs()) {
        const std::string path = join_path(dir, dso_name);
        void* handle = try_open_backend_dso(path, path, tried);
        if (handle) {
            return handle;
        }
    }

    for (const std::string& dir : search_dirs) {
        if (dir.empty()) {
            continue;
        }
        const std::string path = join_path(dir, dso_name);
        void* handle = try_open_backend_dso(path, path, tried);
        if (handle) {
            return handle;
        }
    }

    return try_open_backend_dso(dso_name, dso_name + " (default)", tried);
}

const char* optional_string_symbol(void* handle, const char* symbol) {
    dlerror();
    auto fn = reinterpret_cast<const char* (*)()>(dlsym(handle, symbol));
    if (!fn) {
        return "";
    }
    const char* value = fn();
    return value ? value : "";
}

std::string backend_abi_contract_mismatch_impl(const BackendDsoAbiContractV2& actual) {
    const BackendDsoAbiContractV2 expected = make_runtime_memory_backend_dso_abi_contract_v2(0);

#define TRTMC_CHECK_BACKEND_ABI_FIELD(field)                                                       \
    if (actual.field != expected.field) {                                                          \
        return std::string(#field) + " (core=" + std::to_string(expected.field) +                  \
               ", backend=" + std::to_string(actual.field) + ")";                                  \
    }

    // struct_size is checked first so a future query implementation may fill
    // only its own shorter prefix without the loader interpreting missing
    // fields as a compatible current contract.
    TRTMC_CHECK_BACKEND_ABI_FIELD(struct_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(contract_version)
    TRTMC_CHECK_BACKEND_ABI_FIELD(interface_fingerprint)
    TRTMC_CHECK_BACKEND_ABI_FIELD(runtime_memory_layout_fingerprint)
    TRTMC_CHECK_BACKEND_ABI_FIELD(cxx_standard)
    TRTMC_CHECK_BACKEND_ABI_FIELD(compiler_id)
    TRTMC_CHECK_BACKEND_ABI_FIELD(compiler_version)
    TRTMC_CHECK_BACKEND_ABI_FIELD(cxx_abi_version)
    TRTMC_CHECK_BACKEND_ABI_FIELD(stdlib_id)
    TRTMC_CHECK_BACKEND_ABI_FIELD(stdlib_version)
    TRTMC_CHECK_BACKEND_ABI_FIELD(cxx11_string_abi)
    TRTMC_CHECK_BACKEND_ABI_FIELD(pointer_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(size_t_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(std_string_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(std_string_alignment)
    TRTMC_CHECK_BACKEND_ABI_FIELD(std_vector_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(std_vector_alignment)
    TRTMC_CHECK_BACKEND_ABI_FIELD(std_shared_ptr_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(std_shared_ptr_alignment)
    TRTMC_CHECK_BACKEND_ABI_FIELD(std_unique_ptr_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(std_unique_ptr_alignment)
    TRTMC_CHECK_BACKEND_ABI_FIELD(dtype_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(tensor_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(tensor_alignment)
    TRTMC_CHECK_BACKEND_ABI_FIELD(tensor_map_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(tensor_map_alignment)
    TRTMC_CHECK_BACKEND_ABI_FIELD(device_tensor_map_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(device_tensor_map_alignment)
    TRTMC_CHECK_BACKEND_ABI_FIELD(tensor_info_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(tensor_info_alignment)
    TRTMC_CHECK_BACKEND_ABI_FIELD(module_create_options_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(module_create_options_alignment)
    TRTMC_CHECK_BACKEND_ABI_FIELD(backend_dual_profile_modules_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(backend_profile_module_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(backend_profile_modules_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(backend_context_modules_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(i_trt_module_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(i_trt_module_alignment)
    TRTMC_CHECK_BACKEND_ABI_FIELD(i_backend_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(i_backend_alignment)
    TRTMC_CHECK_BACKEND_ABI_FIELD(runtime_memory_api_version)
    TRTMC_CHECK_BACKEND_ABI_FIELD(runtime_memory_binding_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(runtime_memory_shape_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(runtime_memory_alias_shape_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(runtime_input_shape_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(runtime_memory_alias_pair_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(runtime_memory_alias_binding_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(runtime_memory_module_options_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(runtime_memory_context_requirement_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(runtime_memory_context_block_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(runtime_memory_engine_stats_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(runtime_memory_transfer_counter_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(runtime_memory_transfer_snapshot_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(runtime_memory_module_interface_size)
    TRTMC_CHECK_BACKEND_ABI_FIELD(runtime_memory_backend_interface_size)

#undef TRTMC_CHECK_BACKEND_ABI_FIELD

    if ((actual.capability_flags & ~kBackendDsoKnownCapabilitiesV2) != 0) {
        return "capability_flags contains unknown bits (backend=" +
               std::to_string(actual.capability_flags) + ")";
    }
    return {};
}

BackendDsoAbiContractV2 query_backend_abi_contract(const std::string& dso_name, void* handle) {
    dlerror();
    auto query =
        reinterpret_cast<BackendDsoAbiQueryFnV2>(dlsym(handle, kBackendDsoAbiQuerySymbolV2));
    const char* query_error = dlerror();
    if (query_error != nullptr || query == nullptr) {
        const std::string detail =
            query_error == nullptr ? "symbol not found" : std::string(query_error);
        dlclose(handle);
        throw std::runtime_error(dso_name + " loaded but missing required " +
                                 kBackendDsoAbiQuerySymbolV2 + " symbol (" + detail +
                                 "); refusing stale backend before trtmc_create_backend()");
    }

    BackendDsoAbiContractV2 contract{};
    const std::int32_t status = query(&contract, sizeof(contract));
    if (status != 0) {
        dlclose(handle);
        throw std::runtime_error(dso_name + ": " + kBackendDsoAbiQuerySymbolV2 +
                                 "() returned status " + std::to_string(status) +
                                 "; refusing backend before trtmc_create_backend()");
    }

    const std::string mismatch = backend_dso_abi_contract_mismatch(contract);
    if (!mismatch.empty()) {
        dlclose(handle);
        throw std::runtime_error(dso_name + " backend/core ABI contract mismatch: " + mismatch +
                                 "; refusing backend before trtmc_create_backend()");
    }
    return contract;
}

CachedBackend create_backend(const std::string& requested_name, const std::string& dso_name,
                             void* handle) {
    const BackendDsoAbiContractV2 abi_contract = query_backend_abi_contract(dso_name, handle);

    auto create_fn = reinterpret_cast<IBackend* (*)()>(dlsym(handle, "trtmc_create_backend"));
    if (!create_fn) {
        dlclose(handle);
        throw std::runtime_error(dso_name + " loaded but missing trtmc_create_backend symbol");
    }

    IBackend* backend = create_fn();
    if (!backend) {
        dlclose(handle);
        throw std::runtime_error(dso_name + ": trtmc_create_backend() returned nullptr");
    }

    BackendLoadMetadata metadata;
    metadata.requested_name = requested_name;
    metadata.dso_name = dso_name;
    metadata.backend_name = backend->name() ? backend->name() : "";
    metadata.backend_abi_contract_version = abi_contract.contract_version;
    metadata.runtime_memory_backend_api_version = abi_contract.runtime_memory_api_version;
    metadata.backend_capability_flags = abi_contract.capability_flags;
    metadata.backend_interface_fingerprint = abi_contract.interface_fingerprint;
    metadata.runtime_memory_layout_fingerprint = abi_contract.runtime_memory_layout_fingerprint;
    metadata.trt_abi = optional_string_symbol(handle, "trtmc_backend_abi");
    metadata.trt_runtime_version = optional_string_symbol(handle, "trtmc_backend_runtime_version");
    metadata.runtime_memory_stack_json =
        optional_string_symbol(handle, "trtmc_backend_runtime_memory_stack_json_v1");

    return CachedBackend{handle, backend, std::move(metadata)};
}

std::string backend_dso_name(const std::string& backend_name) {
    return "libtrtmc_backend_" + backend_name + ".so";
}

void populate_load_outputs(const std::string& backend_name,
                           const BackendLoadMetadata& cached_metadata,
                           std::string* loaded_backend_name, BackendLoadMetadata* metadata) {
    if (loaded_backend_name)
        *loaded_backend_name = backend_name;
    if (metadata)
        *metadata = cached_metadata;
}

IBackend* load_cached_backend(const std::string& backend_name, std::string* loaded_backend_name,
                              BackendLoadMetadata* metadata) {
    auto it = g_cache.find(backend_name);
    if (it == g_cache.end())
        return nullptr;

    populate_load_outputs(backend_name, it->second.metadata, loaded_backend_name, metadata);
    return it->second.backend;
}

IBackend* load_backend_candidate(const std::string& backend_name,
                                 const std::vector<std::string>& search_dirs,
                                 std::string& all_tried, std::string* loaded_backend_name,
                                 BackendLoadMetadata* metadata) {
    const std::string dso_name = backend_dso_name(backend_name);
    std::string tried;
    void* handle = open_backend_dso(dso_name, search_dirs, tried);
    if (!handle) {
        all_tried += "Candidate \"" + backend_name + "\" (" + dso_name + "):\n" + tried;
        return nullptr;
    }

    CachedBackend entry = create_backend(backend_name, dso_name, handle);
    IBackend* backend = entry.backend;
    g_cache[backend_name] = entry;
    populate_load_outputs(backend_name, g_cache[backend_name].metadata, loaded_backend_name,
                          metadata);
    std::cerr << "[trtmc] Backend loaded: " << backend->name() << " (" << dso_name << ")"
              << std::endl;
    return backend;
}

std::string join_backend_names(const std::vector<std::string>& backend_names) {
    std::string names;
    for (const auto& name : backend_names) {
        if (!names.empty())
            names += ", ";
        names += name;
    }
    return names;
}

[[noreturn]] void throw_backend_load_failure(const std::vector<std::string>& backend_names,
                                             const std::string& all_tried) {
    if (backend_names.size() == 1) {
        const std::string& backend_name = backend_names.front();
        const std::string dso_name = backend_dso_name(backend_name);
        throw std::runtime_error("Backend \"" + backend_name +
                                 "\" not available.\n"
                                 "Could not load " +
                                 dso_name + ":\n" + all_tried +
                                 "\n"
                                 "To use " +
                                 backend_name + " bundles, ensure " + dso_name +
                                 " is next to the trtmc binary,\n"
                                 "inside the installed tensorrt_model_connect/bin package "
                                 "directory,\n"
                                 "in a LoadOptions::backend_search_paths / --backend-dir "
                                 "directory, or in LD_LIBRARY_PATH.");
    }

    throw std::runtime_error("No compatible backend DSO available for candidates: " +
                             join_backend_names(backend_names) + ".\n" + all_tried +
                             "\nEnsure the matching libtrtmc_backend_<backend>.so is next to the "
                             "trtmc binary, inside the installed tensorrt_model_connect/bin "
                             "package directory, in a LoadOptions::backend_search_paths / "
                             "--backend-dir directory, or in LD_LIBRARY_PATH.");
}

} // namespace

std::string backend_dso_abi_contract_mismatch(const BackendDsoAbiContractV2& actual) {
    return backend_abi_contract_mismatch_impl(actual);
}

IBackend* BackendLoader::load(const std::string& backend_name) {
    return load(backend_name, {});
}

IBackend* BackendLoader::load(const std::string& backend_name,
                              const std::vector<std::string>& search_dirs) {
    return load_first_available({backend_name}, search_dirs);
}

void BackendLoader::preload_dependency(const std::string& path) {
    if (path.empty())
        return;

    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_preloaded_dependencies.find(path);
    if (it != g_preloaded_dependencies.end())
        return;

    register_cleanup_once();

    dlerror();
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        const char* error = dlerror();
        throw std::runtime_error("Failed to preload dependency " + path + ": " +
                                 (error ? error : "unknown dlopen error"));
    }
    g_preloaded_dependencies[path] = handle;
    std::cerr << "[trtmc] Preloaded dependency: " << path << std::endl;
}

IBackend* BackendLoader::load_first_available(const std::vector<std::string>& backend_names,
                                              const std::vector<std::string>& search_dirs,
                                              std::string* loaded_backend_name,
                                              BackendLoadMetadata* metadata) {
    std::lock_guard<std::mutex> lock(g_mu);

    register_cleanup_once();

    std::string all_tried;
    for (const std::string& backend_name : backend_names) {
        if (IBackend* cached = load_cached_backend(backend_name, loaded_backend_name, metadata))
            return cached;
        if (IBackend* backend = load_backend_candidate(backend_name, search_dirs, all_tried,
                                                       loaded_backend_name, metadata))
            return backend;
    }

    throw_backend_load_failure(backend_names, all_tried);
}

} // namespace trtmc

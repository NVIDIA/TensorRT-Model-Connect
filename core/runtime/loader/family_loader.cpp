/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/family_loader.h"

#include "runtime/bundle/bundle_format.h"
#include "trtmc/runtime/family_factory.h"
#include "trtmc/runtime/plugin_abi.h"
#include "trtmc/runtime/runtime_root.h"
#include "trtmc/runtime/trt_backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

namespace fs = std::filesystem;

using CreateBackendFn = IBackend* (*)();
using DestroyBackendFn = void (*)(IBackend*);

std::string backend_library_name(const std::string& backend_id) {
    return "libtrtmc_backend_" + backend_id + ".so";
}

std::string family_library_name(const std::string& family_id) {
    return "libtrtmc_model_" + family_id + ".so";
}

std::string runtime_extension_library_name(const std::string& extension_id) {
    return "libtrtmc_byok_" + extension_id + ".so";
}

const char* plugin_kind_name(PluginKind kind) {
    switch (kind) {
    case PluginKind::kBackend:
        return "backend";
    case PluginKind::kFamily:
        return "family";
    case PluginKind::kRuntimeExtension:
        return "runtime extension";
    }
    return "unknown";
}

bool is_safe_id(const std::string& value) {
    if (value.empty() || value.front() < 'a' || value.front() > 'z')
        return false;
    for (const unsigned char character : value) {
        if ((character >= 'a' && character <= 'z') || (character >= '0' && character <= '9') ||
            character == '_') {
            continue;
        }
        return false;
    }
    return true;
}

void require_safe_id(const std::string& field, const std::string& value) {
    if (!is_safe_id(value)) {
        throw std::runtime_error("Bundle " + field + " must match [a-z][a-z0-9_]*: '" + value +
                                 "'");
    }
}

fs::path explicit_runtime_root(const std::string& runtime_root) {
    if (runtime_root.empty())
        throw std::invalid_argument("runtime_root must be explicit and non-empty");
    std::error_code error;
    fs::path root = fs::absolute(fs::path(runtime_root), error);
    if (error)
        throw std::runtime_error("Unable to resolve runtime_root '" + runtime_root +
                                 "': " + error.message());
    return root.lexically_normal();
}

fs::path loaded_library_path(const void* symbol) {
    Dl_info info{};
    if (dladdr(symbol, &info) == 0 || info.dli_fname == nullptr)
        return {};
    std::error_code error;
    fs::path path = fs::absolute(info.dli_fname, error);
    if (error)
        return info.dli_fname;
    fs::path normalized = fs::weakly_canonical(path, error);
    return error ? path.lexically_normal() : normalized;
}

bool contains_root_local_library(const fs::path& root, const std::string& library) {
    std::error_code error;
    const fs::path resolved = fs::canonical(root / library, error);
    return !error && resolved.parent_path() == root && fs::is_regular_file(resolved, error) &&
           !error;
}

void require_matching_build(const std::string& path, const PluginDescriptorV1& descriptor) {
    if (descriptor.build_id != nullptr && std::string(descriptor.build_id) == kPluginBuildId)
        return;
    const std::string actual = descriptor.build_id != nullptr ? descriptor.build_id : "<null>";
    throw std::runtime_error("Library '" + path + "' belongs to product build '" + actual +
                             "'; active runtime requires '" + kPluginBuildId + "'");
}

void require_matching_core_build() {
    const char* core_build_id = trtmc_core_build_id();
    if (core_build_id != nullptr && std::string(core_build_id) == kPluginBuildId)
        return;
    const std::string actual = core_build_id != nullptr ? core_build_id : "<null>";
    throw std::runtime_error("Active libtrtmc_core.so belongs to product build '" + actual +
                             "'; libtrtmc_runtime.so requires '" + kPluginBuildId + "'");
}

class SharedLibrary {
  public:
    explicit SharedLibrary(const fs::path& path) : path_(path.string()) {
        dlerror();
        handle_ = dlopen(path_.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
        if (handle_ == nullptr) {
            const char* error = dlerror();
            throw std::runtime_error("Unable to load '" + path_ +
                                     "': " + (error != nullptr ? error : "unknown dlopen error"));
        }
    }

    SharedLibrary(const SharedLibrary&) = delete;
    SharedLibrary& operator=(const SharedLibrary&) = delete;

    ~SharedLibrary() {
        if (handle_ != nullptr)
            dlclose(handle_);
    }

    void* require_symbol(const char* name) const {
        dlerror();
        void* symbol = dlsym(handle_, name);
        const char* error = dlerror();
        if (error != nullptr || symbol == nullptr) {
            throw std::runtime_error("Library '" + path_ + "' is missing required symbol '" + name +
                                     "'");
        }
        return symbol;
    }

    void require_plugin(PluginKind expected_kind, const std::string& expected_id) const {
        const auto descriptor_function =
            reinterpret_cast<PluginDescriptorFn>(require_symbol(kPluginDescriptorSymbol));
        const PluginDescriptorV1* descriptor = descriptor_function();
        if (descriptor == nullptr) {
            throw std::runtime_error("Library '" + path_ + "' returned a null plugin descriptor");
        }
        if (descriptor->struct_size != sizeof(PluginDescriptorV1)) {
            throw std::runtime_error("Library '" + path_ + "' declares descriptor size " +
                                     std::to_string(descriptor->struct_size) + "; expected " +
                                     std::to_string(sizeof(PluginDescriptorV1)));
        }
        if (descriptor->descriptor_version != kPluginDescriptorVersion) {
            throw std::runtime_error("Library '" + path_ + "' declares plugin descriptor version " +
                                     std::to_string(descriptor->descriptor_version) +
                                     "; expected " + std::to_string(kPluginDescriptorVersion));
        }
        if (descriptor->kind != expected_kind) {
            throw std::runtime_error("Library '" + path_ + "' declares a " +
                                     plugin_kind_name(descriptor->kind) + " plugin; expected " +
                                     plugin_kind_name(expected_kind));
        }
        if (descriptor->id == nullptr || expected_id != descriptor->id) {
            const std::string actual = descriptor->id != nullptr ? descriptor->id : "<null>";
            throw std::runtime_error("Library '" + path_ + "' declares " +
                                     plugin_kind_name(expected_kind) + " '" + actual +
                                     "'; expected '" + expected_id + "'");
        }
        require_matching_build(path_, *descriptor);
    }

  private:
    std::string path_;
    void* handle_{nullptr};
};

class BackendLibrary {
  public:
    BackendLibrary(const fs::path& runtime_root, const std::string& backend_id)
        : library_(runtime_root / backend_library_name(backend_id)) {
        library_.require_plugin(PluginKind::kBackend, backend_id);
        const auto create =
            reinterpret_cast<CreateBackendFn>(library_.require_symbol("trtmc_create_backend"));
        destroy_ =
            reinterpret_cast<DestroyBackendFn>(library_.require_symbol("trtmc_destroy_backend"));
        backend_ = create();
        if (backend_ == nullptr)
            throw std::runtime_error("trtmc_create_backend returned nullptr");

        const char* actual_name = backend_->name();
        if (actual_name == nullptr || backend_id != actual_name) {
            const std::string actual = actual_name != nullptr ? actual_name : "<null>";
            destroy_(backend_);
            backend_ = nullptr;
            throw std::runtime_error("Backend identity mismatch: bundle requested '" + backend_id +
                                     "' but DSO created '" + actual + "'");
        }
    }

    BackendLibrary(const BackendLibrary&) = delete;
    BackendLibrary& operator=(const BackendLibrary&) = delete;

    ~BackendLibrary() {
        if (backend_ != nullptr)
            destroy_(backend_);
    }

    IBackend& get() const { return *backend_; }

  private:
    SharedLibrary library_;
    IBackend* backend_{nullptr};
    DestroyBackendFn destroy_{nullptr};
};

class FamilyLibrary {
  public:
    FamilyLibrary(const fs::path& runtime_root, const std::string& family_id)
        : library_(runtime_root / family_library_name(family_id)) {
        library_.require_plugin(PluginKind::kFamily, family_id);
        create_ = reinterpret_cast<CreateFamilyFn>(library_.require_symbol(kCreateFamilySymbol));
    }

    FamilyLibrary(const FamilyLibrary&) = delete;
    FamilyLibrary& operator=(const FamilyLibrary&) = delete;

    ITask* create(const FamilyContext& context) const { return create_(context); }

  private:
    SharedLibrary library_;
    CreateFamilyFn create_{nullptr};
};

class RuntimeExtensionLibrary {
  public:
    explicit RuntimeExtensionLibrary(const fs::path& runtime_root)
        : library_(runtime_root / runtime_extension_library_name("tvm_ffi")) {
        library_.require_plugin(PluginKind::kRuntimeExtension, "tvm_ffi");
        load_ = reinterpret_cast<LoadKernelFn>(library_.require_symbol("trtmc_load_byok_kernel"));
    }

    void load(const std::string& library, const std::string& function,
              const std::string& kernel_name) const {
        if (const char* error = load_(library.c_str(), function.c_str(), kernel_name.c_str()))
            throw std::runtime_error(error);
    }

  private:
    using LoadKernelFn = const char* (*)(const char*, const char*, const char*) noexcept;

    SharedLibrary library_;
    LoadKernelFn load_{nullptr};
};

class RuntimeOptionsBackend final : public IBackend {
  public:
    RuntimeOptionsBackend(IBackend& backend, std::string runtime_cache_path, bool cuda_graphs)
        : backend_(backend), runtime_cache_path_(std::move(runtime_cache_path)),
          cuda_graphs_(cuda_graphs) {}

    std::unique_ptr<ITrtModule> create_module(const void* plan_data, size_t plan_size,
                                              const ModuleCreateOptions& options) override {
        return backend_.create_module(plan_data, plan_size, with_runtime_options(options));
    }

    std::unique_ptr<ITrtModule>
    create_module_prebound(const void* plan_data, size_t plan_size,
                           const ModuleCreateOptions& options,
                           const std::vector<ModuleExternalBinding>& bindings) override {
        return backend_.create_module_prebound(plan_data, plan_size, with_runtime_options(options),
                                               bindings);
    }

    BackendDualProfileModules
    create_dual_profile_modules(const void* plan_data, size_t plan_size,
                                const ModuleCreateOptions& options) override {
        return backend_.create_dual_profile_modules(plan_data, plan_size,
                                                    with_runtime_options(options));
    }

    const char* name() const override { return backend_.name(); }

  private:
    ModuleCreateOptions with_runtime_options(ModuleCreateOptions options) const {
        options.runtime_cache_path = runtime_cache_path_.c_str();
        options.cuda_graphs = cuda_graphs_;
        return options;
    }

    IBackend& backend_;
    std::string runtime_cache_path_;
    bool cuda_graphs_;
};

struct ConfiguredBackendKey {
    IBackend* backend{nullptr};
    std::string runtime_cache_path;
    bool cuda_graphs{false};

    bool operator==(const ConfiguredBackendKey& other) const noexcept {
        return backend == other.backend && runtime_cache_path == other.runtime_cache_path &&
               cuda_graphs == other.cuda_graphs;
    }
};

struct ConfiguredBackendKeyHash {
    std::size_t operator()(const ConfiguredBackendKey& key) const noexcept {
        std::size_t value = std::hash<IBackend*>{}(key.backend);
        value ^= std::hash<std::string>{}(key.runtime_cache_path) + 0x9e3779b9U + (value << 6U) +
                 (value >> 2U);
        value ^= std::hash<bool>{}(key.cuda_graphs) + 0x9e3779b9U + (value << 6U) + (value >> 2U);
        return value;
    }
};

struct RuntimeLibraryCache {
    std::mutex mutex;
    std::unordered_map<std::string, std::unique_ptr<BackendLibrary>> backends;
    std::unordered_map<std::string, std::unique_ptr<FamilyLibrary>> families;
    std::unordered_map<std::string, std::unique_ptr<RuntimeExtensionLibrary>> extensions;
    std::unordered_map<ConfiguredBackendKey, std::unique_ptr<RuntimeOptionsBackend>,
                       ConfiguredBackendKeyHash>
        configured_backends;
};

RuntimeLibraryCache& runtime_library_cache() {
    // The cache deliberately lives until process exit. Family tasks may defer
    // module creation, so their code, backend, and immutable runtime-options
    // adapter must never be unloaded underneath them.
    static RuntimeLibraryCache* cache = new RuntimeLibraryCache();
    return *cache;
}

IBackend& cached_backend(const fs::path& runtime_root, const std::string& backend_id) {
    const std::string path = (runtime_root / backend_library_name(backend_id)).string();
    auto& cache = runtime_library_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto found = cache.backends.find(path);
    if (found != cache.backends.end())
        return found->second->get();

    auto library = std::make_unique<BackendLibrary>(runtime_root, backend_id);
    IBackend& backend = library->get();
    cache.backends.emplace(path, std::move(library));
    return backend;
}

IBackend& cached_configured_backend(IBackend& backend, const std::string& runtime_cache_path,
                                    bool cuda_graphs) {
    ConfiguredBackendKey key{&backend, runtime_cache_path, cuda_graphs};
    auto& cache = runtime_library_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto found = cache.configured_backends.find(key);
    if (found != cache.configured_backends.end())
        return *found->second;

    auto configured =
        std::make_unique<RuntimeOptionsBackend>(backend, runtime_cache_path, cuda_graphs);
    IBackend& result = *configured;
    cache.configured_backends.emplace(std::move(key), std::move(configured));
    return result;
}

FamilyLibrary& cached_family(const fs::path& runtime_root, const std::string& family_id) {
    const std::string path = (runtime_root / family_library_name(family_id)).string();
    auto& cache = runtime_library_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto found = cache.families.find(path);
    if (found != cache.families.end())
        return *found->second;

    auto library = std::make_unique<FamilyLibrary>(runtime_root, family_id);
    FamilyLibrary& family = *library;
    cache.families.emplace(path, std::move(library));
    return family;
}

RuntimeExtensionLibrary& cached_runtime_extension(const fs::path& runtime_root) {
    const std::string path = (runtime_root / runtime_extension_library_name("tvm_ffi")).string();
    auto& cache = runtime_library_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto found = cache.extensions.find(path);
    if (found != cache.extensions.end())
        return *found->second;

    auto library = std::make_unique<RuntimeExtensionLibrary>(runtime_root);
    RuntimeExtensionLibrary& extension = *library;
    cache.extensions.emplace(path, std::move(library));
    return extension;
}

void require_matching_task(const BundleInfo& info, const ITask& task) {
    const char* actual = task.task();
    if (actual == nullptr || info.task != actual) {
        throw std::runtime_error("Family factory task mismatch: bundle declares '" + info.task +
                                 "' but factory returned '" +
                                 (actual != nullptr ? std::string(actual) : std::string("<null>")) +
                                 "'");
    }
}

} // namespace

std::string loaded_runtime_root() {
    using LoadTaskFn = std::unique_ptr<ITask> (*)(const std::string&, const std::string&,
                                                  std::uint64_t, const std::string&, bool);
    const auto load_task_function = static_cast<LoadTaskFn>(&load_task);
    const fs::path runtime_library =
        loaded_library_path(reinterpret_cast<const void*>(load_task_function));
    if (runtime_library.empty())
        throw std::runtime_error("Unable to locate the active TRTMC runtime loader");
    return runtime_library.parent_path().string();
}

bool runtime_root_contains_bundle(const BundleInfo& bundle, const std::string& runtime_root,
                                  bool require_byok) {
    require_matching_core_build();
    require_safe_id("family", bundle.family);
    require_safe_id("backend", bundle.backend);
    if (runtime_root.empty())
        return false;

    std::error_code error;
    fs::path root = fs::canonical(fs::absolute(runtime_root, error), error);
    if (error || !fs::is_directory(root, error) || error)
        return false;

    std::vector<std::string> required{
        backend_library_name(bundle.backend),
        family_library_name(bundle.family),
    };
    if (require_byok)
        required.emplace_back(runtime_extension_library_name("tvm_ffi"));

    return std::all_of(required.begin(), required.end(), [&](const std::string& library) {
        return contains_root_local_library(root, library);
    });
}

void load_byok_kernel_from_runtime(const std::string& runtime_root, const std::string& library,
                                   const std::string& function, const std::string& kernel_name) {
    require_matching_core_build();
    RuntimeExtensionLibrary& extension =
        cached_runtime_extension(explicit_runtime_root(runtime_root));
    extension.load(library, function, kernel_name);
}

std::unique_ptr<ITask> load_task(const std::string& bundle_path, const std::string& runtime_root,
                                 std::uint64_t kv_cache_size_bytes,
                                 const std::string& runtime_cache_path, bool cuda_graphs) {
    require_matching_core_build();
    const BundleReader reader(bundle_path);
    const BundleInfo& info = reader.info();
    require_safe_id("family", info.family);
    require_safe_id("task", info.task);
    require_safe_id("backend", info.backend);
    if ((!runtime_cache_path.empty() || cuda_graphs) && info.backend != "trt_rtx") {
        throw std::invalid_argument(
            "runtime cache and whole-graph capture require a TensorRT-RTX bundle");
    }

    const fs::path root = explicit_runtime_root(runtime_root);
    IBackend& backend = cached_backend(root, info.backend);
    FamilyLibrary& family = cached_family(root, info.family);
    IBackend& configured_backend =
        cached_configured_backend(backend, runtime_cache_path, cuda_graphs);
    FamilyContext context{reader, configured_backend, kv_cache_size_bytes};
    std::unique_ptr<ITask> task(family.create(context));
    if (task == nullptr)
        throw std::runtime_error("trtmc_create_family returned nullptr");
    require_matching_task(info, *task);
    return task;
}

} // namespace trtmc

extern "C" const char* trtmc_runtime_build_id() noexcept {
    return trtmc::kPluginBuildId;
}
